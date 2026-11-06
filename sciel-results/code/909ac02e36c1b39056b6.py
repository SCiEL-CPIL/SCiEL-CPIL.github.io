"""XOR-circuit synthesis via Steiner tree (Prim's MST) over the Boolean hypercube."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """
    Verify that the given gates and outputs correctly compute matrix @ x over GF(2).
    matrix: shape (m, n) binary. Input wires 0..n-1. Constant-zero wire n.
    Gate k produces wire n + 1 + k. gates is list of [wire_a, wire_b].
    outputs: list of wire indices, one per output row.
    Returns True if valid and depth <= depth_limit.
    """
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

    # Check depth
    if depth_limit is not None:
        depth = [0] * total_wires
        for k in range(num_gates):
            a, b = gates[k]
            if a < 0 or a >= total_wires or b < 0 or b >= total_wires:
                return False
            if a >= n + 1 + k or b >= n + 1 + k:
                return False
            depth[n + 1 + k] = 1 + max(depth[a], depth[b])
        max_depth = max(depth[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
        for wire in outputs:
            if wire < 0 or wire >= total_wires:
                return False
            max_depth = max(max_depth, depth[wire])
        if max_depth > depth_limit:
            return False

    # Symbolic verification
    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1

    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2

    for i in range(m):
        wire = outputs[i]
        if wire < 0 or wire >= total_wires:
            return False
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False

    return True

def _baseline_synthesis(matrix, depth_limit=16):
    """
    Baseline synthesis: independent XOR chains for each row.
    """
    m, n = matrix.shape
    gates = []
    outputs = []

    for i in range(m):
        row = matrix[i]
        positions = np.where(row)[0].tolist()
        if len(positions) == 0:
            outputs.append(n)
        elif len(positions) == 1:
            outputs.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                a = current
                b = positions[j]
                gate_idx = len(gates)
                new_wire = n + 1 + gate_idx
                gates.append([a, b])
                current = new_wire
            outputs.append(current)

    return {'gates': gates, 'outputs': outputs}

def _steiner_tree_synthesis(matrix, depth_limit=16, time_budget=15.0):
    """
    Steiner tree synthesis: treat zero vector and target rows as terminals in the
    Boolean hypercube. Use Prim's MST over the complete graph of reachable vectors
    to find a minimal set of intermediate XOR nodes.
    
    The key insight: in the XOR hypercube, the distance between two vectors a and b
    is popcount(a XOR b). However, for our purposes, each XOR gate connects two
    existing wires to produce a new wire. We model this as: we start with all input
    wires (unit vectors) and the zero wire. We want to reach all target rows.
    
    We use a Prim-like algorithm: start from the set of "available" wires (inputs + zero),
    and iteratively add the cheapest XOR combination that either:
    1. Produces a target row, or
    2. Produces an intermediate vector that will help produce future targets.
    
    To decide "helpfulness," we use a heuristic: prefer vectors that are close (in Hamming
    distance) to multiple unmet targets.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget

    # Identify unique target vectors
    target_map = {}  # vector_bytes -> list of row indices
    for i in range(m):
        v = matrix[i].tobytes()
        if v not in target_map:
            target_map[v] = []
        target_map[v].append(i)

    num_targets = len(target_map)
    if num_targets == 0:
        return {'gates': [], 'outputs': [n] * m}

    # Initialize available wires
    # wire_vec: dict wire_index -> vector (np array)
    # vec_to_wire: dict vector_bytes -> wire_index
    wire_vec = {}
    vec_to_wire = {}

    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec[i] = v
        vec_to_wire[v.tobytes()] = i

    zero_vec = np.zeros(n, dtype=np.int64)
    wire_vec[n] = zero_vec
    vec_to_wire[zero_vec.tobytes()] = n

    # Track which targets are met
    met_targets = set()
    for v_bytes in target_map:
        if v_bytes in vec_to_wire:
            met_targets.add(v_bytes)

    gates = []
    outputs = [None] * m

    # Assign outputs for already-met targets
    for v_bytes, row_indices in target_map.items():
        if v_bytes in met_targets:
            wire = vec_to_wire[v_bytes]
            for idx in row_indices:
                outputs[idx] = wire

    if len(met_targets) == num_targets:
        return {'gates': gates, 'outputs': outputs}

    # Prim-like expansion: maintain a priority queue of candidate XOR pairs
    # that could be added next. Each candidate is (cost, wire_a, wire_b, resulting_vec).
    # We use a heuristic priority: cost + penalty_for_not_helping_targets.

    # Precompute target vectors as numpy arrays for distance computation
    target_vecs = {}
    for v_bytes in target_map:
        if v_bytes not in met_targets:
            # Reconstruct from matrix
            # Find first row index
            first_idx = target_map[v_bytes][0]
            target_vecs[v_bytes] = matrix[first_idx].copy()

    # Initial candidate generation: all pairs of available wires
    available_wires = list(wire_vec.keys())
    candidate_heap = []  # (priority, counter, wire_a, wire_b, result_vec_bytes)
    counter = 0

    def compute_priority(result_vec, unmet_target_vecs):
        """
        Compute priority for adding result_vec.
        Lower is better. We want vectors that are close to unmet targets.
        """
        # Cost is always 1 (one XOR gate)
        cost = 1
        # Heuristic: sum of Hamming distances to all unmet targets (lower = closer)
        # But this can be expensive. Use a sampled subset for large cases.
        unmet_list = list(unmet_target_vecs.values())
        if len(unmet_list) > 20:
            # Sample up to 20 targets
            sample = random.sample(unmet_list, 20)
        else:
            sample = unmet_list

        total_dist = 0
        for tv in sample:
            dist = int(np.sum((result_vec + tv) % 2))  # Hamming distance
            total_dist += dist
        avg_dist = total_dist / len(sample) if sample else 0

        # Priority: cost + alpha * avg_dist
        # We want to prefer vectors that are close to targets
        return cost + 0.1 * avg_dist

    # Generate initial candidates
    for i in range(len(available_wires)):
        for j in range(i + 1, len(available_wires)):
            a_wire = available_wires[i]
            b_wire = available_wires[j]
            a_vec = wire_vec[a_wire]
            b_vec = wire_vec[b_wire]
            result_vec = (a_vec + b_vec) % 2
            result_bytes = result_vec.tobytes()

            # Skip if already available
            if result_bytes in vec_to_wire:
                continue

            # Check if this result is a target
            if result_bytes in target_map and result_bytes not in met_targets:
                priority = 0  # Highest priority: directly meets a target
            else:
                priority = compute_priority(result_vec, target_vecs)

            heapq.heappush(candidate_heap, (priority, counter, a_wire, b_wire, result_bytes, result_vec))
            counter += 1

    # Prim-like expansion
    max_iterations = 5000  # Safety limit
    iteration = 0

    while len(met_targets) < num_targets and candidate_heap and iteration < max_iterations:
        if time.time() > deadline:
            break

        # Pop the best candidate
        priority, _, a_wire, b_wire, result_bytes, result_vec = heapq.heappop(candidate_heap)

        # Check if this vector is already available (might have been added via another path)
        if result_bytes in vec_to_wire:
            continue

        # Check if this candidate is still valid (a_wire and b_wire still exist)
        if a_wire not in wire_vec or b_wire not in wire_vec:
            continue

        # Add this gate
        gate_idx = len(gates)
        new_wire = n + 1 + gate_idx
        gates.append([a_wire, b_wire])
        wire_vec[new_wire] = result_vec
        vec_to_wire[result_bytes] = new_wire

        # Check if this meets any targets
        if result_bytes in target_map and result_bytes not in met_targets:
            met_targets.add(result_bytes)
            for idx in target_map[result_bytes]:
                outputs[idx] = new_wire
            # Remove from target_vecs
            if result_bytes in target_vecs:
                del target_vecs[result_bytes]

        # Generate new candidates involving the new wire
        for other_wire in wire_vec.keys():
            if other_wire == new_wire:
                continue
            other_vec = wire_vec[other_wire]
            new_result = (result_vec + other_vec) % 2
            new_result_bytes = new_result.tobytes()

            if new_result_bytes in vec_to_wire:
                continue

            if new_result_bytes in target_map and new_result_bytes not in met_targets:
                new_priority = 0
            else:
                new_priority = compute_priority(new_result, target_vecs)

            heapq.heappush(candidate_heap, (new_priority, counter, new_wire, other_wire, new_result_bytes, new_result))
            counter += 1

        iteration += 1

    # Check if all targets are met
    if len(met_targets) < num_targets:
        # Fallback: use baseline for unmet targets
        for v_bytes, row_indices in target_map.items():
            if v_bytes not in met_targets:
                # Use baseline for this target
                first_idx = row_indices[0]
                row = matrix[first_idx]
                positions = np.where(row)[0].tolist()
                if len(positions) == 0:
                    for idx in row_indices:
                        outputs[idx] = n
                elif len(positions) == 1:
                    for idx in row_indices:
                        outputs[idx] = positions[0]
                else:
                    current = positions[0]
                    for j in range(1, len(positions)):
                        a = current
                        b = positions[j]
                        gate_idx = len(gates)
                        new_wire = n + 1 + gate_idx
                        gates.append([a, b])
                        current = new_wire
                    for idx in row_indices:
                        outputs[idx] = current
                met_targets.add(v_bytes)

    # Verify and clean up
    if all(o is not None for o in outputs):
        if _verify_solution(matrix, gates, outputs, depth_limit):
            return {'gates': gates, 'outputs': outputs}

    # Fallback to baseline
    return _baseline_synthesis(matrix, depth_limit)

def _local_search_optimize(matrix, gates, outputs, depth_limit=16, time_budget=5.0):
    """
    Local search: try to remove redundant gates and simplify the circuit.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget

    # Build wire_vec
    def build_wire_vec(gates):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec

    wire_vec = build_wire_vec(gates)

    # Build vec_to_wire
    def build_vec_to_wire(wire_vec):
        v2w = {}
        for w in range(len(wire_vec)):
            v2w[wire_vec[w].tobytes()] = w
        return v2w

    vec_to_wire = build_vec_to_wire(wire_vec)

    # Pass 1: Remove unused gates
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for k in range(len(gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for out_wire in outputs:
                if out_wire == wire_k:
                    used = True
                    break
            if not used:
                for other_gate in gates:
                    if other_gate[0] == wire_k or other_gate[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                changed = True
                wire_vec = build_wire_vec(gates)
                vec_to_wire = build_vec_to_wire(wire_vec)
                break

    # Pass 2: Replace gates with direct references if possible
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for k in range(len(gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_bytes = vec_k.tobytes()
            if vec_bytes in vec_to_wire:
                existing_wire = vec_to_wire[vec_bytes]
                if existing_wire != wire_k and existing_wire < wire_k:
                    # Replace all uses of wire_k with existing_wire
                    for j in range(k, len(gates)):
                        if gates[j][0] == wire_k:
                            gates[j][0] = existing_wire
                        if gates[j][1] == wire_k:
                            gates[j][1] = existing_wire
                    for i in range(m):
                        if outputs[i] == wire_k:
                            outputs[i] = existing_wire
                    gates.pop(k)
                    changed = True
                    wire_vec = build_wire_vec(gates)
                    vec_to_wire = build_vec_to_wire(wire_vec)
                    break

    # Pass 3: Simplify outputs
    for i in range(m):
        if time.time() > deadline:
            break
        out_wire = outputs[i]
        if out_wire >= n + 1:
            vec_out = wire_vec[out_wire]
            vec_bytes = vec_out.tobytes()
            if vec_bytes in vec_to_wire:
                existing_wire = vec_to_wire[vec_bytes]
                if existing_wire != out_wire and existing_wire < out_wire:
                    outputs[i] = existing_wire

    return {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    """
    Entry point. Returns a dictionary with 'gates' and 'outputs'.
    """
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape

    # Start with baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    baseline_gates = len(baseline['gates'])

    if baseline_gates == 0:
        return baseline

    # Time budget management
    start_time = time.time()
    total_budget = 85.0  # Safety margin before 90s kill

    best_result = baseline
    best_count = baseline_gates

    # Try Steiner tree synthesis
    try:
        steiner = _steiner_tree_synthesis(matrix, depth_limit, time_budget=40.0)
        if len(steiner['gates']) < best_count:
            best_result = steiner
            best_count = len(steiner['gates'])
    except Exception:
        pass

    # Try local search optimization on the best result so far
    elapsed = time.time() - start_time
    if elapsed < 70.0 and best_count > 0:
        try:
            remaining_budget = min(15.0, 70.0 - elapsed)
            optimized = _local_search_optimize(matrix, best_result['gates'][:], best_result['outputs'][:], depth_limit, time_budget=remaining_budget)
            if len(optimized['gates']) < best_count:
                best_result = optimized
                best_count = len(optimized['gates'])
        except Exception:
            pass

    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > 80.0:
        return best_result

    # Verify final result
    if _verify_solution(matrix, best_result['gates'], best_result['outputs'], depth_limit):
        return best_result

    # Fallback to baseline
    return baseline

# EVOLVE-BLOCK-END
