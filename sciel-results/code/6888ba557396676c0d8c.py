"""Exact XOR-circuit synthesis: MST-based global subexpression sharing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict
from heapq import heappush, heappop

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

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

    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1
    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
    for i in range(m):
        wire = outputs[i]
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    return True


def _baseline_synthesis(matrix, depth_limit=16):
    """Independent XOR chains per row."""
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
                gates.append([a, b])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    return {'gates': gates, 'outputs': outputs}


def _hamming_dist(v1, v2):
    """Hamming distance between two GF(2) vectors."""
    return int(np.sum((v1 + v2) % 2))


def _mst_synthesis(matrix, depth_limit=16, time_budget=35.0):
    """
    MST-based circuit synthesis:
    1. Collect all required vectors (targets) and input basis vectors.
    2. Build a complete graph where nodes are unique vectors, edges are XOR pairs.
    3. Edge weight = 1 if the XOR result is already in the set, else estimated cost.
    4. Use Kruskal's algorithm to find minimum spanning tree over required vectors.
    5. Convert tree to circuit gates.
    
    Key insight: In an XOR circuit, each gate produces v = a XOR b. If we view
    the set of all produced vectors as nodes, the circuit forms a directed acyclic
    graph. The minimum gate count corresponds to finding a minimum-cost way to
    generate all target vectors from inputs, which is related to the minimum
    spanning tree in the vector space.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    
    # Collect all target vectors
    targets = [matrix[i].copy() for i in range(m)]
    target_bytes = [t.tobytes() for t in targets]
    
    # Initialize with input basis vectors and zero
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))  # wire n = zero
    
    gates = []
    outputs = [None] * m
    satisfied = [False] * m
    
    vec_to_wires = defaultdict(list)
    for w in range(n + 1):
        vec_to_wires[wire_vec[w].tobytes()].append(w)
    
    # Check which targets are already satisfied
    for i in range(m):
        if target_bytes[i] in vec_to_wires:
            outputs[i] = vec_to_wires[target_bytes[i]][0]
            satisfied[i] = True
    
    unsatisfied = [i for i in range(m) if not satisfied[i]]
    if not unsatisfied:
        return {'gates': gates, 'outputs': outputs}
    
    # MST approach: Build graph over all unique vectors we need to produce
    # Nodes: all unique target vectors + input basis + zero
    # For each pair of nodes (u, v), edge exists if u XOR v can be produced cheaply
    
    # Strategy: Use a priority queue (Kruskal-like) to greedily add cheapest edges
    # that connect new target vectors to the existing circuit
    
    # Represent the "circuit" as a set of available vectors with their wire IDs
    # We'll use a best-first search: at each step, find the cheapest way to
    # produce any unsatisfied target using 1-2 existing wires
    
    # Precompute: for each unsatisfied target, find the minimum number of
    # additional gates needed to produce it from available wires
    
    # Phase 1: Try to satisfy targets with 0 or 1 additional gates
    # Phase 2: For remaining, use MST-like expansion
    
    # Build adjacency: for each available wire, what new vectors can we make?
    # Use a min-heap: (cost, target_idx, wire_a, wire_b)
    # cost = number of new gates needed
    
    # For 1-gate completion: target = a XOR b where a, b are available
    # For 2-gate completion: target = (a XOR b) XOR c where a, b, c available,
    #   and (a XOR b) is not already available
    
    W = n + 1 + len(gates)
    avail_set = set(vec_to_wires.keys())
    
    # Priority queue: (estimated_cost, target_idx, method, params)
    # method 0: direct (already satisfied)
    # method 1: 1-gate: target = wire_a XOR wire_b
    # method 2: 2-gate: target = (wire_a XOR wire_b) XOR wire_c
    
    pq = []
    
    # Check 1-gate completions for all unsatisfied targets
    for i in unsatisfied:
        target = targets[i]
        # For each available wire a, check if target XOR a is also available
        for a in range(W):
            comp = (target + wire_vec[a]) % 2
            cb = comp.tobytes()
            if cb in avail_set:
                for b in vec_to_wires[cb]:
                    if b != a:
                        heappush(pq, (1, i, 1, a, b))
    
    # Process queue
    processed_targets = set()
    
    while unsatisfied and time.time() < deadline:
        # Find the cheapest completion
        best = None
        while pq:
            cost, i, method, w1, w2 = heappop(pq)
            if i in processed_targets or satisfied[i]:
                continue
            # Verify this is still valid
            if method == 1:
                # Check w1, w2 still exist and produce target
                if w1 < W and w2 < W:
                    v = (wire_vec[w1] + wire_vec[w2]) % 2
                    if v.tobytes() == target_bytes[i]:
                        best = (cost, i, method, w1, w2)
                        break
            elif method == 2:
                # w1 XOR w2 XOR w3 should equal target
                if w1 < W and w2 < W and w3 < W:
                    v12 = (wire_vec[w1] + wire_vec[w2]) % 2
                    v = (v12 + wire_vec[w3]) % 2
                    if v.tobytes() == target_bytes[i]:
                        best = (cost, i, method, w1, w2, w3)
                        break
        if best is None:
            break
        
        cost, i, method = best[0], best[1], best[2]
        processed_targets.add(i)
        
        if method == 1:
            a, b = best[3], best[4]
            k = len(gates)
            new_wire = n + 1 + k
            gates.append([a, b])
            wire_vec.append((wire_vec[a] + wire_vec[b]) % 2)
            vec_to_wires[wire_vec[new_wire].tobytes()].append(new_wire)
            outputs[i] = new_wire
            satisfied[i] = True
            unsatisfied.remove(i)
            
            # Add new 1-gate completions enabled by this new wire
            W = n + 1 + len(gates)
            for j in unsatisfied:
                target_j = targets[j]
                comp = (target_j + wire_vec[new_wire]) % 2
                cb = comp.tobytes()
                if cb in avail_set:
                    for b2 in vec_to_wires[cb]:
                        if b2 != new_wire:
                            heappush(pq, (1, j, 1, new_wire, b2))
                # Also check if target_j itself is now available
                if target_j.tobytes() == wire_vec[new_wire].tobytes():
                    outputs[j] = new_wire
                    satisfied[j] = True
                    unsatisfied.remove(j)
                    processed_targets.add(j)
        elif method == 2:
            a, b, c = best[3], best[4], best[5]
            # First gate: a XOR b
            k1 = len(gates)
            new_wire1 = n + 1 + k1
            v12 = (wire_vec[a] + wire_vec[b]) % 2
            gates.append([a, b])
            wire_vec.append(v12)
            vb12 = v12.tobytes()
            if vb12 in vec_to_wires:
                # Already exists, reuse
                new_wire1 = vec_to_wires[vb12][0]
                gates.pop()
                wire_vec.pop()
            else:
                vec_to_wires[vb12].append(new_wire1)
            
            # Second gate: new_wire1 XOR c
            k2 = len(gates)
            new_wire2 = n + 1 + k2
            v_final = (wire_vec[new_wire1] + wire_vec[c]) % 2
            gates.append([new_wire1, c])
            wire_vec.append(v_final)
            vec_to_wires[v_final.tobytes()].append(new_wire2)
            outputs[i] = new_wire2
            satisfied[i] = True
            unsatisfied.remove(i)
            
            # Add new 1-gate completions
            W = n + 1 + len(gates)
            for j in unsatisfied:
                target_j = targets[j]
                comp = (target_j + wire_vec[new_wire2]) % 2
                cb = comp.tobytes()
                if cb in avail_set:
                    for b2 in vec_to_wires[cb]:
                        if b2 != new_wire2:
                            heappush(pq, (1, j, 1, new_wire2, b2))
                if target_j.tobytes() == v_final.tobytes():
                    outputs[j] = new_wire2
                    satisfied[j] = True
                    unsatisfied.remove(j)
                    processed_targets.add(j)
    
    # Fallback: for remaining unsatisfied, use chains
    for i in unsatisfied:
        if time.time() > deadline:
            break
        target = targets[i]
        positions = np.where(target)[0].tolist()
        if len(positions) == 0:
            outputs[i] = n
        elif len(positions) == 1:
            outputs[i] = positions[0]
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                b = positions[j]
                new_vec = (wire_vec[current] + wire_vec[b]) % 2
                nb = new_vec.tobytes()
                if nb in vec_to_wires:
                    current = vec_to_wires[nb][0]
                else:
                    k = len(gates)
                    new_w = n + 1 + k
                    gates.append([current, b])
                    wire_vec.append(new_vec)
                    vec_to_wires[nb].append(new_w)
                    current = new_w
            outputs[i] = current
    
    if all(o is not None for o in outputs) and _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return None


def _cse_polish(matrix, gates, outputs, depth_limit=16, time_budget=12.0):
    """
    Generalized Common Subexpression Elimination:
    - Identify all subexpressions (vectors) in the circuit.
    - For each pair of gates producing the same vector, redirect uses to the earlier one.
    - Remove redundant gates.
    - Repeat until no improvement.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]
    
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break
        
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        
        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w
        
        improved = False
        
        # Remove unused gates
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for ow in outputs:
                if ow == wire_k:
                    used = True
                    break
            if not used:
                for g in gates:
                    if g[0] == wire_k or g[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break
        
        if improved:
            continue
        
        # Redirect redundant gates
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k_bytes = wire_vec[wire_k].tobytes()
            earlier = vec_to_earliest.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k:
                        gates[j][0] = earlier
                    if gates[j][1] == wire_k:
                        gates[j][1] = earlier
                for i in range(m):
                    if outputs[i] == wire_k:
                        outputs[i] = earlier
                gates.pop(k)
                improved = True
                break
        
        if not improved:
            break
    
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return None


def _greedy_vector_synthesis(matrix, depth_limit=16, time_budget=20.0):
    """
    Greedy vector-space synthesis with better scoring:
    At each step, pick the XOR pair that maximizes the number of unsatisfied
    targets that become "closer" (in Hamming distance) to being satisfied.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    
    targets = [matrix[i].copy() for i in range(m)]
    target_bytes = [t.tobytes() for t in targets]
    
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))
    
    gates = []
    outputs = [None] * m
    satisfied = [False] * m
    
    vec_to_wires = defaultdict(list)
    for w in range(n + 1):
        vec_to_wires[wire_vec[w].tobytes()].append(w)
    
    for i in range(m):
        if target_bytes[i] in vec_to_wires:
            outputs[i] = vec_to_wires[target_bytes[i]][0]
            satisfied[i] = True
    
    unsatisfied = [i for i in range(m) if not satisfied[i]]
    if not unsatisfied:
        return {'gates': gates, 'outputs': outputs}
    
    # Precompute target popcounts
    target_popcounts = [int(np.sum(targets[i])) for i in range(m)]
    
    max_gates = m * n
    gate_count_limit = 250
    
    while unsatisfied and len(gates) < max_gates and len(gates) < gate_count_limit:
        if time.time() > deadline:
            break
        
        W = n + 1 + len(gates)
        avail_set = set(vec_to_wires.keys())
        U = len(unsatisfied)
        
        if U == 0:
            break
        
        targets_U = np.array([targets[i] for i in unsatisfied], dtype=np.int64)
        W_mat = np.array(wire_vec, dtype=np.int64)
        
        # Compute distances from each target to each available wire
        xor_mat = (targets_U[:, None, :] + W_mat[None, :, :]) % 2
        dist_mat = xor_mat.sum(axis=2)  # U x W
        
        min_dist = dist_mat.min(axis=1)  # U
        min_idx = dist_mat.argmin(axis=1)  # U
        
        best_score = -1.0
        best_pair = None
        best_vec = None
        
        # Sample pairs if W is large
        if W <= 60:
            pairs = [(a, b) for a in range(W) for b in range(a + 1, W)]
        else:
            n_samples = min(300, W * (W - 1) // 2)
            pairs = []
            for _ in range(n_samples):
                a = random.randint(0, W - 1)
                b = random.randint(0, W - 1)
                if a != b:
                    pairs.append((min(a,b), max(a,b)))
            pairs = list(set(pairs))
        
        for a, b in pairs:
            v = (wire_vec[a] + wire_vec[b]) % 2
            vb = v.tobytes()
            if vb in avail_set:
                continue
            
            # Score: how many targets get closer?
            xor_v = (targets_U + v[None, :]) % 2
            hamming = xor_v.sum(axis=1)  # U
            
            # A target is "closer" if hamming < min_dist
            improvements = (hamming < min_dist).sum()
            # Bonus if it exactly matches
            exact_matches = (hamming == 0).sum()
            
            score = float(improvements) + 2.0 * float(exact_matches)
            
            if score > best_score:
                best_score = score
                best_pair = (a, b)
                best_vec = v
        
        if best_pair is None or best_score <= 0:
            break
        
        a, b = best_pair
        k = len(gates)
        new_wire = n + 1 + k
        gates.append([a, b])
        wire_vec.append(best_vec)
        vec_to_wires[best_vec.tobytes()].append(new_wire)
        
        vb = best_vec.tobytes()
        still_unsatisfied = []
        for i in unsatisfied:
            if vb == target_bytes[i]:
                outputs[i] = new_wire
                satisfied[i] = True
            else:
                still_unsatisfied.append(i)
        unsatisfied = still_unsatisfied
    
    # Fallback
    for i in unsatisfied:
        if time.time() > deadline:
            break
        target = targets[i]
        positions = np.where(target)[0].tolist()
        if len(positions) == 0:
            outputs[i] = n
        elif len(positions) == 1:
            outputs[i] = positions[0]
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                b = positions[j]
                new_vec = (wire_vec[current] + wire_vec[b]) % 2
                nb = new_vec.tobytes()
                if nb in vec_to_wires:
                    current = vec_to_wires[nb][0]
                else:
                    k = len(gates)
                    new_w = n + 1 + k
                    gates.append([current, b])
                    wire_vec.append(new_vec)
                    vec_to_wires[nb].append(new_w)
                    current = new_w
            outputs[i] = current
    
    if all(o is not None for o in outputs) and _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return None


def synthesize(matrix, depth_limit):
    """Entry point."""
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    start_time = time.time()
    
    # Baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_result = baseline
    best_count = len(baseline['gates'])
    
    if best_count == 0:
        return baseline
    
    # Strategy 1: MST-based synthesis
    try:
        mst = _mst_synthesis(matrix, depth_limit, time_budget=25.0)
        if mst is not None and len(mst['gates']) < best_count:
            best_result = mst
            best_count = len(mst['gates'])
    except Exception:
        pass
    
    # Strategy 2: Greedy vector synthesis
    elapsed = time.time() - start_time
    if elapsed < 45.0 and best_count > 0:
        try:
            gv = _greedy_vector_synthesis(matrix, depth_limit, time_budget=20.0)
            if gv is not None and len(gv['gates']) < best_count:
                best_result = gv
                best_count = len(gv['gates'])
        except Exception:
            pass
    
    # CSE Polish
    elapsed = time.time() - start_time
    if elapsed < 50.0 and best_count > 0:
        try:
            polished = _cse_polish(matrix, best_result['gates'], best_result['outputs'],
                                   depth_limit, time_budget=10.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass
    
    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result
    
    # Try MST again with more time if we have room
    remaining = 55.0 - elapsed
    if remaining > 15.0 and best_count > 0:
        try:
            mst2 = _mst_synthesis(matrix, depth_limit, time_budget=remaining - 5.0)
            if mst2 is not None and len(mst2['gates']) < best_count:
                best_result = mst2
                best_count = len(mst2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _cse_polish(matrix, best_result['gates'], best_result['outputs'],
                                         depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
