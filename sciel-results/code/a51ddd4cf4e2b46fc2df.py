"""Exact XOR-circuit synthesis: Global Minimum-Weight Matching (Hungarian) for subexpression sharing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

    # Depth check
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


def _hungarian(cost_matrix):
    """
    Hungarian algorithm for minimum weight perfect matching on a square cost matrix.
    cost_matrix: 2D numpy array, shape (N, N).
    Returns: list of column indices (one per row) giving the optimal assignment.
    """
    N = len(cost_matrix)
    if N == 0:
        return []
    # Use 1-indexed internally
    INF = float('inf')
    u = [0.0] * (N + 1)
    v = [0.0] * (N + 1)
    p = [0] * (N + 1)  # p[j] = row assigned to column j
    way = [0] * (N + 1)

    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (N + 1)
        used = [False] * (N + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, N + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augmenting path
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    # Build result: for each row i (1..N), find which column j has p[j] == i
    result = [0] * N
    for j in range(1, N + 1):
        if p[j] != 0:
            result[p[j] - 1] = j - 1
    return result


def _build_candidate_wires(matrix, n, max_candidates=60, time_budget=5.0):
    """
    Build a set of candidate intermediate wires (beyond inputs and zero).
    Strategy: include all pairwise XORs of input vectors, and iteratively add
    XORs of existing candidates that reduce the total Hamming weight of unsatisfied targets.
    """
    m = matrix.shape[0]
    deadline = time.time() + time_budget

    # Start with input vectors and zero
    candidates = []  # list of numpy vectors
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
    candidates.append(np.zeros(n, dtype=np.int64))  # zero wire

    vec_set = set()
    for v in candidates:
        vec_set.add(v.tobytes())

    # Add pairwise XORs of inputs
    for a in range(n):
        for b in range(a + 1, n):
            v = (candidates[a] + candidates[b]) % 2
            vb = v.tobytes()
            if vb not in vec_set:
                vec_set.add(vb)
                candidates.append(v)

    # Iteratively add useful XORs
    # Score a candidate v by how many targets it directly satisfies or helps
    for iteration in range(5):
        if len(candidates) >= max_candidates:
            break
        if time.time() > deadline:
            break

        # Find the best new candidate: XOR of two existing candidates
        best_score = -1
        best_a = -1
        best_b = -1
        best_v = None

        num_cands = len(candidates)
        # Limit pairs for efficiency
        max_pairs = min(num_cands * (num_cands - 1) // 2, 2000)
        if num_cands <= 40:
            pairs = [(a, b) for a in range(num_cands) for b in range(a + 1, num_cands)]
        else:
            indices = list(range(num_cands))
            random.shuffle(indices)
            pairs = []
            for a_idx in indices:
                for b_idx in indices:
                    if a_idx < b_idx:
                        pairs.append((a_idx, b_idx))
                    if len(pairs) >= max_pairs:
                        break
                if len(pairs) >= max_pairs:
                    break

        for a, b in pairs:
            v = (candidates[a] + candidates[b]) % 2
            vb = v.tobytes()
            if vb in vec_set:
                continue
            # Score: how many targets does this help?
            score = 0
            for i in range(m):
                tv = matrix[i]
                tb = tv.tobytes()
                if tb == vb:
                    score += 10
                else:
                    # Check if tv XOR v is in vec_set (1-step help)
                    cand = (tv + v) % 2
                    if cand.tobytes() in vec_set:
                        score += 1
            if score > best_score:
                best_score = score
                best_a = a
                best_b = b
                best_v = v

        if best_score <= 0 or best_a < 0:
            break

        vec_set.add(best_v.tobytes())
        candidates.append(best_v)

    return candidates


def _matching_synthesis(matrix, depth_limit=16, time_budget=30.0):
    """
    Global minimum-weight matching synthesis.
    
    1. Build candidate wires (inputs, zero, and useful intermediates).
    2. For each target row, compute the cost of satisfying it via each candidate wire.
    3. Use Hungarian algorithm to find the minimum-cost assignment.
    4. Construct the circuit from the assignment.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget

    # Build candidate wires
    candidates = _build_candidate_wires(matrix, n, max_candidates=50, time_budget=5.0)
    num_cands = len(candidates)

    # vec_to_idx: map vector bytes to candidate index
    vec_to_idx = {}
    for idx, v in enumerate(candidates):
        vec_to_idx[v.tobytes()] = idx

    # For each target row i, compute cost of using each candidate wire j.
    # Cost interpretation:
    #   If target[i] == candidates[j]: cost 0 (directly available)
    #   If target[i] XOR candidates[j] is in candidates: cost 1 (one more gate)
    #   Otherwise: cost = Hamming weight of target[i] (chain length estimate)
    # We want to minimize total cost, but also want to minimize the number of
    # NEW gates needed. The cost matrix guides which target uses which wire.

    # Actually, let's think differently. The matching assigns each target to a
    # "realization strategy". But the circuit is built from gates, not directly
    # from the matching. Let me reframe:
    #
    # The matching tells us which intermediate wire each target "wants" to use.
    # Then we build the circuit to realize all needed intermediate wires.
    #
    # Better approach: The matching is between targets and "partial realizations".
    # A partial realization for target t via wire w is: t = w XOR (something available).
    # The cost is the number of additional gates needed.
    #
    # Let's use a simpler but effective formulation:
    # For each target i and candidate wire j, define cost[i][j] as:
    #   0 if matrix[i] == candidates[j]
    #   1 if matrix[i] XOR candidates[j] is in vec_to_idx (i.e., one gate: new = j XOR k)
    #   2 if matrix[i] XOR candidates[j] XOR candidates[k] is in vec_to_idx for some k (two gates)
    #   else: Hamming weight of matrix[i] (fallback chain)
    #
    # Then the matching minimizes sum of costs. After matching, we build the circuit.

    # Compute cost matrix
    # For efficiency, precompute for each candidate wire j, the set of vectors at distance 1
    dist1 = [set() for _ in range(num_cands)]
    for j in range(num_cands):
        for k in range(num_cands):
            if j == k:
                continue
            v = (candidates[j] + candidates[k]) % 2
            dist1[j].add(v.tobytes())

    # Cost matrix: m x num_cands
    # We need a square matrix for Hungarian. Pad with dummy columns/rows.
    # If m > num_cands, we can't do perfect matching. Use rectangular approach or pad.
    # Let's handle the case where m <= num_cands (typical for our problem sizes).

    if m > num_cands:
        # Pad candidates with more vectors
        for a in range(n):
            for b in range(a + 1, n):
                v = (candidates[a] + candidates[b]) % 2
                vb = v.tobytes()
                if vb not in vec_to_idx:
                    vec_to_idx[vb] = num_cands
                    candidates.append(v)
                    num_cands += 1
                    dist1.append(set())
                    for k in range(num_cands):
                        v2 = (candidates[num_cands - 1] + candidates[k]) % 2
                        dist1[num_cands - 1].add(v2.tobytes())
        # Recompute dist1 for new candidates
        for j in range(num_cands - len(dist1) + 1, num_cands):
            dist1[j] = set()
            for k in range(num_cands):
                if j == k:
                    continue
                v = (candidates[j] + candidates[k]) % 2
                dist1[j].add(v.tobytes())

    if m > num_cands:
        # Still too many; fall back to greedy
        return _greedy_fallback(matrix, depth_limit, time_budget)

    # Cost matrix (m x num_cands)
    cost = np.zeros((m, num_cands), dtype=np.float64)
    for i in range(m):
        tv = matrix[i]
        tb = tv.tobytes()
        hw = np.count_nonzero(tv)
        for j in range(num_cands):
            if tb == candidates[j].tobytes():
                cost[i][j] = 0.0
            elif tb in dist1[j]:
                # target = candidates[j] XOR candidates[k] for some k
                cost[i][j] = 1.0
            else:
                # Check distance 2
                found_d2 = False
                for k in range(num_cands):
                    v = (tv + candidates[k]) % 2
                    if v.tobytes() in dist1[j]:
                        found_d2 = True
                        break
                if found_d2:
                    cost[i][j] = 2.0
                else:
                    cost[i][j] = float(hw)

    # Pad to square: N = max(m, num_cands)
    N = max(m, num_cands)
    cost_sq = np.zeros((N, N), dtype=np.float64)
    cost_sq[:m, :num_cands] = cost
    # Dummy entries: high cost
    cost_sq[m:, :] = 1000.0
    cost_sq[:, num_cands:] = 1000.0

    # Run Hungarian
    assignment = _hungarian(cost_sq)

    # Build the circuit from the assignment
    # For each target i, it's assigned to candidate wire j = assignment[i].
    # We need to realize candidate wire j (if not already an input/zero) and then
    # possibly one more gate to get the target.

    gates = []
    outputs = []
    wire_vec = candidates[:]  # wires 0..num_cands-1 are candidates
    # But wait: candidates[0..n-1] are inputs, candidates[n] is zero.
    # New gates start at wire num_cands.
    # Actually, let's re-index: wires 0..n-1 are inputs, wire n is zero.
    # Candidates beyond n are intermediate wires that need gates.

    # Let's rebuild: we have a list of vectors. Wires 0..n-1 are inputs, wire n is zero.
    # For each candidate index j >= n+1, we need to create a gate.
    # But the order matters for depth. Let's create gates in order of candidate index.

    # Actually, the candidate list already has inputs (0..n-1) and zero (n) at the front.
    # So candidates[0..n] correspond to wires 0..n.
    # candidates[n+1..] are new wires that need gates.

    # Create gates for all candidate wires that are not inputs or zero
    cand_wire_map = {}  # candidate index -> wire number
    for j in range(n + 1):
        cand_wire_map[j] = j

    next_wire = n + 1
    for j in range(n + 1, num_cands):
        # Need to create candidates[j] as a gate
        # Find a, b such that candidates[a] XOR candidates[b] == candidates[j]
        found = False
        for a in range(j):
            v = (candidates[j] + candidates[a]) % 2
            vb = v.tobytes()
            if vb in vec_to_idx:
                b = vec_to_idx[vb]
                if b < j:  # Ensure we use already-created wires
                    gates.append([cand_wire_map[a], cand_wire_map[b]])
                    cand_wire_map[j] = next_wire
                    wire_vec.append(candidates[j])
                    next_wire += 1
                    found = True
                    break
        if not found:
            # Fallback: chain from inputs
            positions = np.where(candidates[j])[0].tolist()
            if len(positions) == 0:
                cand_wire_map[j] = n
            elif len(positions) == 1:
                cand_wire_map[j] = positions[0]
            else:
                current = positions[0]
                for pos in positions[1:]:
                    gates.append([current, pos])
                    current = next_wire
                    wire_vec.append(np.zeros(n, dtype=np.int64))  # placeholder
                    next_wire += 1
                cand_wire_map[j] = current

    # Now assign outputs
    for i in range(m):
        j = assignment[i]
        tv = matrix[i]
        tb = tv.tobytes()
        if tb == candidates[j].tobytes():
            outputs.append(cand_wire_map[j])
        else:
            # Need one more gate: target = candidates[j] XOR candidates[k]
            # Find k such that candidates[j] XOR candidates[k] == tv
            found = False
            for k in range(num_cands):
                v = (candidates[j] + candidates[k]) % 2
                if v.tobytes() == tb:
                    # Create gate: wire = cand_wire_map[j] XOR cand_wire_map[k]
                    a_wire = cand_wire_map[j]
                    b_wire = cand_wire_map[k]
                    gates.append([a_wire, b_wire])
                    outputs.append(next_wire)
                    wire_vec.append(tv)
                    vec_to_idx[tb] = num_cands  # mark as available
                    next_wire += 1
                    found = True
                    break
            if not found:
                # Fallback: chain
                positions = np.where(tv)[0].tolist()
                if len(positions) == 0:
                    outputs.append(n)
                elif len(positions) == 1:
                    outputs.append(positions[0])
                else:
                    current = positions[0]
                    for pos in positions[1:]:
                        gates.append([current, pos])
                        current = next_wire
                        wire_vec.append(np.zeros(n, dtype=np.int64))
                        next_wire += 1
                    outputs.append(current)

    return {'gates': gates, 'outputs': outputs}


def _greedy_fallback(matrix, depth_limit=16, time_budget=20.0):
    """Simple greedy fallback if matching fails."""
    m, n = matrix.shape
    deadline = time.time() + time_budget
    gates = []
    outputs = []
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))
    vec_to_wire = {}
    for w in range(n + 1):
        vec_to_wire[wire_vec[w].tobytes()] = w

    for i in range(m):
        if time.time() > deadline:
            break
        tv = matrix[i]
        tb = tv.tobytes()
        if tb in vec_to_wire:
            outputs.append(vec_to_wire[tb])
            continue
        # Try 1-gate
        found = False
        W = len(wire_vec)
        for a in range(W):
            cand = (tv + wire_vec[a]) % 2
            cb = cand.tobytes()
            if cb in vec_to_wire:
                b = vec_to_wire[cb]
                if b != a:
                    k = len(gates)
                    new_wire = n + 1 + k
                    gates.append([a, b])
                    wire_vec.append(cand)
                    vec_to_wire[cb] = new_wire
                    outputs.append(new_wire)
                    found = True
                    break
        if found:
            continue
        # Fallback: chain
        positions = np.where(tv)[0].tolist()
        if len(positions) == 0:
            outputs.append(n)
        elif len(positions) == 1:
            outputs.append(positions[0])
        else:
            current = positions[0]
            for pos in positions[1:]:
                new_vec = (wire_vec[current] + wire_vec[pos]) % 2
                nb = new_vec.tobytes()
                if nb in vec_to_wire:
                    current = vec_to_wire[nb]
                else:
                    k = len(gates)
                    new_wire = n + 1 + k
                    gates.append([current, pos])
                    wire_vec.append(new_vec)
                    vec_to_wire[nb] = new_wire
                    current = new_wire
            outputs.append(current)

    if len(outputs) < m:
        # Fill remaining with baseline
        base = _baseline_synthesis(matrix, depth_limit)
        for i in range(len(outputs), m):
            outputs.append(base['outputs'][i])
        gates.extend(base['gates'][len(gates):])

    return {'gates': gates, 'outputs': outputs}


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=8.0):
    """Redundancy-elimination polish."""
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(30):
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

        # Pass 1: remove unused gates
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

        # Pass 2: redirect to earlier wire
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

    return {'gates': gates, 'outputs': outputs}


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

    # Matching synthesis (main workhorse)
    try:
        matched = _matching_synthesis(matrix, depth_limit, time_budget=30.0)
        if _verify_solution(matrix, matched['gates'], matched['outputs'], depth_limit):
            if len(matched['gates']) < best_count:
                best_result = matched
                best_count = len(matched['gates'])
        else:
            # Matching produced invalid circuit; use greedy fallback
            greedy = _greedy_fallback(matrix, depth_limit, time_budget=15.0)
            if _verify_solution(matrix, greedy['gates'], greedy['outputs'], depth_limit):
                if len(greedy['gates']) < best_count:
                    best_result = greedy
                    best_count = len(greedy['gates'])
    except Exception:
        # Fallback to greedy
        try:
            greedy = _greedy_fallback(matrix, depth_limit, time_budget=20.0)
            if _verify_solution(matrix, greedy['gates'], greedy['outputs'], depth_limit):
                if len(greedy['gates']) < best_count:
                    best_result = greedy
                    best_count = len(greedy['gates'])
        except Exception:
            pass

    # Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
                               depth_limit, time_budget=8.0)
            if _verify_solution(matrix, polished['gates'], polished['outputs'], depth_limit):
                if len(polished['gates']) < best_count:
                    best_result = polished
                    best_count = len(polished['gates'])
        except Exception:
            pass

    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result

    # Additional round with remaining time
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            greedy2 = _greedy_fallback(matrix, depth_limit, time_budget=remaining - 3.0)
            if _verify_solution(matrix, greedy2['gates'], greedy2['outputs'], depth_limit):
                if len(greedy2['gates']) < best_count:
                    best_result = greedy2
                    best_count = len(greedy2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if _verify_solution(matrix, polished2['gates'], polished2['outputs'], depth_limit):
                    if len(polished2['gates']) < best_count:
                        best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
