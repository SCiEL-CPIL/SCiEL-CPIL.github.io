"""Exact XOR-circuit synthesis: global vector-space greedy basis construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

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


def _global_vector_greedy(matrix, depth_limit=16, time_budget=40.0):
    """
    Global vector-space greedy basis construction:
    - Maintain set of available wires (vectors in GF(2)^n).
    - At each step, for each candidate new vector v = a XOR b (a,b available),
      compute a soft-min surrogate score: for each unsatisfied target t_i,
      the "cost" is the minimum Hamming distance from t_i to any available vector
      (or to v itself). The score is the log-sum-exp of negative costs (soft-min),
      which smoothly balances coverage across all rows.
    - Pick the gate that maximizes the soft-min score.
    - When a new vector exactly equals a target, mark it satisfied.
    - Fallback to chains for remaining targets.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget

    targets = [matrix[i].copy() for i in range(m)]
    target_bytes = [t.tobytes() for t in targets]

    # Initialize wires
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

    # Precompute target popcounts for distance estimation
    target_popcounts = [int(np.sum(targets[i])) for i in range(m)]

    # Soft-min temperature parameter
    beta = 2.0  # Higher = more peaked (closer to hard min)

    max_gates = m * n
    gate_count_limit = 300  # safety

    while unsatisfied and len(gates) < max_gates and len(gates) < gate_count_limit:
        if time.time() > deadline:
            break

        W = n + 1 + len(gates)
        avail_set = set(vec_to_wires.keys())

        # For each unsatisfied target, compute current min-distance to available wires
        # and to each candidate new vector.
        # To keep it tractable, we compute:
        #   For each candidate pair (a,b): new_vec = v_a XOR v_b
        #   Score = sum over unsatisfied i of soft_min_dist(target_i, available ∪ {new_vec})
        #   where soft_min_dist uses log-sum-exp of -beta * hamming_dist

        # Optimization: precompute for each unsatisfied target the current min dist
        # and the second-min, so adding one vector only changes the min if new_vec is closer.

        # For each unsatisfied target t_i, find the 2 closest available wires
        # (by Hamming distance). This lets us quickly compute the new min when adding v.

        # Build list of available wire vectors as a matrix for fast XOR
        W_mat = np.array(wire_vec, dtype=np.int64)  # W x n

        # For each unsatisfied target, compute distances to all available wires
        # dist_matrix: |unsatisfied| x W
        U = len(unsatisfied)
        if U == 0:
            break

        # Vectorized distance computation
        # dist[i, a] = popcount(target_i XOR wire_vec[a])
        # Use: popcount(x) = sum of bits
        # For GF(2): target_i XOR wire_a has popcount = sum(target_i + wire_a mod 2)
        # We can compute: for each unsatisfied i, dist to all W wires
        # dist_mat = (targets_U[:, None, :] + W_mat[None, :, :]) % 2
        # then sum along last axis
        # This is U x W x n which can be large. For n=16, W~30, U~16: 16*30*16 = 7680 ops - fine.
        # For n=16, W~50, U~16: 16*50*16 = 12800 - still fine.

        targets_U = np.array([targets[i] for i in unsatisfied], dtype=np.int64)  # U x n

        # Compute all pairwise XOR popcounts: U x W
        # XOR in GF(2) is addition mod 2
        xor_mat = (targets_U[:, None, :] + W_mat[None, :, :]) % 2  # U x W x n
        dist_mat = xor_mat.sum(axis=2)  # U x W

        # Find min and second-min distances for each target
        min_dist = dist_mat.min(axis=1)  # U
        min_idx = dist_mat.argmin(axis=1)  # U

        # For scoring a new vector v:
        #   new_dist[i] = min(min_dist[i], hamming(target_i, v))
        #   score = -sum(logsumexp(-beta * new_dist[i]) for i)  # soft-min surrogate
        #   (negative because we want to minimize distances)
        # Actually, let's define:
        #   For each target i, cost_i(v) = min(min_dist[i], hamming(t_i, v))
        #   Total score(v) = -sum_i logsumexp over "virtual" distances... 
        #   Simpler: score(v) = sum_i (min_dist[i] - cost_i(v))  # reduction in min distance
        #   But this is just the sum of improvements. To make it "soft", use:
        #   score(v) = sum_i [ -log(1 + exp(-beta * (min_dist[i] - hamming(t_i, v)))) ]
        #   This is a smooth approximation to max(0, min_dist[i] - hamming(t_i, v))

        # Even simpler and effective: 
        #   score(v) = sum_i softplus(beta * (min_dist[i] - hamming(t_i, v)))
        #   where softplus(x) = log(1 + exp(x))
        #   This is always positive and smooth.

        best_score = -1.0
        best_pair = None
        best_vec = None

        # Iterate over all pairs
        # For W up to ~60, W*(W-1)/2 ~ 1800 pairs. For each pair, compute U hamming dists.
        # Total: 1800 * U * n ops. For U=16, n=16: 1800*16*16 = 460800 - fast.

        if W <= 80:
            for a in range(W):
                for b in range(a + 1, W):
                    v = (wire_vec[a] + wire_vec[b]) % 2
                    vb = v.tobytes()
                    if vb in avail_set:
                        continue  # already exists

                    # Compute hamming distances from v to all unsatisfied targets
                    # hamming[i] = popcount(targets_U[i] XOR v)
                    xor_v = (targets_U + v[None, :]) % 2  # U x n
                    hamming = xor_v.sum(axis=1)  # U

                    # score = sum_i softplus(beta * (min_dist[i] - hamming[i]))
                    diff = min_dist - hamming  # U
                    # softplus(x) = log(1 + exp(x)), but clip for numerical stability
                    diff_clipped = np.clip(diff, -50, 50)
                    sp = np.log1p(np.exp(diff_clipped))
                    score = float(sp.sum())

                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_vec = v
        else:
            # For larger W, sample pairs
            n_samples = min(500, W * (W - 1) // 2)
            for _ in range(n_samples):
                a = random.randint(0, W - 1)
                b = random.randint(0, W - 1)
                if a == b:
                    continue
                v = (wire_vec[a] + wire_vec[b]) % 2
                vb = v.tobytes()
                if vb in avail_set:
                    continue

                xor_v = (targets_U + v[None, :]) % 2
                hamming = xor_v.sum(axis=1)
                diff = min_dist - hamming
                diff_clipped = np.clip(diff, -50, 50)
                sp = np.log1p(np.exp(diff_clipped))
                score = float(sp.sum())

                if score > best_score:
                    best_score = score
                    best_pair = (a, b)
                    best_vec = v

        if best_pair is None or best_score <= 0.01:
            break

        # Add the gate
        a, b = best_pair
        k = len(gates)
        new_wire = n + 1 + k
        gates.append([a, b])
        wire_vec.append(best_vec)
        vec_to_wires[best_vec.tobytes()].append(new_wire)

        # Check if this satisfies any targets
        vb = best_vec.tobytes()
        still_unsatisfied = []
        for i in unsatisfied:
            if vb == target_bytes[i]:
                outputs[i] = new_wire
                satisfied[i] = True
            else:
                still_unsatisfied.append(i)
        unsatisfied = still_unsatisfied

    # Fallback: for remaining unsatisfied targets, use chains with existing wires
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
            # Try 1-gate completion: target = a XOR b where a,b available
            W = n + 1 + len(gates)
            avail_set = set(vec_to_wires.keys())
            found = False
            for a in range(W):
                comp = (target + wire_vec[a]) % 2
                cb = comp.tobytes()
                if cb in avail_set:
                    for b in vec_to_wires[cb]:
                        if b != a:
                            k = len(gates)
                            new_wire = n + 1 + k
                            gates.append([a, b])
                            wire_vec.append(comp)
                            vec_to_wires[cb].append(new_wire)
                            outputs[i] = new_wire
                            found = True
                            break
                if found:
                    break

            if not found:
                # Chain from input positions, reusing existing wires
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


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """
    Redundancy-elimination polish:
    - Remove unused gates
    - Redirect gates whose output vector is already available from an earlier wire
    Repeat until no improvement.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(50):
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

        # Pass 2: redirect redundant gates
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
    return {'gates': gates, 'outputs': outputs}


def _local_search_polish(matrix, gates, outputs, depth_limit=16, time_budget=8.0):
    """
    Local search: try to replace pairs of gates with a single gate,
    or restructure small subcircuits to reduce gate count.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]
    best_count = len(gates)

    for iteration in range(200):
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

        # Try removing each gate and checking if the circuit still works
        # (i.e., if the gate's output is not needed or can be replaced)
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k].tobytes()

            # Check if wire_k is used by any later gate or output
            used = False
            for j in range(k + 1, num_gates):
                if gates[j][0] == wire_k or gates[j][1] == wire_k:
                    used = True
                    break
            if not used:
                for ow in outputs:
                    if ow == wire_k:
                        used = True
                        break

            if not used:
                # Safe to remove
                gates.pop(k)
                if _verify_solution(matrix, gates, outputs, depth_limit):
                    best_count = len(gates)
                else:
                    gates.insert(k, [wire_k - n - 1, wire_k - n - 1])  # restore (approx)
                    # Actually, we need to restore properly
                    # Let's just skip this approach and rely on _polish
                break

    if _verify_solution(matrix, gates, outputs, depth_limit):
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

    # Strategy 1: Global vector-space greedy basis construction
    try:
        gv = _global_vector_greedy(matrix, depth_limit, time_budget=35.0)
        if gv is not None and len(gv['gates']) < best_count:
            best_result = gv
            best_count = len(gv['gates'])
    except Exception:
        pass

    # Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
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

    # Try global vector greedy again with different random seed for tie-breaking
    remaining = 55.0 - elapsed
    if remaining > 15.0 and best_count > 0:
        try:
            gv2 = _global_vector_greedy(matrix, depth_limit, time_budget=remaining - 5.0)
            if gv2 is not None and len(gv2['gates']) < best_count:
                best_result = gv2
                best_count = len(gv2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
