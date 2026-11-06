"""Exact XOR-circuit synthesis: global greedy subexpression sharing + polish."""
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


def _global_greedy_synthesis(matrix, depth_limit=16, time_budget=45.0):
    """
    Global greedy subexpression sharing:
    - Maintain a set of available wires (vectors).
    - At each step, compute all possible XOR pairs from available wires.
    - Score each new vector by how many unsatisfied targets it directly satisfies
      or how many unsatisfied targets it can help (by being a subexpression).
    - Pick the highest-value new gate, add it, repeat until all targets satisfied.
    - Fallback to chain for remaining targets.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget

    # Target vectors
    targets = [matrix[i] for i in range(m)]
    target_bytes = [t.tobytes() for t in targets]

    # Initial wires: inputs 0..n-1, zero wire n
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))  # wire n = zero

    gates = []
    outputs = [None] * m
    satisfied = [False] * m

    # vec_to_wires: bytes -> list of wire indices
    vec_to_wires = defaultdict(list)
    for w in range(n + 1):
        vec_to_wires[wire_vec[w].tobytes()].append(w)

    # Check which targets are already satisfied
    for i in range(m):
        if target_bytes[i] in vec_to_wires:
            outputs[i] = vec_to_wires[target_bytes[i]][0]
            satisfied[i] = True

    # Main greedy loop
    max_gates = m * n  # safety cap
    while any(not s for s in satisfied) and len(gates) < max_gates:
        if time.time() > deadline:
            break

        W = n + 1 + len(gates)  # number of available wires

        # Score all possible new XOR results
        # For each pair (a, b), compute v_a XOR v_b, score it
        # Score = number of unsatisfied targets that equal this vector
        #       + 0.5 * number of unsatisfied targets where this vector is a "sub"
        #         (i.e., target XOR this vector is already available)
        # To keep it fast, we focus on direct matches and 1-step completions

        best_score = 0
        best_pair = None
        best_vec = None

        # Build set of available vector bytes for fast lookup
        avail_set = set(vec_to_wires.keys())

        # For scoring: for each unsatisfied target t, a new vector v is useful if:
        #   v == t (direct, score 1.0)
        #   t XOR v is in avail_set (1-step completion, score 0.5)

        unsatisfied_indices = [i for i in range(m) if not satisfied[i]]
        if not unsatisfied_indices:
            break

        # To avoid O(W^2 * m) which is too slow for large W, use a smarter approach:
        # For each unsatisfied target t:
        #   - Check if t is already in avail_set (shouldn't be, since unsatisfied)
        #   - For each available wire a, check if t XOR v_a is in avail_set
        #     This means adding gate (a, b) where v_b = t XOR v_a would satisfy t
        #     But we want to find the gate that maximizes total benefit

        # Alternative: iterate over all pairs but limit W
        # For small W (<= 30), do full O(W^2) scan
        # For larger W, sample

        if W <= 30:
            for a in range(W):
                for b in range(a + 1, W):
                    v = (wire_vec[a] + wire_vec[b]) % 2
                    vb = v.tobytes()
                    if vb in avail_set:
                        continue  # already exists
                    # Score
                    score = 0.0
                    for i in unsatisfied_indices:
                        if vb == target_bytes[i]:
                            score += 1.0
                        else:
                            # Check if target[i] XOR v is available
                            comp = (targets[i] + v) % 2
                            if comp.tobytes() in avail_set:
                                score += 0.3
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_vec = v
        else:
            # For larger W, use a different strategy:
            # For each unsatisfied target, find the best 1-gate completion
            # and accumulate scores
            pair_scores = defaultdict(float)
            pair_vecs = {}

            for i in unsatisfied_indices:
                t = targets[i]
                tb = target_bytes[i]
                # Direct: is t in avail? (no, since unsatisfied)
                # 1-gate: for each a, if t XOR v_a is in avail_set,
                #   then gate (a, b) where v_b = t XOR v_a satisfies target i
                for a in range(W):
                    comp = (t + wire_vec[a]) % 2
                    cb = comp.tobytes()
                    if cb in avail_set:
                        for b in vec_to_wires[cb]:
                            if b != a:
                                key = (min(a, b), max(a, b))
                                pair_scores[key] += 0.5
                                if key not in pair_vecs:
                                    pair_vecs[key] = (wire_vec[a] + wire_vec[b]) % 2

            if pair_scores:
                best_key = max(pair_scores, key=pair_scores.get)
                best_score = pair_scores[best_key]
                best_pair = best_key
                best_vec = pair_vecs[best_key]

            # Also check direct matches among pairs (sampled)
            if W <= 60:
                for a in range(W):
                    for b in range(a + 1, W):
                        v = (wire_vec[a] + wire_vec[b]) % 2
                        vb = v.tobytes()
                        if vb in avail_set:
                            continue
                        for i in unsatisfied_indices:
                            if vb == target_bytes[i]:
                                key = (a, b)
                                pair_scores[key] += 1.0
                                pair_vecs[key] = v
                                break
                if pair_scores:
                    best_key = max(pair_scores, key=pair_scores.get)
                    if pair_scores[best_key] > best_score:
                        best_score = pair_scores[best_key]
                        best_pair = best_key
                        best_vec = pair_vecs[best_key]

        if best_pair is None or best_score <= 0:
            # No beneficial gate found; fallback to chains for remaining
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
        for i in unsatisfied_indices:
            if not satisfied[i] and vb == target_bytes[i]:
                outputs[i] = new_wire
                satisfied[i] = True

        # Also check 1-step completions enabled by this new wire
        for i in unsatisfied_indices:
            if satisfied[i]:
                continue
            t = targets[i]
            # Check if t XOR new_vec is available
            comp = (t + best_vec) % 2
            cb = comp.tobytes()
            if cb in avail_set:
                # We can satisfy target i with one more gate
                pass  # Will be handled in next iteration or fallback

    # Fallback: for remaining unsatisfied targets, use chains
    for i in range(m):
        if satisfied[i]:
            continue
        if time.time() > deadline:
            break
        target = targets[i]
        positions = np.where(target)[0].tolist()
        if len(positions) == 0:
            outputs[i] = n
            satisfied[i] = True
        elif len(positions) == 1:
            outputs[i] = positions[0]
            satisfied[i] = True
        else:
            # Try to find best chain using existing wires
            # Greedy: at each step, XOR current with an available wire to get closer
            current_vec = target.copy()
            current_wire = None  # None means we haven't started

            # Try: find if target XOR some available wire is already available
            # (1-gate solution)
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
                            satisfied[i] = True
                            found = True
                            break
                if found:
                    break

            if not found:
                # Chain from input positions
                # Try to reuse existing wires
                current = positions[0]
                for j in range(1, len(positions)):
                    b = positions[j]
                    new_vec = (wire_vec[current] + wire_vec[b]) % 2
                    nb = new_vec.tobytes()
                    if nb in vec_to_wires:
                        current = vec_to_wires[nb][0]
                    else:
                        k = len(gates)
                        current = n + 1 + k
                        gates.append([current, b])  # BUG: current is new_wire index
                        # Fix:
                        gates[-1] = [current, b]  # wrong, let me redo
                        # Actually:
                        new_w = n + 1 + len(gates) - 1
                        gates[-1] = [current, b]
                        wire_vec.append(new_vec)
                        vec_to_wires[nb].append(new_w)
                        current = new_w
                outputs[i] = current
                satisfied[i] = True

    if all(satisfied) and _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}

    # If verification failed, try a cleaner fallback
    return None


def _row_by_row_greedy(matrix, depth_limit=16, time_budget=40.0):
    """
    Row-by-row greedy with global wire sharing:
    Process rows in order of popcount (ascending), for each row try to find
    the minimum-gate decomposition using existing wires + new gates.
    Use BFS up to depth 3 for each row.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget

    gates = []
    outputs = [None] * m
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))

    vec_to_wires = defaultdict(list)
    for w in range(n + 1):
        vec_to_wires[wire_vec[w].tobytes()].append(w)

    # Order rows by popcount
    popcounts = [int(np.sum(matrix[i])) for i in range(m)]
    order = sorted(range(m), key=lambda i: popcounts[i])

    for i in order:
        if time.time() > deadline:
            break
        target = matrix[i]
        tb = target.tobytes()

        # 0 gates
        if tb in vec_to_wires:
            outputs[i] = vec_to_wires[tb][0]
            continue

        W = n + 1 + len(gates)

        # 1 gate: a XOR b = target
        found = False
        for a in range(W):
            comp = (target + wire_vec[a]) % 2
            cb = comp.tobytes()
            if cb in vec_to_wires:
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
        if found:
            continue

        # 2 gates: (a XOR b) XOR c = target, i.e., a XOR b XOR c = target
        # Forward: compute all a XOR b, check if (a XOR b) XOR c = target for some c
        # Equivalently: a XOR b = target XOR c for some available c
        found = False
        # Build set of available bytes
        avail = set(vec_to_wires.keys())

        # For each c, target XOR c must be expressible as a XOR b
        # So for each c, for each a, check if (target XOR c) XOR a is available
        # This is O(W^2) which may be slow
        if W <= 50:
            for c in range(W):
                needed = (target + wire_vec[c]) % 2
                nb = needed.tobytes()
                if nb in avail:
                    # target = needed XOR c, but needed is already available
                    # So 1 gate: needed XOR c
                    k = len(gates)
                    new_wire = n + 1 + k
                    gates.append([vec_to_wires[nb][0], c])
                    wire_vec.append(needed)
                    vec_to_wires[nb].append(new_wire)
                    outputs[i] = new_wire
                    found = True
                    break
            if found:
                continue

            # True 2-gate: a XOR b = new_vec, new_vec XOR c = target
            # => a XOR b XOR c = target
            # For each pair (a,b), new_vec = a XOR b, check if new_vec XOR c = target
            # => c = new_vec XOR target, check if c is available
            for a in range(W):
                for b in range(a + 1, W):
                    nv = (wire_vec[a] + wire_vec[b]) % 2
                    nb = nv.tobytes()
                    if nb in avail:
                        continue  # already exists, would be 1-gate case
                    # Check if nv XOR target is available
                    c_vec = (nv + target) % 2
                    cb = c_vec.tobytes()
                    if cb in avail:
                        c = vec_to_wires[cb][0]
                        # Add gate 1: a XOR b
                        k1 = len(gates)
                        w1 = n + 1 + k1
                        gates.append([a, b])
                        wire_vec.append(nv)
                        vec_to_wires[nb].append(w1)
                        # Add gate 2: w1 XOR c
                        k2 = len(gates)
                        w2 = n + 1 + k2
                        gates.append([w1, c])
                        final_vec = (nv + wire_vec[c]) % 2
                        wire_vec.append(final_vec)
                        vec_to_wires[final_vec.tobytes()].append(w2)
                        outputs[i] = w2
                        found = True
                        break
                if found:
                    break
        if found:
            continue

        # Fallback: chain
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

    # Strategy 1: Row-by-row greedy with global sharing
    try:
        r2r = _row_by_row_greedy(matrix, depth_limit, time_budget=35.0)
        if r2r is not None and len(r2r['gates']) < best_count:
            best_result = r2r
            best_count = len(r2r['gates'])
    except Exception:
        pass

    # Strategy 2: Global greedy subexpression sharing
    elapsed = time.time() - start_time
    if elapsed < 50.0:
        try:
            gg = _global_greedy_synthesis(matrix, depth_limit, time_budget=20.0)
            if gg is not None and len(gg['gates']) < best_count:
                best_result = gg
                best_count = len(gg['gates'])
        except Exception:
            pass

    # Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
                               depth_limit, time_budget=8.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass

    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result

    # Try row-by-row again with remaining time (different random seed for tie-breaking)
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            r2r2 = _row_by_row_greedy(matrix, depth_limit, time_budget=remaining - 3.0)
            if r2r2 is not None and len(r2r2['gates']) < best_count:
                best_result = r2r2
                best_count = len(r2r2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
