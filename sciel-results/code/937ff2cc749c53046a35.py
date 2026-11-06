import numpy as np
import time

def _popcount(x):
    return int(bin(x).count('1'))

def synthesize(matrix, depth_limit):
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    matrix = np.array(matrix, dtype=np.uint8) % 2
    m, n = matrix.shape

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to ints
    row_ints = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i, j] & 1:
                v |= (1 << j)
        row_ints.append(v)

    if all(v == 0 for v in row_ints):
        return {'gates': [], 'outputs': [n] * m}

    # ---------- 1. Greedy Minimum Weight Basis ----------
    active = [v for v in row_ints if v != 0]
    basis = []
    for col in range(n):
        bit = 1 << col
        min_idx = -1
        min_weight = float('inf')
        for i, v in enumerate(active):
            if v & bit:
                w = _popcount(v)
                if w < min_weight:
                    min_weight = w
                    min_idx = i
        if min_idx == -1:
            continue
        pivot_vec = active[min_idx]
        basis.append(pivot_vec)
        new_active = []
        for i, v in enumerate(active):
            if i == min_idx:
                continue
            if v & bit:
                v = v ^ pivot_vec
            if v != 0:
                new_active.append(v)
        active = new_active

    r = len(basis)
    if r == 0:
        return {'gates': [], 'outputs': [n] * m}

    # ---------- 2. Compute combos (coefficients of each row in basis) ----------
    def compute_combos(basis):
        combos = []
        for j in range(m):
            v = row_ints[j]
            c = 0
            for i in range(r - 1, -1, -1):
                pv = basis[i]
                pcol = (pv & -pv).bit_length() - 1
                if v & (1 << pcol):
                    v ^= pv
                    c |= (1 << i)
            combos.append(c)
        return combos

    def total_cost(basis):
        combos = compute_combos(basis)
        cost = 0
        for b in basis:
            cost += _popcount(b)
        for c in combos:
            cost += _popcount(c)
        return cost

    current_basis = basis[:]
    current_cost = total_cost(current_basis)

    # ---------- 3. Local Search to Improve Basis ----------
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(r):
            if time.time() > deadline - 15.0:
                break
            for j in range(r):
                if i == j:
                    continue
                new_bi = current_basis[i] ^ current_basis[j]
                if new_bi == 0:
                    continue
                old_bi = current_basis[i]
                current_basis[i] = new_bi
                new_cost = total_cost(current_basis)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    current_basis[i] = old_bi

    # ---------- 4. Build Circuit with Sub-expression Sharing ----------
    combos = compute_combos(current_basis)

    gates = []
    wire_count = n + 1

    def xor_gate(a, b):
        nonlocal wire_count
        # Avoid trivial gates
        if a == b:
            return n  # x ^ x = 0
        w = wire_count
        wire_count += 1
        gates.append([a, b])
        return w

    # Compute basis wires with sharing among basis vectors themselves
    # Each basis vector is a set of input indices.
    def compute_vector_with_sharing(bits):
        """Compute XOR of a set of input wires, sharing sub-expressions."""
        bits = list(bits)
        if not bits:
            return n
        if len(bits) == 1:
            return bits[0]
        # Sort by wire index for determinism
        bits.sort()
        w = bits[0]
        for b in bits[1:]:
            w = xor_gate(w, b)
        return w

    # Build basis wires
    basis_wires = []
    for i in range(r):
        bits = [j for j in range(n) if current_basis[i] & (1 << j)]
        basis_wires.append(compute_vector_with_sharing(bits))

    # Now outputs are XORs of subsets of basis_wires.
    # Use greedy sub-expression sharing across outputs.
    # Each output is represented as a frozenset of wire indices.
    exprs = []
    for j in range(m):
        c = combos[j]
        idxs = [i for i in range(r) if c & (1 << i)]
        exprs.append(frozenset(idxs))

    # Greedy sharing: repeatedly find the most common subset (of size >= 2)
    # across the current expressions, factor it out.
    # We track each expression as a set of wire indices (could be inputs, gate wires, or basis wires).
    # Start: each expr is a set of basis_wire indices.
    expr_sets = [set(e) for e in exprs]

    # To share, we look for common subsets among expr_sets.
    # A "subset" is a set of wires that appears in multiple expressions.
    # We factor out the most frequent large subset.

    def _get_subset_counts(expr_sets):
        """Count frequency of each subset (size 2..min_size) across expressions."""
        from collections import defaultdict
        counts = defaultdict(int)
        for es in expr_sets:
            if len(es) < 2:
                continue
            # Only consider subsets of size 2 up to some limit for efficiency
            # For larger sets, consider the full set and its 2-subsets
            elems = sorted(es)
            # Full set
            counts[frozenset(es)] += 1
            # 2-subsets
            for a in range(len(elems)):
                for b in range(a + 1, len(elems)):
                    counts[frozenset([elems[a], elems[b]])] += 1
        return counts

    max_share_iters = 200
    for _ in range(max_share_iters):
        if time.time() > deadline - 5.0:
            break
        counts = _get_subset_counts(expr_sets)
        # Find the subset with highest frequency * (size - 1) savings
        best_subset = None
        best_score = 0
        for subset, cnt in counts.items():
            if cnt >= 2:  # must appear in at least 2 expressions
                size = len(subset)
                if size < 2:
                    continue
                savings = (cnt - 1) * (size - 1)  # gates saved by factoring
                if savings > best_score:
                    best_score = savings
                    best_subset = subset

        if best_subset is None:
            break

        # Factor out best_subset
        subset_list = sorted(best_subset)
        # Compute the shared wire
        w = subset_list[0]
        for b in subset_list[1:]:
            w = xor_gate(w, b)

        # Update all expressions that contain this subset
        for es in expr_sets:
            if best_subset.issubset(es):
                es.difference_update(best_subset)
                es.add(w)

    # Build final outputs from expr_sets
    outputs = []
    for es in expr_sets:
        if not es:
            outputs.append(n)
        elif len(es) == 1:
            outputs.append(list(es)[0])
        else:
            elems = sorted(es)
            w = elems[0]
            for b in elems[1:]:
                w = xor_gate(w, b)
            outputs.append(w)

    # ---------- 5. Verification ----------
    wire_val = [0] * wire_count
    for j in range(n):
        wire_val[j] = 1 << j
    wire_val[n] = 0
    for k in range(len(gates)):
        a, b = gates[k]
        if a < wire_count and b < wire_count:
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        else:
            wire_val[n + 1 + k] = 0

    ok = True
    for j in range(m):
        if outputs[j] >= wire_count or wire_val[outputs[j]] != row_ints[j]:
            ok = False
            break

    if not ok:
        # Fallback: Direct synthesis (no sharing)
        gates = []
        wire_count = n + 1
        def xor_gate2(a, b):
            nonlocal wire_count
            w = wire_count
            wire_count += 1
            gates.append([a, b])
            return w

        outputs = []
        for j in range(m):
            bits = [k for k in range(n) if row_ints[j] & (1 << k)]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = xor_gate2(w, b)
                outputs.append(w)

    # Depth check
    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        wire_depth = [0] * (n + 1 + len(gates))
        for k, (a, b) in enumerate(gates):
            if a < len(wire_depth) and b < len(wire_depth):
                wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < len(wire_depth):
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    depth = _compute_depth(gates, outputs)
    if depth > depth_limit:
        # Fallback to direct synthesis (which has depth = max_row_weight - 1, usually small)
        gates = []
        wire_count = n + 1
        def xor_gate3(a, b):
            nonlocal wire_count
            w = wire_count
            wire_count += 1
            gates.append([a, b])
            return w

        outputs = []
        for j in range(m):
            bits = [k for k in range(n) if row_ints[j] & (1 << k)]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = xor_gate3(w, b)
                outputs.append(w)

    return {'gates': gates, 'outputs': outputs}
