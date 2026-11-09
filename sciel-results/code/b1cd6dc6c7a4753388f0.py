"""
Exact XOR-circuit synthesis using CSE with basis expansion and local search.

Strategy:
1. Build initial circuit with common sub-expression elimination (CSE).
2. Expand the basis by adding useful XOR combinations of existing wires.
3. Re-solve the output assignment using the expanded basis.
4. Local search: try alternative row orderings and gate reassignments.
"""

import time
import random
import numpy as np

DEADLINE_SECONDS = 85  # safety margin before 90s kill


def _popcount(x):
    """Count set bits in an integer."""
    return bin(x).count('1')


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        wire_mask[wire] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        if wire_mask.get(out_wire, 0) != expected_mask:
            return False

    if depth_limit < 0:
        return True

    depth = {}
    for i in range(n + 1):
        depth[i] = 0
    max_depth = 0
    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        if d > max_depth:
            max_depth = d
    for out in circuit['outputs']:
        if depth.get(out, 0) > max_depth:
            max_depth = depth.get(out, 0)

    return max_depth <= depth_limit


def _build_circuit_cse(matrix, n, row_order=None):
    """
    Build a circuit using common sub-expression elimination (CSE).
    Each row is built as a balanced XOR tree, sharing sub-expressions
    via a global gate registry.
    """
    m = matrix.shape[0]
    if row_order is None:
        row_order = list(range(m))

    # Precompute row masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    def xor_tree(indices):
        if len(indices) == 1:
            return indices[0]
        mid = len(indices) // 2
        left = xor_tree(indices[:mid])
        right = xor_tree(indices[mid:])
        return get_xor_wire(left, right)

    outputs = [None] * m
    for idx in row_order:
        mask = row_masks[idx]
        ones = [j for j in range(n) if (mask >> j) & 1]
        if len(ones) == 0:
            outputs[idx] = n
        elif len(ones) == 1:
            outputs[idx] = ones[0]
        else:
            outputs[idx] = xor_tree(ones)

    return {'gates': gates, 'outputs': outputs}


def _mst_order(row_masks):
    """
    Order rows via minimum spanning tree on Hamming distances.
    This tends to group similar rows together, increasing sharing.
    """
    m = len(row_masks)
    if m <= 1:
        return list(range(m))

    in_mst = [False] * m
    in_mst[0] = True
    min_dist = [_popcount(row_masks[0] ^ row_masks[i]) for i in range(m)]
    min_dist[0] = 0
    ordered = [0]

    for _ in range(m - 1):
        best = -1
        best_d = float('inf')
        for i in range(m):
            if not in_mst[i] and min_dist[i] < best_d:
                best_d = min_dist[i]
                best = i
        if best == -1:
            break
        in_mst[best] = True
        ordered.append(best)
        for j in range(m):
            if not in_mst[j]:
                d = _popcount(row_masks[best] ^ row_masks[j])
                if d < min_dist[j]:
                    min_dist[j] = d

    return ordered


def _build_with_basis_expansion(matrix, n, depth_limit=16, time_budget=20.0):
    """
    Build circuit with basis expansion:
    1. Start with input wires as basis.
    2. Iteratively add the most useful XOR combination.
    3. Re-assign outputs to use the expanded basis.
    """
    start_time = time.time()
    m = matrix.shape[0]

    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        target_masks.append(mask)

    # Basis: wire_id -> mask
    basis = {}
    for i in range(n):
        basis[i] = (1 << i)
    basis[n] = 0  # zero wire

    # Gate registry for CSE
    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    # Track which targets are satisfied by the current basis
    def get_satisfied():
        satisfied = set()
        for i in range(m):
            for w, mask in basis.items():
                if mask == target_masks[i]:
                    satisfied.add(i)
                    break
        return satisfied

    # Iteratively add useful wires
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break

        remaining = [i for i in range(m) if i not in get_satisfied()]
        if not remaining:
            break

        # Score all candidate pairs
        wire_ids = list(basis.keys())
        num_wires = len(wire_ids)

        best_score = 0
        best_pair = None

        for ai in range(num_wires):
            a = wire_ids[ai]
            mask_a = basis[a]
            for bi in range(ai + 1, num_wires):
                b = wire_ids[bi]
                mask_b = basis[b]
                pair_mask = mask_a ^ mask_b
                if pair_mask == 0:
                    continue

                # Score: how many remaining targets can be expressed
                # using this new wire plus existing basis
                score = 0
                for t in remaining:
                    res = target_masks[t] ^ pair_mask
                    # Check if res can be expressed as XOR of at most 2 basis wires
                    # (this is a heuristic; we'll refine later)
                    if res in basis.values():
                        score += 2
                    else:
                        # Check if res can be expressed as XOR of two basis wires
                        for w1 in wire_ids:
                            for w2 in wire_ids:
                                if w1 != w2 and basis[w1] ^ basis[w2] == res:
                                    score += 1
                                    break
                            if score > 0 and score % 2 == 0:
                                break

                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

        if best_pair is None or best_score == 0:
            break

        a, b = best_pair
        new_wire = get_xor_wire(a, b)
        basis[new_wire] = basis[a] ^ basis[b]

    # Assign outputs using the final basis
    outputs = [None] * m
    for i in range(m):
        target = target_masks[i]
        # Try to find a wire in basis
        found = False
        for w, mask in basis.items():
            if mask == target:
                outputs[i] = w
                found = True
                break
        if not found:
            # Build from scratch
            ones = [j for j in range(n) if (target >> j) & 1]
            if not ones:
                outputs[i] = n
            elif len(ones) == 1:
                outputs[i] = ones[0]
            else:
                def xor_tree(indices):
                    if len(indices) == 1:
                        return indices[0]
                    mid = len(indices) // 2
                    return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))
                outputs[i] = xor_tree(ones)

    return {'gates': gates, 'outputs': outputs}


def _local_search(matrix, circuit, depth_limit, time_budget=10.0):
    """
    Local search: try alternative row orderings and gate reassignments.
    """
    start = time.time()
    best = circuit
    best_count = len(circuit['gates'])
    m, n = matrix.shape

    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    rng = random.Random(42)
    attempts = 0

    # Try MST order
    mst_ord = _mst_order(row_masks)
    c = _build_circuit_cse(matrix, n, mst_ord)
    if len(c['gates']) < best_count:
        best = c
        best_count = len(c['gates'])

    # Try random orderings
    while time.time() - start < time_budget and attempts < 30:
        attempts += 1
        row_order = list(range(m))
        rng.shuffle(row_order)
        c = _build_circuit_cse(matrix, n, row_order)
        if len(c['gates']) < best_count:
            best = c
            best_count = len(c['gates'])

    return best


def _fallback_chain(matrix, n):
    """
    Guaranteed-valid fallback: sequential XOR chain per row, no sharing.
    """
    m = matrix.shape[0]
    gates = []
    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            current = ones[0]
            for j in range(1, len(ones)):
                new_wire = n + 1 + len(gates)
                gates.append([current, ones[j]])
                current = new_wire
            outputs.append(current)

    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """
    Main entry point for XOR-circuit synthesis.

    Args:
        matrix: 2D array (m x n) of 0s and 1s
        depth_limit: maximum allowed depth

    Returns:
        {'gates': [[wire_a, wire_b], ...], 'outputs': [wire, ...]}
    """
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape

    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    candidates = []

    # Strategy 1: Direct CSE with natural order
    if time.time() < deadline - 10:
        c1 = _build_circuit_cse(matrix, n)
        candidates.append(c1)

    # Strategy 2: CSE with MST order
    if time.time() < deadline - 10:
        row_masks = []
        for i in range(m):
            mask = 0
            for j in range(n):
                if matrix[i, j]:
                    mask |= (1 << j)
            row_masks.append(mask)
        mst_ord = _mst_order(row_masks)
        c2 = _build_circuit_cse(matrix, n, mst_ord)
        candidates.append(c2)

    # Strategy 3: Basis expansion
    if time.time() < deadline - 15:
        c3 = _build_with_basis_expansion(matrix, n, depth_limit,
                                         time_budget=min(15.0, deadline - time.time() - 15))
        candidates.append(c3)

    # Pick the best verified candidate
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        if _verify_circuit(matrix, c, depth_limit):
            count = len(c['gates'])
            if count < best_count:
                best_count = count
                best_circuit = c

    # If no valid candidate, use fallback
    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)
        if not _verify_circuit(matrix, best_circuit, depth_limit):
            best_circuit = _build_circuit_cse(matrix, n)

    # Local search on the best
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 2.0:
            improved = _local_search(matrix, best_circuit, depth_limit,
                                     time_budget=min(remaining_time, 15.0))
            if _verify_circuit(matrix, improved, depth_limit):
                best_circuit = improved

    return best_circuit
