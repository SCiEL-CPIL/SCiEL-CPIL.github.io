"""
Exact XOR-circuit synthesis with structure-aware decomposition, CSE,
and local-search refinement.

Key ideas:
1. Convert each row to a bitmask for efficient manipulation.
2. Use a global gate registry (CSE) so each unique XOR pair is computed once.
3. Multiple construction strategies:
   a. Direct balanced XOR trees with CSE.
   b. Incremental construction with row reordering.
   c. Common-factor extraction.
   d. Pattern grouping.
   e. Basis reduction via Gaussian elimination.
   f. Local-search refinement: try swapping/merging gates.
4. Pick the candidate with the fewest gates.
5. Verify exact correctness over GF(2) and depth constraint.
"""

import time
import numpy as np

DEADLINE_SECONDS = 85


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # constant zero

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        mask_a = wire_mask.get(a, 0)
        mask_b = wire_mask.get(b, 0)
        wire_mask[wire] = mask_a ^ mask_b

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        actual_mask = wire_mask.get(out_wire, 0)
        if actual_mask != expected_mask:
            return False

    if depth_limit < 0:
        return True

    max_depth = 0
    depth = {}
    for i in range(n + 1):
        depth[i] = 0

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        max_depth = max(max_depth, d)

    for out in circuit['outputs']:
        max_depth = max(max_depth, depth.get(out, 0))

    if max_depth > depth_limit:
        return False

    return True


def _make_gate_factory(n):
    """Create a gate factory with CSE registry."""
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_xor_wire(a, b):
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    return get_xor_wire, gates, gate_registry


def _xor_tree(indices, get_xor_wire):
    """Build a balanced XOR tree for a list of input indices."""
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    get_xor_wire, gates, _ = _make_gate_factory(n)
    outputs = []
    for i in range(matrix.shape[0]):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            outputs.append(_xor_tree(ones, get_xor_wire))
    return {'gates': gates, 'outputs': outputs}


def _build_incremental(matrix, n):
    """
    Incremental construction: order rows by Hamming distance,
    build each row from the previous using minimal XOR updates.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # Greedy nearest-neighbor ordering starting from lowest-weight row
    remaining = list(range(m))
    ordered = []
    start = min(remaining, key=lambda i: bin(row_masks[i]).count('1'))
    ordered.append(start)
    remaining.remove(start)

    while remaining:
        last = ordered[-1]
        best = min(remaining, key=lambda i: bin(row_masks[last] ^ row_masks[i]).count('1'))
        ordered.append(best)
        remaining.remove(best)

    get_xor_wire, gates, _ = _make_gate_factory(n)
    outputs = [None] * m
    current_wire = None
    current_set = set()

    for idx in ordered:
        row_mask = row_masks[idx]
        row_set = set(j for j in range(n) if (row_mask >> j) & 1)

        if current_wire is None:
            if len(row_set) == 0:
                current_wire = n
            elif len(row_set) == 1:
                current_wire = list(row_set)[0]
            else:
                current_wire = _xor_tree(list(row_set), get_xor_wire)
            current_set = row_set
        else:
            to_remove = current_set - row_set
            to_add = row_set - current_set

            if not to_remove and not to_add:
                pass
            else:
                change_count = len(to_remove) + len(to_add)

                if change_count <= 4:
                    new_wire = current_wire
                    for bit in to_remove:
                        new_wire = get_xor_wire(new_wire, bit)
                    for bit in to_add:
                        new_wire = get_xor_wire(new_wire, bit)
                    current_wire = new_wire
                else:
                    if len(row_set) == 0:
                        current_wire = n
                    elif len(row_set) == 1:
                        current_wire = list(row_set)[0]
                    else:
                        current_wire = _xor_tree(list(row_set), get_xor_wire)
                current_set = row_set

        outputs[idx] = current_wire

    return {'gates': gates, 'outputs': outputs}


def _build_common_factors(matrix, n):
    """
    Pre-compute frequent column-pair XORs, then build each row
    using those pre-computed wires where possible.
    """
    m = matrix.shape[0]

    pair_counts = {}
    row_ones = []
    for i in range(m):
        ones = np.nonzero(matrix[i])[0].tolist()
        row_ones.append(ones)
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                key = (min(ones[a], ones[b]), max(ones[a], ones[b]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    get_xor_wire, gates, gate_registry = _make_gate_factory(n)

    frequent = sorted(pair_counts.items(), key=lambda x: -x[1])
    for pair, count in frequent:
        if count >= 3:
            get_xor_wire(pair[0], pair[1])

    outputs = []
    for i in range(m):
        ones = row_ones[i]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            remaining = list(ones)
            wire = None
            changed = True
            while remaining and changed:
                changed = False
                for a_idx in range(len(remaining)):
                    for b_idx in range(a_idx + 1, len(remaining)):
                        a, b = remaining[a_idx], remaining[b_idx]
                        key = (min(a, b), max(a, b))
                        if key in gate_registry:
                            pw = gate_registry[key]
                            if wire is None:
                                wire = pw
                            else:
                                wire = get_xor_wire(wire, pw)
                            remaining.pop(b_idx)
                            remaining.pop(a_idx)
                            changed = True
                            break
                    if changed:
                        break
            for elem in remaining:
                if wire is None:
                    wire = elem
                else:
                    wire = get_xor_wire(wire, elem)
            if wire is None:
                wire = n
            outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}


def _build_pattern_grouped(matrix, n):
    """Group identical rows and compute each unique pattern once."""
    m = matrix.shape[0]
    pattern_map = {}
    for i in range(m):
        key = tuple(np.nonzero(matrix[i])[0].tolist())
        if key not in pattern_map:
            pattern_map[key] = i

    get_xor_wire, gates, _ = _make_gate_factory(n)
    pattern_wire = {}
    for pattern in pattern_map:
        if len(pattern) == 0:
            pattern_wire[pattern] = n
        elif len(pattern) == 1:
            pattern_wire[pattern] = pattern[0]
        else:
            pattern_wire[pattern] = _xor_tree(list(pattern), get_xor_wire)

    outputs = []
    for i in range(m):
        key = tuple(np.nonzero(matrix[i])[0].tolist())
        outputs.append(pattern_wire[key])

    return {'gates': gates, 'outputs': outputs}


def _build_basis(matrix, n):
    """
    Gaussian-elimination-based construction: compute a basis of pivot rows,
    then express all rows as combinations of basis wires.
    """
    m = matrix.shape[0]
    mat = matrix.copy().astype(int)

    # Perform Gaussian elimination to find pivot rows
    rank = 0
    pivot_rows = []
    pivot_cols = []

    for col in range(n):
        # Find a row with 1 in this column
        pivot_row = None
        for row in range(rank, m):
            if mat[row, col]:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        # Swap rows
        mat[rank], mat[pivot_row] = mat[pivot_row].copy(), mat[rank].copy()

        # Eliminate this column from other rows
        for row in range(m):
            if row != rank and mat[row, col]:
                mat[row] = (mat[row] + mat[rank]) % 2

        pivot_rows.append(rank)
        pivot_cols.append(col)
        rank += 1

    if rank == 0:
        return _build_direct(matrix, n)

    # Build circuit: first compute pivot row outputs, then combine for non-pivot rows
    get_xor_wire, gates, _ = _make_gate_factory(n)
    outputs = [None] * m

    # Compute each pivot row as a direct XOR tree
    pivot_wires = {}
    for pr in pivot_rows:
        ones = np.nonzero(matrix[pr])[0].tolist()
        if len(ones) == 0:
            w = n
        elif len(ones) == 1:
            w = ones[0]
        else:
            w = _xor_tree(ones, get_xor_wire)
        pivot_wires[pr] = w
        outputs[pr] = w

    # For non-pivot rows, express as combination of pivot rows
    # We need to track the linear combination
    # Re-do elimination tracking coefficients
    mat2 = matrix.copy().astype(int)
    coeffs = np.eye(m, dtype=int)  # coeffs[i] tells how row i is expressed in original rows

    rank2 = 0
    for col in range(n):
        pivot_row = None
        for row in range(rank2, m):
            if mat2[row, col]:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        mat2[rank2], mat2[pivot_row] = mat2[pivot_row].copy(), mat2[rank2].copy()
        coeffs[rank2], coeffs[pivot_row] = coeffs[pivot_row].copy(), coeffs[rank2].copy()

        for row in range(m):
            if row != rank2 and mat2[row, col]:
                mat2[row] = (mat2[row] + mat2[rank2]) % 2
                coeffs[row] = (coeffs[row] + coeffs[rank2]) % 2

        rank2 += 1

    # Now for each non-pivot row, express it as XOR of pivot rows
    for i in range(m):
        if outputs[i] is not None:
            continue
        # Find which pivot rows combine to give row i
        wire = None
        for pr in pivot_rows:
            if coeffs[i, pr]:
                if wire is None:
                    wire = pivot_wires[pr]
                else:
                    wire = get_xor_wire(wire, pivot_wires[pr])
        if wire is None:
            wire = n
        outputs[i] = wire

    return {'gates': gates, 'outputs': outputs}


def _local_search(circuit, matrix, n, max_iter=50):
    """
    Local search: try to reduce gate count by finding redundant gates
    or alternative decompositions.
    """
    gates = [g[:] for g in circuit['gates']]
    outputs = circuit['outputs'][:]
    m = matrix.shape[0]

    # Compute wire masks
    def compute_masks(gates):
        wire_mask = {}
        for i in range(n):
            wire_mask[i] = (1 << i)
        wire_mask[n] = 0
        for k, (a, b) in enumerate(gates):
            wire = n + 1 + k
            wire_mask[wire] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)
        return wire_mask

    best_gates = gates[:]
    best_count = len(gates)

    for iteration in range(max_iter):
        improved = False
        wire_mask = compute_masks(best_gates)

        # Try removing each gate if its wire is not used by any output or other gate
        used_wires = set()
        for g in best_gates:
            used_wires.add(g[0])
            used_wires.add(g[1])
        for out in outputs:
            used_wires.add(out)

        for k in range(len(best_gates)):
            wire = n + 1 + k
            if wire not in used_wires:
                # This gate is unused, remove it
                new_gates = best_gates[:k] + best_gates[k+1:]
                # Renumber all wires above k
                remap = {}
                for old_w in range(n + 1):
                    remap[old_w] = old_w
                for old_w in range(n + 1 + len(best_gates)):
                    if old_w >= n + 1 + k:
                        remap[old_w] = old_w - 1
                    else:
                        remap[old_w] = old_w

                new_gates2 = []
                for (a, b) in new_gates:
                    new_gates2.append([remap[a], remap[b]])
                new_outputs = [remap[o] for o in outputs]

                # Verify
                if _verify_circuit(matrix, {'gates': new_gates2, 'outputs': new_outputs}, 16):
                    best_gates = new_gates2
                    outputs = new_outputs
                    best_count = len(best_gates)
                    improved = True
                    break

        if not improved:
            break

    return {'gates': best_gates, 'outputs': outputs}


def _fallback_chain(matrix, n):
    """Simple chain construction as a guaranteed-correct fallback."""
    m = matrix.shape[0]
    gates = []
    outputs = []
    for i in range(m):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            wire = ones[0]
            for j in range(1, len(ones)):
                a = wire
                b = ones[j]
                wire = n + 1 + len(gates)
                gates.append([a, b])
            outputs.append(wire)
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

    # Strategy 1: Direct with CSE
    if time.time() < deadline - 5:
        c1 = _build_direct(matrix, n)
        candidates.append(c1)

    # Strategy 2: Incremental with row ordering
    if time.time() < deadline - 5:
        c2 = _build_incremental(matrix, n)
        candidates.append(c2)

    # Strategy 3: Common factors
    if time.time() < deadline - 5:
        c3 = _build_common_factors(matrix, n)
        candidates.append(c3)

    # Strategy 4: Pattern grouped
    if time.time() < deadline - 5:
        c4 = _build_pattern_grouped(matrix, n)
        candidates.append(c4)

    # Strategy 5: Basis reduction
    if time.time() < deadline - 5:
        c5 = _build_basis(matrix, n)
        candidates.append(c5)

    # Pick the best
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        count = len(c['gates'])
        if count < best_count:
            best_count = count
            best_circuit = c

    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)

    # Local search refinement
    if time.time() < deadline - 3:
        refined = _local_search(best_circuit, matrix, n, max_iter=30)
        if len(refined['gates']) < len(best_circuit['gates']):
            best_circuit = refined

    # Verify
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        found = False
        for c in candidates:
            if _verify_circuit(matrix, c, depth_limit):
                best_circuit = c
                found = True
                break
        if not found:
            best_circuit = _fallback_chain(matrix, n)

    # Final safety check
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        best_circuit = _fallback_chain(matrix, n)

    return best_circuit
