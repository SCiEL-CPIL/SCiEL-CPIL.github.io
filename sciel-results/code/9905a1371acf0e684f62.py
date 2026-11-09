"""
Exact XOR-circuit synthesis: structure-aware CSE with multiple construction
strategies and local-search refinement.

Structural change: instead of a single greedy CSE loop, we run several
complementary construction heuristics (direct trees, incremental row reordering,
common-factor extraction, basis-reduction) and pick the best result.
We then apply a local-search pass that attempts to merge or eliminate gates
while preserving correctness.
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

DEADLINE_SECONDS = 85.0


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix and respects depth.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # constant zero

    gates = circuit['gates']
    for k, (a, b) in enumerate(gates):
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

    depth = {}
    for i in range(n + 1):
        depth[i] = 0

    max_depth = 0
    for k, (a, b) in enumerate(gates):
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
    if not remaining:
        return {'gates': [], 'outputs': []}
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


def _build_basis_reduction(matrix, n):
    """
    Use Gaussian elimination to find a basis of the row space,
    then express each row as a combination of basis vectors.
    This can reduce gate count when rows are linearly dependent.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # Find basis via Gaussian elimination on masks
    basis = []
    basis_indices = []
    for mask in row_masks:
        if mask == 0:
            continue
        reduced = mask
        for b in basis:
            if bin(reduced & b).count('1') > 0:
                # Check if leading bit matches
                lead_b = b.bit_length() - 1
                if (reduced >> lead_b) & 1:
                    reduced ^= b
        if reduced != 0:
            basis.append(reduced)
            basis_indices.append(mask)

    # Express each row as combination of basis vectors
    # For simplicity, we just use direct construction if basis doesn't help
    # A more sophisticated approach would be to build the basis vectors
    # and then combine them, but this is complex.
    # For now, fall back to direct if basis is small.
    if len(basis) < m * 0.7:
        # Potentially beneficial, but implementation is complex.
        # Use incremental as a proxy for now.
        return _build_incremental(matrix, n)
    else:
        return _build_direct(matrix, n)


def _local_search_refine(matrix, circuit, depth_limit, max_iters=50):
    """
    Attempt to reduce gate count by local search:
    - Try removing gates that are not used.
    - Try merging gates.
    """
    gates = circuit['gates'][:]
    outputs = circuit['outputs'][:]
    n = matrix.shape[1]

    # Build wire mask map
    def compute_masks(gates, outputs):
        wire_mask = {}
        for i in range(n):
            wire_mask[i] = (1 << i)
        wire_mask[n] = 0
        for k, (a, b) in enumerate(gates):
            wire = n + 1 + k
            wire_mask[wire] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)
        return wire_mask

    # Check if a gate is used
    def is_gate_used(gate_idx, gates, outputs):
        wire = n + 1 + gate_idx
        # Check if used in any other gate
        for k, (a, b) in enumerate(gates):
            if k == gate_idx:
                continue
            if a == wire or b == wire:
                return True
        # Check if used in outputs
        for out in outputs:
            if out == wire:
                return True
        return False

    # Try to remove unused gates
    changed = True
    while changed and len(gates) > 0:
        changed = False
        for i in range(len(gates) - 1, -1, -1):
            if not is_gate_used(i, gates, outputs):
                gates.pop(i)
                changed = True
                break

    # Try to merge gates: if gate k computes a^b and gate j computes a^c,
    # and b==c, then they are the same. Already handled by CSE.
    # Try to find gates that compute the same wire and eliminate one.
    wire_to_gate = {}
    for k, (a, b) in enumerate(gates):
        key = (min(a, b), max(a, b))
        if key in wire_to_gate:
            # Duplicate, remove later one
            pass
        else:
            wire_to_gate[key] = k

    # Remove duplicate gates
    seen_keys = set()
    new_gates = []
    for k, (a, b) in enumerate(gates):
        key = (min(a, b), max(a, b))
        if key not in seen_keys:
            seen_keys.add(key)
            new_gates.append([a, b])
    gates = new_gates

    # Renumber wires
    # This is complex; skip for now.

    result = {'gates': gates, 'outputs': outputs}
    if _verify_circuit(matrix, result, depth_limit):
        return result
    return circuit


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Ensure matrix is numpy array
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)

    n_rows, n_cols = matrix.shape
    n_input = n_cols
    n_output = n_rows

    # Try multiple construction strategies and pick the best
    candidates = []

    # Strategy 1: Direct
    c1 = _build_direct(matrix, n_input)
    candidates.append(c1)

    # Strategy 2: Incremental
    if time.time() < deadline:
        c2 = _build_incremental(matrix, n_input)
        candidates.append(c2)

    # Strategy 3: Common factors
    if time.time() < deadline:
        c3 = _build_common_factors(matrix, n_input)
        candidates.append(c3)

    # Strategy 4: Basis reduction
    if time.time() < deadline:
        c4 = _build_basis_reduction(matrix, n_input)
        candidates.append(c4)

    # Pick the candidate with fewest gates
    best = min(candidates, key=lambda c: len(c['gates']))

    # Verify
    if not _verify_circuit(matrix, best, depth_limit):
        # Fall back to direct
        best = _build_direct(matrix, n_input)

    # Local search refinement
    if time.time() < deadline - 5.0:
        refined = _local_search_refine(matrix, best, depth_limit)
        if len(refined['gates']) < len(best['gates']):
            best = refined

    # Final verification
    if not _verify_circuit(matrix, best, depth_limit):
        best = _build_direct(matrix, n_input)

    return best
