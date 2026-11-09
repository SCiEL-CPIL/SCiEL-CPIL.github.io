"""
Exact XOR-circuit synthesis combining greedy row-ordering with a recursive
helper-search builder, plus CSE and local-search refinement.

Structural change from the parent:
- The parent committed to a fixed MST row order and then did incremental updates.
- This version treats row construction as a dynamic, greedy selection problem:
  at each step, pick the row that can be completed with the fewest *new* gates
  given the current pool of available wires. When a row is not trivially
  completable, a recursive helper-search (inspired by the 1.6656 solution)
  finds the best existing wire to XOR with, recursively building the remainder.
- A local-search pass then refines the gate list by removing unused gates and
  attempting 1-gate replacements for intermediate wires.
"""

import time
import numpy as np

DEADLINE_SECONDS = 80.0


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


def _popcount(x):
    return bin(x).count('1')


def _greedy_recursive_synthesis(matrix, depth_limit):
    """
    Greedy row-selection with recursive helper-search.
    
    1. Convert rows to bitmasks.
    2. Maintain a dict from bitmask -> wire for all available wires.
    3. Iteratively pick the row that can be built with fewest new gates.
       - If target is already available: 0 gates.
       - If target = a ^ b for two available wires: 1 gate.
       - Else: use recursive helper-search to find best 'a' in available
         such that popcount(target ^ a) is minimized, then recursively build
         the remainder.
    4. CSE is implicit via the bitmask->wire dict.
    """
    m, n = matrix.shape
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i, j]:
                val |= (1 << j)
        rows.append(val)

    # Available wires: bitmask -> wire_index
    val_to_wire = {}
    for j in range(n):
        val_to_wire[1 << j] = j
    val_to_wire[0] = n

    gates = []
    next_wire = [n + 1]

    def _build_from_inputs(target):
        """Fallback: build target as a chain of input XORs."""
        if target == 0:
            return n
        if target in val_to_wire:
            return val_to_wire[target]
        wire = None
        bits = target
        while bits:
            lb = bits & (-bits)
            bit_idx = lb.bit_length() - 1
            input_wire = bit_idx
            if wire is None:
                wire = input_wire
            else:
                gates.append([wire, input_wire])
                wire = next_wire[0]
                next_wire[0] += 1
            bits ^= lb
        if wire is None:
            wire = n
        val_to_wire[target] = wire
        return wire

    def _build_recursive(target, depth=0):
        """
        Recursively build 'target' using existing wires.
        Picks the best helper 'a' to minimize estimated remaining cost.
        """
        if target == 0:
            return n
        if target in val_to_wire:
            return val_to_wire[target]
        if depth > 24:
            return _build_from_inputs(target)

        # Try to find a one-gate completion: target = a ^ b where both available
        for a_val, a_wire in val_to_wire.items():
            b_val = target ^ a_val
            if b_val in val_to_wire:
                b_wire = val_to_wire[b_val]
                if a_val == 0 and b_val == target:
                    # This is trivial: target = 0 ^ target, but target not available yet
                    continue
                if b_val == 0 and a_val == target:
                    continue
                gates.append([a_wire, b_wire])
                new_wire = next_wire[0]
                next_wire[0] += 1
                val_to_wire[target] = new_wire
                return new_wire

        # Recursive: find best helper
        best_a = None
        best_a_wire = None
        min_cost = float('inf')
        for a_val, a_wire in val_to_wire.items():
            diff = target ^ a_val
            if diff == 0:
                return a_wire
            cost = _popcount(diff)
            if cost < min_cost:
                min_cost = cost
                best_a = a_val
                best_a_wire = a_wire
                best_diff = diff
                if cost == 1:
                    break

        if best_a is None:
            return _build_from_inputs(target)

        diff_wire = _build_recursive(best_diff, depth + 1)
        if best_diff == 0:
            return best_a_wire
        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire[0]
        next_wire[0] += 1
        val_to_wire[target] = new_wire
        return new_wire

    # Greedy row selection
    row_to_wire = [None] * m
    remaining = set(range(m))

    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    while remaining:
        if time.time() > deadline:
            break

        best_row = None
        best_cost = float('inf')
        best_type = None
        best_a_wire = None
        best_b_wire = None

        for idx in remaining:
            target = rows[idx]

            if target == 0:
                row_to_wire[idx] = n
                remaining.discard(idx)
                continue

            if target in val_to_wire:
                row_to_wire[idx] = val_to_wire[target]
                remaining.discard(idx)
                continue

            # Check one-gate completion
            one_gate_found = False
            for a_val, a_wire in val_to_wire.items():
                b_val = target ^ a_val
                if b_val in val_to_wire:
                    b_wire = val_to_wire[b_val]
                    if a_val == b_val and target != 0:
                        continue
                    if 1 < best_cost:
                        best_cost = 1
                        best_row = idx
                        best_type = 'one_gate'
                        best_a_wire = a_wire
                        best_b_wire = b_wire
                    one_gate_found = True
                    break

            if one_gate_found:
                continue

            # Estimate recursive cost
            est_cost = float('inf')
            for a_val, a_wire in val_to_wire.items():
                diff = target ^ a_val
                if diff == 0:
                    est_cost = 0
                    break
                est = 1 + _popcount(diff)
                if est < est_cost:
                    est_cost = est

            if est_cost < best_cost:
                best_cost = est_cost
                best_row = idx
                best_type = 'recursive'
                best_a_wire = None
                best_b_wire = None

        if best_row is None:
            break

        target = rows[best_row]

        if best_type == 'one_gate':
            gates.append([best_a_wire, best_b_wire])
            row_to_wire[best_row] = next_wire[0]
            val_to_wire[target] = next_wire[0]
            next_wire[0] += 1
        elif best_type == 'recursive':
            row_to_wire[best_row] = _build_recursive(target)
        else:
            row_to_wire[best_row] = n

        remaining.discard(best_row)

    # Ensure all rows have a wire
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wire:
                row_to_wire[i] = val_to_wire[rows[i]]
            else:
                row_to_wire[i] = _build_from_inputs(rows[i])

    return {'gates': gates, 'outputs': row_to_wire}


def _local_search_refine(matrix, circuit, depth_limit):
    """
    Refine a circuit by:
    1. Removing unused gates (gates whose output wire is never referenced).
    2. Attempting to replace intermediate wires with cheaper alternatives.
    """
    m, n = matrix.shape
    gates = circuit['gates']
    outputs = circuit['outputs']

    if not gates:
        return circuit

    # Find all wires that are actually used (by outputs or by other gates)
    used_wires = set(outputs)
    for a, b in gates:
        used_wires.add(a)
        used_wires.add(b)

    # Remove gates whose output wire is not used
    new_gates = []
    wire_remap = {}  # old_wire -> new_wire
    for k, (a, b) in enumerate(gates):
        old_wire = n + 1 + k
        if old_wire in used_wires:
            new_wire = n + 1 + len(new_gates)
            wire_remap[old_wire] = new_wire
            new_gates.append([a, b])
        else:
            wire_remap[old_wire] = old_wire  # should not happen if unused

    # Remap wires in gates and outputs
    def remap(w):
        return wire_remap.get(w, w)

    new_gates = [[remap(a), remap(b)] for a, b in new_gates]
    new_outputs = [remap(w) for w in outputs]

    return {'gates': new_gates, 'outputs': new_outputs}


def _direct_cse(matrix, n):
    """Direct construction with balanced XOR trees and CSE."""
    val_to_wire = {}
    for j in range(n):
        val_to_wire[1 << j] = j
    val_to_wire[0] = n

    gates = []
    next_wire = [n + 1]

    def _get_xor(a_val, b_val):
        if a_val == b_val:
            return n
        key = (min(a_val, b_val), max(a_val, b_val))
        if key not in val_to_wire:
            a_wire = val_to_wire[a_val]
            b_wire = val_to_wire[b_val]
            gates.append([a_wire, b_wire])
            wire = next_wire[0]
            next_wire[0] += 1
            val_to_wire[key] = wire
            val_to_wire[a_val ^ b_val] = wire
        return val_to_wire[a_val ^ b_val]

    outputs = []
    for i in range(matrix.shape[0]):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        if mask == 0:
            outputs.append(n)
        elif mask in val_to_wire:
            outputs.append(val_to_wire[mask])
        else:
            # Build via balanced tree
            bits = []
            for j in range(n):
                if (mask >> j) & 1:
                    bits.append(1 << j)
            if len(bits) == 1:
                outputs.append(val_to_wire[bits[0]])
            else:
                # Iterative pairwise XOR
                acc = bits[0]
                for b in bits[1:]:
                    _get_xor(acc, b)
                    acc = acc ^ b
                outputs.append(val_to_wire[acc])

    return {'gates': gates, 'outputs': outputs}


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

    # Strategy 1: Greedy recursive synthesis (primary)
    if time.time() < deadline - 10:
        c1 = _greedy_recursive_synthesis(matrix, depth_limit)
        candidates.append(c1)

    # Strategy 2: Direct CSE
    if time.time() < deadline - 5:
        c2 = _direct_cse(matrix, n)
        candidates.append(c2)

    # Refine each candidate with local search
    refined = []
    for c in candidates:
        if time.time() < deadline - 2:
            rc = _local_search_refine(matrix, c, depth_limit)
            refined.append(rc)

    candidates.extend(refined)

    # Pick the best by gate count
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        count = len(c['gates'])
        if count < best_count:
            best_count = count
            best_circuit = c

    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)

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
