"""Exact XOR-circuit synthesis with bounded candidate interface."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _int_to_bits(x, n):
    """Convert integer to list of n bits (LSB first)."""
    return [(x >> i) & 1 for i in range(n)]

def _bits_to_int(bits):
    """Convert list of bits to integer."""
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= (1 << i)
    return v

def _xor_mask(a, b):
    """XOR two bitmasks (integers)."""
    return a ^ b

def _popcount(x):
    """Count set bits."""
    return bin(x).count('1')

def _row_to_int(row, n):
    """Convert a row (list/array of 0/1) to an integer bitmask."""
    v = 0
    for i in range(n):
        if row[i]:
            v |= (1 << i)
    return v

def _baseline_gates(matrix, n, m):
    """
    Baseline: each output row is XOR of its set input wires.
    Gate count = sum of popcounts of all rows.
    This is the trivial circuit with no sharing.
    """
    total = 0
    for r in range(m):
        row = matrix[r]
        total += _popcount(_row_to_int(row, n))
    return total

def _verify_gates(matrix, n, m, gates, outputs, depth_limit):
    """
    Verify that the gate list and outputs correctly compute the matrix over GF(2).
    Returns True if valid.
    """
    # wires[0..n-1] = inputs, wires[n] = 0 (constant zero)
    # gate k produces wire n+1+k
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    wire_val = [0] * total_wires
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0  # constant zero

    for k, (a, b) in enumerate(gates):
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

    for r in range(m):
        expected = _row_to_int(matrix[r], n)
        if r < len(outputs):
            if wire_val[outputs[r]] != expected:
                return False
    return True

def _check_depth(gates, n):
    """Check that the circuit depth is within limit. Returns depth."""
    if not gates:
        return 0
    num_gates = len(gates)
    # depth[wire] = longest path from input to this wire
    depth = [0] * (n + 1 + num_gates)
    for i in range(n):
        depth[i] = 0
    depth[n] = 0
    for k in range(num_gates):
        a, b = gates[k]
        wire = n + 1 + k
        depth[wire] = 1 + max(depth[a], depth[b])
    max_depth = max(depth[n+1:n+1+num_gates]) if num_gates > 0 else 0
    return max_depth

def _greedy_synthesis(matrix, n, m, depth_limit, time_budget):
    """
    Greedy synthesis: iteratively find the most useful XOR pair among available wires.
    A pair (a, b) is useful if a^b equals a target row or helps form one.
    """
    start_time = time.time()
    zero_wire = n
    gates = []
    # Available wires: start with inputs + zero
    available = list(range(n + 1))  # wires 0..n
    wire_val = [0] * (n + 1)
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0

    # Target masks
    targets = [_row_to_int(matrix[r], n) for r in range(m)]
    target_set = set(targets)

    # Track which targets are already available as wires
    wire_to_target = {}
    for w in available:
        if wire_val[w] in target_set:
            wire_to_target.setdefault(wire_val[w], []).append(w)

    # For each target, track the best way to form it
    # We'll use a priority-based approach: at each step, find the XOR pair
    # that reduces the total cost the most.

    # Cost model: cost to realize a mask using available wires
    # Initially, cost(mask) = popcount(mask) (XOR of individual inputs)
    # When we create a new wire w = a^b, we can potentially reduce costs.

    def cost_of_mask(mask, wire_vals, wire_map):
        """Estimate cost to realize mask. If mask is in wire_map, cost is 1 (just use that wire).
        Otherwise, greedy: find best split."""
        if mask == 0:
            return 0
        if mask in wire_map:
            return 1
        # Fallback: popcount
        return _popcount(mask)

    # We'll do iterative improvement
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break

        # Find the most beneficial pair to XOR
        # Benefit = reduction in total cost across all targets
        best_pair = None
        best_benefit = 0

        # Try all pairs of available wires
        num_avail = len(available)
        for i in range(num_avail):
            wi = available[i]
            for j in range(i + 1, num_avail):
                wj = available[j]
                new_mask = wire_val[wi] ^ wire_val[wj]
                if new_mask == 0:
                    continue

                # Check if this new mask is useful
                # Benefit: for each target, does having this new wire reduce its cost?
                benefit = 0
                for t in range(m):
                    mask = targets[t]
                    if mask in wire_to_target:
                        continue  # already available
                    # Cost without new wire
                    old_cost = cost_of_mask(mask, wire_val, wire_to_target)
                    # Cost with new wire: can we use new_mask to help?
                    # If mask ^ new_mask has lower popcount, we save
                    remainder = mask ^ new_mask
                    if _popcount(remainder) < old_cost:
                        # New cost would be 1 (for new wire) + cost(remainder)
                        # But remainder might also be available... approximate
                        new_cost = 1 + min(_popcount(remainder), cost_of_mask(remainder, wire_val, wire_to_target))
                        if new_cost < old_cost:
                            benefit += old_cost - new_cost
                    # Also check if new_mask itself is a target
                    if new_mask in target_set and new_mask not in wire_to_target:
                        benefit += _popcount(new_mask) - 1

                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (wi, wj, new_mask)

        if best_pair is None or best_benefit <= 0:
            break

        wi, wj, new_mask = best_pair
        new_wire = n + 1 + len(gates)
        gates.append([wi, wj])
        available.append(new_wire)
        wire_val.append(new_mask)
        if new_mask in target_set:
            wire_to_target.setdefault(new_mask, []).append(new_wire)

        # Check depth
        if len(gates) > 0:
            # Quick depth check
            if _check_depth(gates, n) > depth_limit:
                gates.pop()
                available.pop()
                wire_val.pop()
                if new_mask in wire_to_target:
                    wire_to_target[new_mask].pop()
                break

    return gates, wire_val, available

def _complete_gates(gates, matrix, n, m, wire_val, available, time_budget):
    """
    Complete the synthesis: for each target not yet available, add gates to form it.
    """
    start_time = time.time()
    zero_wire = n
    targets = [_row_to_int(matrix[r], n) for r in range(m)]
    wire_to_target = {}
    for w in available:
        if wire_val[w] in targets:
            wire_to_target.setdefault(wire_val[w], []).append(w)

    outputs = [None] * m
    for r in range(m):
        if targets[r] in wire_to_target:
            outputs[r] = wire_to_target[targets[r]][0]

    # For remaining targets, decompose into XOR of available wires
    for r in range(m):
        if outputs[r] is not None:
            continue
        if time.time() - start_time > time_budget:
            break

        mask = targets[r]
        # Greedy decomposition: repeatedly XOR with the available wire that
        # reduces popcount the most
        current = mask
        used_wires = []
        while current != 0:
            if time.time() - start_time > time_budget:
                break
            best_w = None
            best_reduction = 0
            for w in available:
                nv = current ^ wire_val[w]
                if nv == current:
                    continue
                reduction = _popcount(current) - _popcount(nv)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_w = w
            if best_w is None or best_reduction <= 0:
                # Can't reduce further with available wires; use individual inputs
                # Decompose current into individual bits
                for i in range(n):
                    if (current >> i) & 1:
                        used_wires.append(i)
                current = 0
                break
            used_wires.append(best_w)
            current ^= wire_val[best_w]

        if current != 0:
            # Shouldn't happen, but safety
            for i in range(n):
                if (current >> i) & 1:
                    used_wires.append(i)

        # Now XOR all used_wires together to form the output
        if len(used_wires) == 0:
            outputs[r] = zero_wire
        elif len(used_wires) == 1:
            outputs[r] = used_wires[0]
        else:
            # Chain XOR
            acc = used_wires[0]
            for w in used_wires[1:]:
                new_wire = n + 1 + len(gates)
                gates.append([acc, w])
                wire_val.append(wire_val[acc] ^ wire_val[w])
                available.append(new_wire)
                acc = new_wire
            outputs[r] = acc

    # Check depth
    max_depth = _check_depth(gates, n)
    if max_depth > depth_limit:
        # Fallback: use baseline (no sharing, just individual inputs)
        gates = []
        wire_val = [0] * (n + 1)
        for i in range(n):
            wire_val[i] = (1 << i)
        wire_val[n] = 0
        available = list(range(n + 1))
        outputs = [None] * m
        for r in range(m):
            mask = targets[r]
            bits = [i for i in range(n) if (mask >> i) & 1]
            if not bits:
                outputs[r] = zero_wire
            elif len(bits) == 1:
                outputs[r] = bits[0]
            else:
                acc = bits[0]
                for b in bits[1:]:
                    new_wire = n + 1 + len(gates)
                    gates.append([acc, b])
                    wire_val.append(wire_val[acc] ^ wire_val[b])
                    available.append(new_wire)
                    acc = new_wire
                outputs[r] = acc

    return gates, outputs

def _try_improve(gates, matrix, n, m, depth_limit, time_budget):
    """
    Try to reduce gate count by finding redundant gates or better decompositions.
    """
    start_time = time.time()
    targets = [_row_to_int(matrix[r], n) for r in range(m)]
    zero_wire = n

    # Rebuild wire values
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    wire_val = [0] * total_wires
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0
    for k in range(num_gates):
        a, b = gates[k]
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

    # Try to find if any gate's output is never used (except as intermediate)
    # Build usage count
    usage = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        usage[a] += 1
        usage[b] += 1
    # Outputs usage
    # We don't have outputs here, so skip this optimization

    # Try local search: for each gate, see if we can replace it with a cheaper path
    # This is complex; skip for now and rely on the greedy approach

    return gates

def _baseline_circuit(matrix, n, m):
    """Generate the baseline circuit (no sharing)."""
    gates = []
    wire_val = [0] * (n + 1)
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0
    available = list(range(n + 1))
    zero_wire = n

    outputs = [None] * m
    for r in range(m):
        mask = _row_to_int(matrix[r], n)
        bits = [i for i in range(n) if (mask >> i) & 1]
        if not bits:
            outputs[r] = zero_wire
        elif len(bits) == 1:
            outputs[r] = bits[0]
        else:
            acc = bits[0]
            for b in bits[1:]:
                new_wire = n + 1 + len(gates)
                gates.append([acc, b])
                wire_val.append(wire_val[acc] ^ wire_val[b])
                available.append(new_wire)
                acc = new_wire
            outputs[r] = acc

    return gates, outputs

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given matrix.
    
    Args:
        matrix: 2D array-like of shape (m, n) with entries in {0, 1}
        depth_limit: maximum allowed circuit depth
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b]) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Convert matrix to list of lists of ints
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Ensure matrix entries are 0 or 1
    matrix = [[int(x) & 1 for x in row] for row in matrix]

    # Baseline
    base_gates, base_outputs = _baseline_circuit(matrix, n, m)
    base_cost = len(base_gates)

    # Try greedy synthesis with time budget
    time_budget = min(40.0, DEADLINE - (time.time() - start_time))
    if time_budget > 5:
        gates, wire_val, available = _greedy_synthesis(matrix, n, m, depth_limit, time_budget)
        gates, outputs = _complete_gates(gates, matrix, n, m, wire_val, available, 
                                          min(15.0, DEADLINE - (time.time() - start_time)))
        
        # Verify
        if _verify_gates(matrix, n, m, gates, outputs, depth_limit):
            if _check_depth(gates, n) <= depth_limit:
                if len(gates) < base_cost:
                    return {'gates': gates, 'outputs': outputs}

    # Fallback to baseline
    if _check_depth(base_gates, n) <= depth_limit:
        return {'gates': base_gates, 'outputs': base_outputs}

    # If baseline exceeds depth, we need a different approach
    # Use a balanced tree for each row
    gates = []
    wire_val = [0] * (n + 1)
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0
    available = list(range(n + 1))
    zero_wire = n

    outputs = [None] * m
    for r in range(m):
        mask = _row_to_int(matrix[r], n)
        bits = [i for i in range(n) if (mask >> i) & 1]
        if not bits:
            outputs[r] = zero_wire
        elif len(bits) == 1:
            outputs[r] = bits[0]
        else:
            # Balanced tree
            level = list(bits)
            while len(level) > 1:
                next_level = []
                for i in range(0, len(level) - 1, 2):
                    a, b = level[i], level[i + 1]
                    new_wire = n + 1 + len(gates)
                    gates.append([a, b])
                    wire_val.append(wire_val[a] ^ wire_val[b])
                    available.append(new_wire)
                    next_level.append(new_wire)
                if len(level) % 2 == 1:
                    next_level.append(level[-1])
                level = next_level
            outputs[r] = level[0]

    # Verify
    if _verify_gates(matrix, n, m, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}

    # Ultimate fallback: should not happen
    return {'gates': base_gates, 'outputs': base_outputs}

# EVOLVE-BLOCK-END
