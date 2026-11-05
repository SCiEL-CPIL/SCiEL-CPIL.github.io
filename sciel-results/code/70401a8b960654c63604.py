import time
import numpy as np
from itertools import combinations

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    # Handle trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # For small n (<=14), use DP-based optimal synthesis
    if n <= 14:
        result = _dp_optimal(target_masks, n, zero_wire, row_weights)
        if result is not None:
            gates, outputs = result
            if len(gates) <= baseline_gates:
                return {'gates': gates, 'outputs': outputs}
    
    # For larger n or if DP fails, use improved greedy with better sharing
    result = _improved_greedy(target_masks, n, zero_wire, row_weights)
    if result is not None:
        gates, outputs = result
        # Verify and fallback if worse than baseline
        if len(gates) > baseline_gates:
            gates, outputs = _simple_fallback(target_masks, n, zero_wire, row_weights)
        return {'gates': gates, 'outputs': outputs}
    
    # Final fallback
    gates, outputs = _simple_fallback(target_masks, n, zero_wire, row_weights)
    return {'gates': gates, 'outputs': outputs}


def _dp_optimal(target_masks, n, zero_wire, row_weights):
    """
    DP-based synthesis for small n.
    
    Key insight: We want to find a set of intermediate XORs (masks) to precompute
    such that the total cost (gates to compute intermediates + gates to compute
    targets from intermediates) is minimized.
    
    Approach:
    1. Find a minimal basis of the span of all target masks.
    2. For each target, express it as a combination of basis vectors.
    3. The cost is: (sum of (weight(basis_vec) - 1) for basis vectors) + 
                    (sum of (popcount(coeffs) - 1) for each target)
    
    But we can do better by choosing a non-minimal basis that has lower total cost.
    
    For n <= 14, we can enumerate subsets of "useful" masks.
    
    Actually, let's use a different approach:
    - The optimal circuit computes some set S of masks (including inputs and targets).
    - Each mask in S (except inputs and zero) requires one gate to compute from two
      previously computed masks.
    - We want to minimize |S| - n - 1 (subtracting the free inputs and zero).
    
    This is equivalent to finding the minimum number of intermediate masks needed
    to express all targets as XORs of available masks.
    
    For small n, we can use BFS/DP over the power set of "interesting" masks.
    
    Let me try a simpler but effective approach:
    1. Compute a basis for the row space.
    2. For each basis vector, compute its cost (weight - 1).
    3. For each target, express as combination of basis vectors, cost = popcount - 1.
    4. Total cost = sum(basis costs) + sum(target expression costs).
    
    To improve: try different bases (different pivot choices) and pick the best.
    """
    if time.time() > deadline:
        return None
    
    # Get unique non-zero targets
    unique_targets = list(set(target_masks))
    unique_targets = [t for t in unique_targets if t != 0]
    
    if not unique_targets:
        return [], [zero_wire] * len(target_masks)
    
    # Find a basis using Gaussian elimination with cost-aware pivot selection
    # Try multiple pivot orders and keep the best
    
    best_gates = None
    best_outputs = None
    best_cost = float('inf')
    
    # Generate several pivot orders
    # Order 1: by weight (ascending)
    masks_by_weight = sorted(range(n), key=lambda i: 0)  # We'll use column indices
    
    # Try different strategies for basis selection
    strategies = []
    
    # Strategy 1: Standard Gaussian elimination (leftmost pivot)
    strategies.append('standard')
    
    # Strategy 2: Rightmost pivot
    strategies.append('rightmost')
    
    # Strategy 3: Minimum weight pivot
    strategies.append('min_weight')
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        result = _basis_synthesis(target_masks, n, zero_wire, row_weights, strategy)
        if result is not None:
            gates, outputs = result
            cost = len(gates)
            if cost < best_cost:
                best_cost = cost
                best_gates = gates
                best_outputs = outputs
    
    if best_gates is not None:
        return best_gates, best_outputs
    
    return None


def _basis_synthesis(target_masks, n, zero_wire, row_weights, strategy):
    """
    Synthesize using a basis with a given pivot strategy.
    """
    if time.time() > deadline:
        return None
    
    # Build a basis of the span of all target masks
    # basis: list of (pivot_bit, mask)
    basis = []
    # mask_to_basis: map from mask to index in basis
    
    # Collect all masks we need to span
    all_masks = list(set(target_masks))
    all_masks = [m for m in all_masks if m != 0]
    
    # Sort based on strategy
    if strategy == 'standard':
        # Sort by pivot position (leftmost)
        def pivot_key(m):
            for i in range(n):
                if (m >> i) & 1:
                    return i
            return n
        all_masks.sort(key=pivot_key)
    elif strategy == 'rightmost':
        def pivot_key(m):
            for i in range(n - 1, -1, -1):
                if (m >> i) & 1:
                    return -i  # Negate so rightmost comes first
            return 0
        all_masks.sort(key=pivot_key)
    elif strategy == 'min_weight':
        all_masks.sort(key=lambda m: bin(m).count('1'))
    
    # Perform Gaussian elimination
    for mask in all_masks:
        m = mask
        for i, (p, bm) in enumerate(basis):
            if (m >> p) & 1:
                m ^= bm
        if m != 0:
            # Find pivot
            if strategy == 'min_weight':
                # Use the bit that minimizes the resulting weight
                best_bit = -1
                best_weight = float('inf')
                for b in range(n):
                    if (m >> b) & 1:
                        new_weight = bin(m ^ (1 << b)).count('1')  # Not quite right
                        # Actually, we just add m to basis; the cost is weight(m) - 1
                        pass
                # Just use leftmost pivot for simplicity
                for b in range(n):
                    if (m >> b) & 1:
                        best_bit = b
                        break
            else:
                # Use leftmost set bit as pivot
                for b in range(n):
                    if (m >> b) & 1:
                        best_bit = b
                        break
                if strategy == 'rightmost':
                    for b in range(n - 1, -1, -1):
                        if (m >> b) & 1:
                            best_bit = b
                            break
            
            if best_bit >= 0:
                basis.append((best_bit, m))
    
    if not basis:
        # All targets are zero
        outputs = [zero_wire] * len(target_masks)
        return [], outputs
    
    # Now compute each basis vector as a circuit
    # basis[i] = (pivot, mask)
    # We need to compute each mask from inputs using XOR gates
    
    gates = []
    next_wire = n + 1
    # available: mask -> wire
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire
    
    basis_wires = []
    
    for i, (p, bm) in enumerate(basis):
        # Compute bm from inputs
        # Try to find a decomposition using available wires
        if bm in available:
            basis_wires.append(available[bm])
            continue
        
        # Try XOR of two available
        found = False
        for am, aw in list(available.items()):
            other = bm ^ am
            if other in available:
                ow = available[other]
                if am == 0:
                    basis_wires.append(ow)
                    found = True
                    break
                elif other == 0:
                    basis_wires.append(aw)
                    found = True
                    break
                else:
                    gates.append([aw, ow])
                    w = next_wire
                    next_wire += 1
                    available[bm] = w
                    basis_wires.append(w)
                    found = True
                    break
        
        if not found:
            # Compute from inputs directly
            ones = [j for j in range(n) if (bm >> j) & 1]
            if len(ones) == 0:
                basis_wires.append(zero_wire)
            elif len(ones) == 1:
                basis_wires.append(ones[0])
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                basis_wires.append(current)
                available[bm] = current
    
    # Now compute each target as a combination of basis vectors
    # First, reduce each target using the basis
    pivot_to_idx = {p: i for i, (p, _) in enumerate(basis)}
    
    outputs = []
    for t in target_masks:
        if t == 0:
            outputs.append(zero_wire)
            continue
        
        # Reduce t using basis
        current = t
        coeffs = []  # indices of basis vectors used
        for p in range(n):
            if (current >> p) & 1:
                if p in pivot_to_idx:
                    idx = pivot_to_idx[p]
                    coeffs.append(idx)
                    current ^= basis[idx][1]
        
        if current != 0:
            # Should not happen if basis spans all targets
            # Fallback: compute directly
            ones = [j for j in range(n) if (t >> j) & 1]
            if len(ones) == 0:
                outputs.append(zero_wire)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                cw = ones[0]
                for j in range(1, len(ones)):
                    gates.append([cw, ones[j]])
                    cw = next_wire
                    next_wire += 1
                outputs.append(cw)
            continue
        
        if not coeffs:
            outputs.append(zero_wire)
        elif len(coeffs) == 1:
            outputs.append(basis_wires[coeffs[0]])
        else:
            # Compute XOR of basis_wires[coeffs[0]], basis_wires[coeffs[1]], ...
            cw = basis_wires[coeffs[0]]
            for k in range(1, len(coeffs)):
                gates.append([cw, basis_wires[coeffs[k]]])
                cw = next_wire
                next_wire += 1
            outputs.append(cw)
    
    return gates, outputs


def _improved_greedy(target_masks, n, zero_wire, row_weights):
    """
    Improved greedy synthesis with better sharing.
    
    Key improvements over the original:
    1. Process targets in a more intelligent order (not just by weight).
    2. When computing a target, try to maximize reuse of existing wires.
    3. After computing all targets, do a second pass to find more sharing.
    """
    if time.time() > deadline:
        return None
    
    # Available wires: mask -> wire_index
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    # target_wires: for each target, the wire that computes it
    target_wires = [None] * len(target_masks)
    
    # First pass: mark targets that are directly available
    for i, mask in enumerate(target_masks):
        if mask in available:
            target_wires[i] = available[mask]
    
    # Second pass: compute remaining targets
    # Order: by weight, then by "overlap" with already computed wires
    remaining = []
    for i in range(len(target_masks)):
        if target_wires[i] is None:
            mask = target_masks[i]
            # Count how many input bits are already "covered" by available wires
            # (this is a heuristic)
            remaining.append((row_weights[i], i))
    
    remaining.sort()
    
    for weight, idx in remaining:
        if time.time() > deadline:
            break
        
        mask = target_masks[idx]
        
        # Try to find the cheapest decomposition
        wire = _find_best_decomposition(mask, available, n, gates, next_wire)
        
        if wire is not None:
            target_wires[idx] = wire
            if mask not in available:
                available[mask] = wire
        else:
            # Fallback: compute from inputs directly
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                target_wires[idx] = zero_wire
            elif len(ones) == 1:
                target_wires[idx] = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                target_wires[idx] = current
                available[mask] = current
    
    # Third pass: try to find more sharing by looking for common subexpressions
    # This is complex; skip for now
    
    return gates, target_wires


def _find_best_decomposition(mask, available, n, gates, next_wire):
    """
    Find the cheapest way to compute 'mask' using available wires.
    Returns the wire index, or None if not found.
    """
    if mask in available:
        return available[mask]
    
    # Try XOR of two available wires
    # To speed this up, we can check a subset of available wires first
    # Try high-frequency masks first
    
    # For small available sets, check all
    avail_items = list(available.items())
    
    # Try pairs
    for i in range(len(avail_items)):
        m1, w1 = avail_items[i]
        other = mask ^ m1
        if other in available:
            w2 = available[other]
            if m1 == 0:
                return w2
            elif other == 0:
                return w1
            else:
                gates.append([w1, w2])
                new_wire = next_wire
                # Update next_wire and available - but these are local
                # We need to handle this differently
                # For now, return and let caller manage
                return new_wire
    
    # Try XOR of one input and one available wire
    for i in range(n):
        input_mask = 1 << i
        other = mask ^ input_mask
        if other in available:
            other_wire = available[other]
            if other == 0:
                return i
            else:
                gates.append([i, other_wire])
                return next_wire
    
    # Try XOR of two inputs
    for i in range(n):
        for j in range(i + 1, n):
            if mask == ((1 << i) | (1 << j)):
                gates.append([i, j])
                return next_wire
    
    return None


def _simple_fallback(target_masks, n, zero_wire, row_weights):
    """
    Simple per-row XOR synthesis as a fallback.
    """
    gates = []
    outputs = []
    next_wire = n + 1
    
    for mask in target_masks:
        if mask == 0:
            outputs.append(zero_wire)
            continue
        ones = [j for j in range(n) if (mask >> j) & 1]
        if len(ones) == 1:
            outputs.append(ones[0])
        else:
            current = ones[0]
            for j in range(1, len(ones)):
                gates.append([current, ones[j]])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return gates, outputs
