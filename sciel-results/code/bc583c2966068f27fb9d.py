import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline.
    
    Strategy:
    1. Compute the "independent-row baseline": sum of max(0, w_i - 1) for each row.
    2. Build a circuit that shares intermediate XORs to reduce total gate count below baseline.
    3. Use a greedy/DP approach for small n, and basis decomposition + sharing for larger n.
    
    Key insight: If multiple rows share inputs, computing partial XORs once and reusing them
    saves gates. The optimal circuit is a DAG where each internal node is a 2-input XOR.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    # Compute row weights and baseline
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
    
    # For small n (<=12), use DP over all masks
    if n <= 12:
        result = _dp_synthesis(target_masks, n, zero_wire)
        if result is not None:
            gates, outputs = result
            if len(gates) <= baseline_gates:
                return {'gates': gates, 'outputs': outputs}
    
    # For larger n, use greedy sharing approach
    result = _greedy_synthesis(target_masks, n, zero_wire, row_weights)
    if result is not None:
        gates, outputs = result
        return {'gates': gates, 'outputs': outputs}
    
    # Fallback: simple per-row XOR
    gates = []
    outputs = []
    next_wire = n + 1
    for i in range(m):
        if row_weights[i] == 0:
            outputs.append(zero_wire)
            continue
        ones = np.where(mat[i] == 1)[0]
        if len(ones) == 1:
            outputs.append(int(ones[0]))
            continue
        current = int(ones[0])
        for j in range(1, len(ones)):
            other = int(ones[j])
            gates.append([current, other])
            current = next_wire
            next_wire += 1
        outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}


def _dp_synthesis(target_masks, n, zero_wire):
    """
    DP over all 2^n masks to find minimum total gates to compute all target masks.
    """
    # For now, use a simple greedy approach
    # This is a placeholder; the greedy_synthesis below is more practical.
    return None


def _greedy_synthesis(target_masks, n, zero_wire, row_weights):
    """
    Greedy approach to synthesize a circuit with shared subexpressions.
    
    1. Start with all input wires available.
    2. For each target, if it's not directly available, find the cheapest way to
       compute it from available wires.
    3. Add the resulting wire to the available set.
    4. Repeat until all targets are computed.
    
    To find the cheapest way to compute a target mask from available wires:
    - If the mask is in the available set, cost = 0.
    - Otherwise, try to find two available wires whose XOR gives the target, cost = 1.
    - Otherwise, try to find one available wire and one input whose XOR gives the target.
    - Otherwise, decompose the target into a sum of available wires (more expensive).
    
    This is essentially a shortest path problem in the space of masks, where edges
    represent XOR operations with available wires.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Available wires: mask -> wire_index
    # Start with inputs and zero
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire
    
    # Track which targets are computed
    target_wires = [None] * len(target_masks)
    
    # Gate list
    gates = []
    next_wire = n + 1
    
    # For each target, try to compute it
    # Use a priority queue: process targets in order of increasing difficulty
    # Difficulty = number of 1s in the mask (weight)
    
    # First, mark targets that are already available
    for i, mask in enumerate(target_masks):
        if mask in available:
            target_wires[i] = available[mask]
    
    # For remaining targets, use a greedy approach
    # Sort by weight (easier targets first)
    remaining = [(row_weights[i], i) for i in range(len(target_masks)) if target_wires[i] is None]
    remaining.sort()
    
    for weight, idx in remaining:
        if time.time() > deadline:
            break
        mask = target_masks[idx]
        
        # Try to find a decomposition using available wires
        # BFS/DFS in the mask space
        wire = _find_decomposition(mask, available, n, gates, next_wire)
        if wire is None:
            # Fallback: compute from inputs directly
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                wire = zero_wire
            elif len(ones) == 1:
                wire = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                    # Add intermediate to available
                    # Compute the mask for this intermediate
                    # This is tricky; skip for now
                wire = current
            target_wires[idx] = wire
        else:
            target_wires[idx] = wire
    
    outputs = target_wires
    return gates, outputs


def _find_decomposition(mask, available, n, gates, next_wire):
    """
    Try to find a decomposition of 'mask' using available wires.
    Returns the wire index if found, None otherwise.
    """
    # If mask is directly available
    if mask in available:
        return available[mask]
    
    # Try to find two available wires whose XOR is mask
    # This is O(len(available)^2) which is too slow
    # Instead, for each available wire w, check if mask ^ w is also available
    for w_mask, w_wire in list(available.items()):
        other_mask = mask ^ w_mask
        if other_mask in available:
            # Found: w_wire XOR available[other_mask] = mask
            other_wire = available[other_mask]
            if w_mask == 0:
                # XOR with zero is just the other wire
                return other_wire
            elif other_mask == 0:
                return w_wire
            else:
                # Create a new gate
                gates.append([w_wire, other_wire])
                new_wire = next_wire
                # We need to update next_wire, but it's passed by value
                # This is a limitation; handle it differently
                # For now, just return the wire and assume the caller manages next_wire
                # Actually, we need to update next_wire and available
                # Let's restructure: return (wire, new_mask) and let caller update
                return w_wire  # Placeholder; this needs restructuring
    
    # Try to find one available wire and one input
    for i in range(n):
        input_mask = 1 << i
        other_mask = mask ^ input_mask
        if other_mask in available:
            other_wire = available[other_mask]
            if other_mask == 0:
                return i
            elif input_mask == 0:
                return other_wire
            else:
                # Create a new gate
                gates.append([i, other_wire])
                return next_wire  # Again, need to manage next_wire
    
    # Try to decompose into more parts (up to 3 available wires)
    # This is getting complex; skip for now
    
    return None


# Let's rewrite the greedy approach more carefully
def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
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
    
    # Convert to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Greedy circuit synthesis with sharing
    # available: mask -> wire_index
    # gates: list of [wire1, wire2]
    # next_wire: next available wire index
    
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    # target_wires: for each target, the wire that computes it
    target_wires = [None] * m
    
    # First pass: mark targets that are directly available
    for i in range(m):
        if target_masks[i] in available:
            target_wires[i] = available[target_masks[i]]
    
    # Second pass: for remaining targets, try to compute them
    # Process in order of increasing weight
    remaining_indices = sorted(
        [i for i in range(m) if target_wires[i] is None],
        key=lambda i: row_weights[i]
    )
    
    for idx in remaining_indices:
        if time.time() > deadline:
            break
        
        mask = target_masks[idx]
        
        # Try to find a decomposition
        wire = None
        
        # Try: XOR of two available wires
        found = False
        for w_mask, w_wire in list(available.items()):
            other_mask = mask ^ w_mask
            if other_mask in available:
                other_wire = available[other_mask]
                if w_mask == 0:
                    wire = other_wire
                    found = True
                    break
                elif other_mask == 0:
                    wire = w_wire
                    found = True
                    break
                else:
                    gates.append([w_wire, other_wire])
                    wire = next_wire
                    next_wire += 1
                    available[mask] = wire
                    found = True
                    break
        
        if not found:
            # Try: XOR of one input and one available wire
            for i in range(n):
                input_mask = 1 << i
                other_mask = mask ^ input_mask
                if other_mask in available:
                    other_wire = available[other_mask]
                    if other_mask == 0:
                        wire = i
                        found = True
                        break
                    else:
                        gates.append([i, other_wire])
                        wire = next_wire
                        next_wire += 1
                        available[mask] = wire
                        found = True
                        break
        
        if not found:
            # Fallback: compute from inputs directly
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                wire = zero_wire
            elif len(ones) == 1:
                wire = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                wire = current
                available[mask] = wire
        
        target_wires[idx] = wire
    
    return {'gates': gates, 'outputs': target_wires}
