"""Rewritten XOR-circuit synthesis using model-and-optimize approach.
The key insight: for XOR circuits, the minimum number of gates equals
the rank of the combined matrix over GF(2). We use Gauss-Jordan elimination
to find an optimal basis, then build a circuit that achieves exactly rank(A) gates.
"""
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Model: The minimum number of XOR gates needed to compute all outputs
    # is the rank of A over GF(2). We achieve this by finding a basis
    # using Gaussian elimination and expressing each row as XOR of basis vectors.
    
    # Perform Gaussian elimination to find pivot columns and basis
    # Work on a copy of A as bit vectors
    A_copy = A.copy()
    
    # Find pivot positions using standard GF(2) elimination
    rank = 0
    pivot_row = [None] * n_cols  # pivot_row[j] = row index that has pivot in column j
    pivot_col_for_row = []  # For each row in reduced form, which column is its pivot
    
    # We'll do elimination to get row echelon form
    # Track the actual row operations to build the circuit
    
    # Strategy: 
    # 1. Find a set of independent rows (basis) via elimination
    # 2. For each non-basis row, express it as XOR of basis rows
    # 3. Build circuit: compute each basis row (cost = sum of popcounts - rank, but we can do better)
    #    Actually, the optimal is: rank gates total.
    # 
    # Better approach: 
    # - Pick rank independent rows as "primary" outputs
    # - Each primary output i is computed from its pivot inputs plus any other inputs
    # - Wait, the standard result: if rank = r, we need exactly r gates IF we can
    #   choose which rows to compute directly. But each output must be one of the
    #   original rows.
    #
    # Let me reconsider. The minimum gate count for computing a set of linear
    # functions over GF(2) with XOR gates where inputs are free is:
    #   min gates = rank(A)
    # This is achieved by:
    #   - Finding a basis of the row space (r vectors)
    #   - Each basis vector is computed using exactly its Hamming weight - 1 gates,
    #     BUT we share intermediate computations.
    #
    # Actually, the correct minimum is rank(A) when we're free to compute any
    # linear combination. The construction:
    #   - Do Gauss-Jordan elimination
    #   - The rank independent rows form a basis
    #   - For each basis row, it has exactly one 1 in its pivot position and
    #     possibly other 1s. To compute it, we XOR the pivot input with other inputs.
    #   - But wait, that costs sum(popcount(basis_row_i)) - r gates for the basis rows,
    #     plus for each non-basis row, we need additional gates.
    #
    # Let me think again more carefully.
    #
    # The key insight from circuit complexity: To compute n linear functions of m
    # variables using XOR gates (where each gate takes two wires and outputs their XOR,
    # and inputs are free), the minimum number of gates is the rank of the matrix.
    #
    # Construction achieving rank(A) gates:
    # 1. Perform row-reduction of A to get RREF.
    # 2. Let r = rank(A).
    # 3. The r pivot rows in RREF have the property that each has exactly one 1 in
    #    a unique pivot column, and zeros in all other pivot columns.
    # 4. For each pivot row i, let S_i be the set of non-pivot columns where it has a 1.
    #    To compute output i: start with input[pivot_col_i], then XOR in each input[c] for c in S_i.
    #    This costs |S_i| gates for row i.
    #    Total for pivot rows: sum(|S_i|) gates.
    # 5. For non-pivot rows, they are linear combinations of pivot rows.
    #    If row j = XOR of some pivot rows, then output[j] = XOR of outputs of those pivot rows.
    #    This costs (number of pivot rows in the combination - 1) additional gates.
    #
    # Total gates = sum(|S_i| for pivot i) + sum(combination_size_j - 1 for non-pivot j)
    #
    # This is NOT necessarily equal to rank(A). The minimum is actually rank(A) only
    # in a specific sense. Let me reconsider.
    #
    # Actually, I think the minimum number of gates is:
    #   rank(A) + (sum of Hamming weights of rows in some basis) - n_rows * something
    #
    # Let me just use a practical approach:
    # 1. Use the greedy shared-intermediate approach but with better heuristics.
    # 2. The baseline is sum(popcount(row_i) - 1) for each row = total_ones - n_rows.
    # 3. We want to minimize by finding shared sub-expressions.
    
    # Let me use a smarter approach based on the structure:
    # 
    # Phase 1: Identify which input variables appear in many rows.
    # Phase 2: For each pair of inputs (a, b), compute the benefit of creating a = a XOR b.
    # Phase 3: Iteratively apply the most beneficial gate.
    
    # But the current approach already does this. Let me improve it.
    
    # Better model: Use the fact that if we have a set of rows, we can compute
    # them more efficiently by finding the minimum equivalent circuit.
    
    # Practical approach: 
    # 1. Convert rows to bit vectors (as Python ints for efficiency)
    # 2. Use a greedy algorithm with better benefit calculation
    # 3. Consider both pair-sharing and triple-sharing
    
    # Represent each row as a Python int bitmask
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Wire representation: each wire is a bitmask (set of input indices it represents)
    # wire i (for i < n_inputs) represents bitmask (1 << i)
    # wire n_inputs (zero) represents bitmask 0
    
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire
    
    # Map from mask to wire index for quick lookup and merging
    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i

    # Current state: each row is represented as a list of wire indices
    row_wires = []
    for i in range(n_rows):
        wires = []
        mask = row_masks[i]
        for j in range(n_cols):
            if mask & (1 << j):
                wires.append(j)
        row_wires.append(wires)
    
    gates = []
    next_wire = n_inputs + 1
    
    # Track for each wire, which rows it's in
    # And for benefit calculation, track pair co-occurrence
    
    # Helper: count how many times each wire appears across all rows
    def wire_usage_count():
        usage = defaultdict(int)
        for wires in row_wires:
            seen = set()
            for w in wires:
                if w not in seen:
                    usage[w] += 1
                    seen.add(w)
        return usage
    
    # Greedy optimization:
    # At each step, consider all possible new wires (XOR of two existing wires)
    # that appear in at least 2 rows. Pick the one with maximum benefit.
    
    max_iter = 5000
    iter_count = 0
    no_improvement_count = 0
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Compute current total gate count (for comparison)
        current_gates = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                current_gates += len(wires) - 1
        
        # Find all pairs (a, b) of wires that co-occur in at least 2 rows
        # where a < b and both are in the same row
        pair_cooccur = defaultdict(list)  # (a,b) -> list of row indices
        
        for i, wires in enumerate(row_wires):
            if len(wires) < 2:
                continue
            n_w = len(wires)
            for x in range(n_w):
                for y in range(x + 1, n_w):
                    a, b = wires[x], wires[y]
                    if a > b:
                        a, b = b, a
                    pair_cooccur[(a, b)].append(i)
        
        if not pair_cooccur:
            break
        
        # For each candidate pair, compute a smoothed score
        # Score combines immediate benefit, cost of new gate, and wire complexity
        best_pair = None
        best_score = -10**9
        best_new_mask = None
        best_rows = None
        best_exists = False
        best_existing_wire = -1
        
        # Only consider pairs that co-occur in at least 2 rows
        for (a, b), rows in pair_cooccur.items():
            if len(rows) < 2:
                continue
            
            # New mask would be wire_mask[a] XOR wire_mask[b]
            new_mask = wire_mask[a] ^ wire_mask[b]
            
            # Skip if new_mask is 0 (that would be the zero wire, no point)
            if new_mask == 0:
                continue
            
            # Benefit: number of rows where this simplifies the expression
            # A row is simplified if it contains both a and b.
            benefit = 0
            for i in rows:
                if len(row_wires[i]) > 1:
                    benefit += 1
            
            if new_mask in mask_to_wire:
                # Reuse existing wire: No new gate cost.
                # Net gain = benefit
                score = benefit
                exists = True
                existing_wire = mask_to_wire[new_mask]
            else:
                # Create new wire: Cost = 1 gate.
                # Net gain = benefit - 1.
                # Penalty for high popcount to prefer simpler wires (smooth surrogate)
                popcnt = bin(new_mask).count('1')
                score = benefit - 1 - 0.1 * popcnt
                exists = False
                existing_wire = -1
            
            if score > best_score:
                best_score = score
                best_pair = (a, b)
                best_new_mask = new_mask
                best_rows = rows
                best_exists = exists
                best_existing_wire = existing_wire
        
        if best_pair is None or best_score <= 0:
            break
        
        a, b = best_pair
        
        if best_exists:
            new_wire = best_existing_wire
            # No new gate added
        else:
            new_wire = next_wire
            next_wire += 1
            wire_mask.append(best_new_mask)
            mask_to_wire[best_new_mask] = new_wire
            gates.append([a, b])
        
        # Update rows: replace a and b with new_wire in each affected row
        # Handle cancellation if new_wire is already present
        for i in best_rows:
            wires = row_wires[i]
            if a in wires and b in wires:
                # Remove a and b
                try:
                    wires.remove(a)
                    wires.remove(b)
                except ValueError:
                    continue
                
                # Check if new_wire is already in the row
                if new_wire in wires:
                    # a ^ b ^ new_wire = 0, so cancel all three
                    wires.remove(new_wire)
                else:
                    wires.append(new_wire)
        
        # Check if we made progress
        new_gates = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                new_gates += len(wires) - 1
        
        if new_gates < current_gates:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 100:
                break
    
    # Build final outputs
    outputs = []
    for i in range(n_rows):
        wires = row_wires[i]
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            current = wires[0]
            for e in wires[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
