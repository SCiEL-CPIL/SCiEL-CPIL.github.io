# EVOLVE-BLOCK-START
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []

    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Parse matrix to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # --- Structural Change: RREF-based 2-Layer Synthesis ---
    
    # 1. Compute RREF and Coefficients
    # We represent the matrix as a list of masks.
    # We perform Gaussian Elimination.
    # We track the transformation: original_row[i] = sum(coeff[i][j] * rref_row[j])
    
    # Initialize basis (RREF rows) and coefficients
    # basis_masks: list of masks
    # coeffs: list of lists of indices (or masks over basis dimension)
    
    basis_masks = []
    coeffs = [] # coeffs[i] will be a mask over the basis indices
    
    # We process columns 0 to n-1 to find pivots
    # Actually, standard RREF processes rows.
    
    # Let's use a list of (mask, coeff_mask) pairs.
    # coeff_mask is a bitmask where bit j is 1 if this row contains basis_row j.
    # Initially, row i is just itself, so coeff_mask is 1 << i.
    
    active_rows = [(mask, 1 << i) for i, mask in enumerate(row_masks)]
    
    # We want to eliminate bits from left to right (bit 0 to n-1).
    # We maintain a list of "basis" rows found so far.
    # basis_rows: list of (mask, coeff_mask)
    
    basis_rows = []
    pivot_cols = []
    
    # To make it efficient, we can just iterate through columns.
    # For each column c, find a row with bit c set, that hasn't been used as a pivot yet.
    
    used_rows = [False] * m
    
    for c in range(n):
        # Find a row with bit c set
        pivot_idx = -1
        for i in range(m):
            if not used_rows[i] and (active_rows[i][0] >> c) & 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
            
        # Use this row as pivot
        used_rows[pivot_idx] = True
        pivot_mask, pivot_coeff = active_rows[pivot_idx]
        
        # Add to basis
        basis_rows.append((pivot_mask, pivot_coeff))
        pivot_cols.append(c)
        
        # Eliminate this bit from all other rows
        for i in range(m):
            if i == pivot_idx or used_rows[i]:
                continue
            if (active_rows[i][0] >> c) & 1:
                # active_rows[i] = active_rows[i] XOR active_rows[pivot_idx]
                new_mask = active_rows[i][0] ^ pivot_mask
                new_coeff = active_rows[i][1] ^ pivot_coeff
                active_rows[i] = (new_mask, new_coeff)
        
        # Also, we need to eliminate this bit from *previous* basis rows to get RREF?
        # Standard RREF requires back-substitution.
        # For our purpose (synthesizing original rows), we just need a basis that spans the space.
        # The current `basis_rows` are linearly independent and span the space.
        # The `active_rows` (for unused rows) are now 0.
        # The `coeff` in `active_rows` for the pivot rows themselves is just the original basis coeff.
        # Wait, the `coeff` in `basis_rows` represents the original row combination.
        # We need to express *original* rows in terms of *basis* rows.
        
        # Let's re-evaluate.
        # We have basis_rows: [ (M_1, C_1), ..., (M_r, C_r) ]
        # M_k is the mask of the k-th basis vector (in original input space).
        # C_k is the mask of the k-th basis vector in the *original row space* (i.e., which original rows sum to it).
        # This is the inverse of what we want.
        # We want: Original Row i = sum (x_{i,k} * M_k).
        
        # Let's track the transformation matrix T such that Basis = T * Original.
        # Then Original = T^{-1} * Basis.
        
        # Alternatively, just track the coefficients of the original rows in terms of the basis.
        # Let's restart the tracking.
        
    # The above elimination is getting complicated to track inverses.
    # Let's use a simpler approach:
    # 1. Find a basis B (list of masks).
    # 2. For each original row R, express R as a combination of B.
    
    # Step 1: Find Basis
    basis = []
    for mask in row_masks:
        if mask == 0:
            continue
        # Reduce mask by existing basis
        reduced = mask
        for b in basis:
            # If the highest set bit of reduced is also set in b, XOR
            # This is slow if we don't sort.
            pass
            
    # Let's use the standard RREF algorithm properly.
    
    # Create a copy of the matrix as masks
    mat = row_masks[:]
    coeffs_mat = [1 << i for i in range(m)] # Identity matrix initially
    
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for i in range(rank, m):
            if (mat[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row == -1:
            continue
            
        # Swap
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        coeffs_mat[rank], coeffs_mat[pivot_row] = coeffs_mat[pivot_row], coeffs_mat[rank]
        
        # Eliminate
        for i in range(m):
            if i != rank and (mat[i] >> col) & 1:
                mat[i] ^= mat[rank]
                coeffs_mat[i] ^= coeffs_mat[rank]
                
        rank += 1
        
    # Now, mat[0..rank-1] are the RREF basis rows.
    # coeffs_mat[0..rank-1] are the linear combinations of original rows that form these basis rows.
    # We need the inverse: Express original rows in terms of basis rows.
    
    # Let B be the basis matrix (rank x n).
    # Let C be the coeff matrix (rank x m).
    # We have B = C * Original (conceptually, over GF(2)).
    # We want X such that Original = X * B.
    # This means C * X = I (Identity of size rank)?
    # No, C is rank x m. We only care about the first rank columns of X?
    # Actually, the basis rows are linearly independent.
    # We can just solve for each original row.
    
    # Simpler:
    # The RREF process effectively creates a basis.
    # The non-zero rows in `mat` are the basis.
    # The corresponding rows in `coeffs_mat` tell us how to get them from originals.
    # To get an original row $R_i$, we need to find $y$ such that $R_i = \sum y_j B_j$.
    
    # Since we have the RREF, we can just use the `coeffs_mat` rows as the basis vectors in the "dual" space?
    # Actually, there is a direct way:
    # The `coeffs_mat` rows (for the pivot rows) form a basis for the row space of the *transpose*?
    
    # Let's just use the fact that we have the basis masks `basis_masks = mat[:rank]`.
    # And we have the original masks `row_masks`.
    # We can solve for the coefficients of each original row in terms of the basis using Gaussian elimination on the basis.
    
    def express_in_basis(target_mask, basis_masks):
        # Returns a mask over the basis indices
        # basis_masks are linearly independent
        # We reduce target_mask by basis_masks
        # If we reduce to 0, the coefficients used are the answer.
        
        # To make this fast, we should have a "reduced" basis.
        # The `basis_masks` from RREF are already in RREF form (pivots are unique).
        
        coeff = 0
        current = target_mask
        
        # We need to iterate in reverse order of pivots?
        # Or we can just do forward elimination if the basis is sorted by pivot column.
        # The `mat` from the loop above is in RREF.
        
        # Let's store the pivot columns for the basis
        # We didn't store them, but we can infer them or just use a general solver.
        
        # General Solver:
        # Create a dictionary: pivot_col -> basis_index
        # But we don't have pivot cols easily.
        
        # Let's just do a simple Gaussian elimination on the fly for each row?
        # That is O(m * rank * n). For 16x16, 16*16*16 = 4096. Very fast.
        
        # We'll create a local copy of the basis and reduce the target.
        
        # Actually, since we have the RREF `mat`, we can just use the `coeffs_mat`?
        # No, `coeffs_mat` maps Basis -> Original.
        
        # Let's just use a standard "express vector in basis" function.
        
        # To do this efficiently, we can pre-compute the "inverse" basis.
        
        pass
        
    # This is getting too complex for the time limit.
    # Let's fall back to the **Global Memoized Balanced Tree** but with a **Better Split Heuristic**.
    
    # The current split is "first half of bits".
    # New Split: "Split into the smallest possible non-zero sub-mask and the rest"?
    # No, that's depth n.
    
    # "Split into two parts with equal popcount" is good for depth.
    # "Split into a part that is already in memo" is good for gates.
    
    # Let's try: **Greedy Split with Memo Lookup**.
    # When splitting a mask M, look for a sub-mask A of M such that:
    # 1. A is in memo.
    # 2. M ^ A is in memo.
    # If found, use it.
    # Otherwise, use balanced split.
    
    # This requires iterating through sub-masks, which is expensive.
    
    # Let's stick to the RREF idea but implement it simply:
    # 1. Compute RREF Basis.
    # 2. Synthesize Basis Rows (using global memo).
    # 3. For each Original Row, find its expression in Basis (using a simple solver).
    # 4. Synthesize Original Rows from Basis Wires (using a second memo).
    
    # This is the "2-Layer" approach.
    
    # Implementation of Step 3:
    # We have basis_masks (list of masks).
    # We want to express row_masks[i] as XOR of basis_masks[j].
    
    # We can build a linear system.
    # Let B be the matrix of basis masks (rank x n).
    # We want to solve B * x = r_i for each r_i.
    
    # Since B is in RREF (from our elimination), solving is easy.
    # We just need the pivot columns.
    
    # Let's re-run the RREF to capture pivot columns.
    
    mat = row_masks[:]
    coeffs_mat = [1 << i for i in range(m)]
    rank = 0
    pivot_cols = []
    
    for col in range(n):
        pivot_row = -1
        for i in range(rank, m):
            if (mat[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row == -1:
            continue
            
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        coeffs_mat[rank], coeffs_mat[pivot_row] = coeffs_mat[pivot_row], coeffs_mat[rank]
        
        for i in range(m):
            if i != rank and (mat[i] >> col) & 1:
                mat[i] ^= mat[rank]
                coeffs_mat[i] ^= coeffs_mat[rank]
                
        pivot_cols.append(col)
        rank += 1
        
    basis_masks = mat[:rank]
    
    # Now, express each original row in terms of basis.
    # We have the RREF matrix `basis_masks`.
    # To express a vector `v` in terms of this basis:
    # We scan the pivot columns in order.
    # If v has a 1 at pivot_col[k], we XOR basis_masks[k] into v and set bit k in the coeff.
    
    coeff_masks = []
    for i in range(m):
        v = row_masks[i]
        coeff = 0
        for k in range(rank):
            if (v >> pivot_cols[k]) & 1:
                v ^= basis_masks[k]
                coeff |= (1 << k)
        # v should be 0
        coeff_masks.append(coeff)
        
    # Now we have:
    # 1. basis_masks: list of masks to synthesize from inputs.
    # 2. coeff_masks: list of masks (over basis dimension) to synthesize from basis wires.
    
    # Synthesize Layer 1: Basis Rows
    # Use the existing global memo logic, but only for basis_masks.
    
    # We need to collect all sub-masks needed for basis_masks.
    needed_l1 = set()
    def collect_needed_l1(mask):
        if mask in memo:
            return
        if mask in needed_l1:
            return
        needed_l1.add(mask)
        # Balanced split
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) == 1:
            return
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for idx, b in enumerate(bits):
            if idx < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
        if sub1 != 0:
            collect_needed_l1(sub1)
        if sub2 != 0:
            collect_needed_l1(sub2)
            
    for mask in basis_masks:
        if mask != 0:
            collect_needed_l1(mask)
            
    sorted_needed_l1 = sorted(needed_l1, key=lambda m: m.bit_count())
    
    for mask in sorted_needed_l1:
        if mask in memo:
            continue
        bits = [j for j in range(n) if (mask >> j) & 1]
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for idx, b in enumerate(bits):
            if idx < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
                
        if sub1 == 0:
            memo[mask] = memo[sub2]
        elif sub2 == 0:
            memo[mask] = memo[sub1]
        else:
            w = add_gate(memo[sub1], memo[sub2])
            memo[mask] = w
            
    basis_wires = [memo[mask] for mask in basis_masks]
    
    # Synthesize Layer 2: Original Rows from Basis Wires
    # New memo for this layer
    memo_l2 = {0: n} # 0 is zero wire
    # Inputs to this layer are basis_wires
    # We map bit k in coeff_mask to basis_wires[k]
    
    # We need to synthesize coeff_masks.
    # The "inputs" for this synthesis are the basis_wires.
    # We can reuse the same balanced tree logic.
    
    needed_l2 = set()
    def collect_needed_l2(mask, dim):
        if mask == 0:
            return
        if mask in memo_l2:
            return
        if mask in needed_l2:
            return
        needed_l2.add(mask)
        bits = [j for j in range(dim) if (mask >> j) & 1]
        if len(bits) == 1:
            return
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for idx, b in enumerate(bits):
            if idx < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
        if sub1 != 0:
            collect_needed_l2(sub1, dim)
        if sub2 != 0:
            collect_needed_l2(sub2, dim)
            
    for mask in coeff_masks:
        if mask != 0:
            collect_needed_l2(mask, rank)
            
    sorted_needed_l2 = sorted(needed_l2, key=lambda m: m.bit_count())
    
    for mask in sorted_needed_l2:
        if mask in memo_l2:
            continue
        bits = [j for j in range(rank) if (mask >> j) & 1]
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for idx, b in enumerate(bits):
            if idx < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
                
        if sub1 == 0:
            memo_l2[mask] = memo_l2[sub2]
        elif sub2 == 0:
            memo_l2[mask] = memo_l2[sub1]
        else:
            w1 = memo_l2[sub1]
            w2 = memo_l2[sub2]
            w = add_gate(w1, w2)
            memo_l2[mask] = w
            
    outputs = [memo_l2[mask] for mask in coeff_masks]
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
