# EVOLVE-BLOCK-START
import time

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

    # Perform Gaussian elimination to get RREF
    # We work with the row space spanned by row_masks
    # Step 1: Get a basis for the row space in RREF form
    # Step 2: For each original row, express it as a combination of RREF basis rows
    # Step 3: Synthesize each RREF basis row (which has a pivot and free variables)
    # Step 4: For each output, XOR together the needed RREF basis wires

    # Build RREF basis
    # We'll use the standard algorithm: for each column from left to right,
    # find a row with a 1 in that column, make it the pivot row, eliminate others.
    
    # Work with a copy of the masks
    rows = list(row_masks)
    
    # Gaussian elimination to RREF
    pivot_cols = []  # column index for each pivot row
    rref_rows = []   # the RREF rows (masks)
    
    for col in range(n):
        # Find a row with a 1 in this column
        pivot_idx = -1
        for i, r in enumerate(rows):
            if (r >> col) & 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
        
        # Swap to top of remaining rows
        rows[0], rows[pivot_idx] = rows[pivot_idx], rows[0]
        
        # Eliminate this column from all other rows
        for i in range(1, len(rows)):
            if (rows[i] >> col) & 1:
                rows[i] ^= rows[0]
        
        # This row is a pivot row
        pivot_cols.append(col)
        rref_rows.append(rows[0])
        
        # Remove from working set
        rows = rows[1:]
    
    # Now we have RREF basis. Each rref_row has a unique pivot column.
    # For each rref_row, the non-zero bits are: the pivot col + some free cols.
    # We need to synthesize each rref_row as an XOR of its constituent bits.
    
    # First, synthesize each RREF row
    rref_wires = []
    for rrow in rref_rows:
        if rrow == 0:
            rref_wires.append(n)  # constant zero
            continue
        
        # Decompose rrow into its set bits
        bits = [j for j in range(n) if (rrow >> j) & 1]
        if len(bits) == 1:
            rref_wires.append(memo[1 << bits[0]])
        elif len(bits) == 2:
            w = add_gate(memo[1 << bits[0]], memo[1 << bits[1]])
            rref_wires.append(w)
        else:
            # Build a balanced XOR tree for the bits
            # Use a recursive split approach
            def build_xor_tree(bit_list):
                if len(bit_list) == 1:
                    return memo[1 << bit_list[0]]
                if len(bit_list) == 2:
                    return add_gate(memo[1 << bit_list[0]], memo[1 << bit_list[1]])
                
                mid = len(bit_list) // 2
                left = build_xor_tree(bit_list[:mid])
                right = build_xor_tree(bit_list[mid:])
                return add_gate(left, right)
            
            rref_wires.append(build_xor_tree(bits))
    
    # Now express each original row as a combination of RREF rows
    # We need to solve: original_row = sum(c_i * rref_row_i) over GF(2)
    # where c_i is 0 or 1
    
    # Build a system: for each original row, find which RREF rows to XOR
    # We can do this by reversing the elimination process, or by direct solving
    
    # Direct approach: for each original row, greedily match with RREF rows
    # Since RREF has pivots, we can process columns in reverse pivot order
    
    # Actually, let's use a more systematic approach:
    # Store the transformation matrix during elimination
    
    # Let me redo this more carefully by tracking the linear combinations
    
    # Reset and redo with tracking
    rows_with_coeffs = [(mask, [0] * len(row_masks)) for mask in row_masks]
    for i in range(len(row_masks)):
        rows_with_coeffs[i][1][i] = 1
    
    pivot_cols2 = []
    rref_rows2 = []
    rref_coeffs = []  # for each RREF row, which original rows it's a combination of
    
    for col in range(n):
        pivot_idx = -1
        for i, (r, _) in enumerate(rows_with_coeffs):
            if (r >> col) & 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
        
        rows_with_coeffs[0], rows_with_coeffs[pivot_idx] = rows_with_coeffs[pivot_idx], rows_with_coeffs[0]
        
        for i in range(1, len(rows_with_coeffs)):
            if (rows_with_coeffs[i][0] >> col) & 1:
                rows_with_coeffs[i][0] ^= rows_with_coeffs[0][0]
                for j in range(len(row_masks)):
                    rows_with_coeffs[i][1][j] ^= rows_with_coeffs[0][1][j]
        
        pivot_cols2.append(col)
        rref_rows2.append(rows_with_coeffs[0][0])
        rref_coeffs.append(rows_with_coeffs[0][1])
        
        rows_with_coeffs = rows_with_coeffs[1:]
    
    # Now for each original row i, we need to find which RREF rows to XOR
    # The RREF rows form a basis. We can solve for the coefficients.
    
    # For each original row, express it in terms of RREF basis
    # Since RREF is in reduced form, we can do this column by column
    
    # Actually, we have rref_coeffs which tells us how each RREF row relates to original rows.
    # But we need the inverse: how each original row relates to RREF rows.
    
    # Let's build the transformation matrix T where:
    # rref_row_k = sum_j T[k][j] * original_row_j
    # We have T = rref_coeffs (each row is a coefficient vector)
    
    # We need to solve: original_row_i = sum_k S[i][k] * rref_row_k
    # So S = T^(-1) if T is square, but T may not be square.
    
    # Simpler: for each original row, greedily match with RREF rows
    # Process from last pivot to first
    
    outputs = []
    for orig_mask in row_masks:
        if orig_mask == 0:
            outputs.append(n)
            continue
        
        # Find which RREF rows to XOR
        # Start with a copy of orig_mask and eliminate using RREF rows in reverse pivot order
        remaining = orig_mask
        combo = []
        
        for k in range(len(rref_rows2) - 1, -1, -1):
            pivot_col = pivot_cols2[k]
            if (remaining >> pivot_col) & 1:
                remaining ^= rref_rows2[k]
                combo.append(k)
        
        # remaining should be 0 if orig_mask is in the span
        # XOR together the selected RREF wires
        if not combo:
            outputs.append(n)
        elif len(combo) == 1:
            outputs.append(rref_wires[combo[0]])
        elif len(combo) == 2:
            w = add_gate(rref_wires[combo[0]], rref_wires[combo[1]])
            outputs.append(w)
        else:
            # Build XOR tree for the combo
            def build_combo_xor(indices):
                if len(indices) == 1:
                    return rref_wires[indices[0]]
                if len(indices) == 2:
                    return add_gate(rref_wires[indices[0]], rref_wires[indices[1]])
                mid = len(indices) // 2
                left = build_combo_xor(indices[:mid])
                right = build_combo_xor(indices[mid:])
                return add_gate(left, right)
            
            outputs.append(build_combo_xor(combo))
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
