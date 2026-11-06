"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    import numpy as np
    
    # Parse input matrix
    try:
        A = np.array(matrix, dtype=int)
    except Exception:
        A = np.array(matrix)
        A = (A != 0).astype(int)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    m, n = A.shape
    
    # Work in GF(2)
    # We need to find a basis for the row space of A and the coefficients
    # expressing each original row in terms of this basis.
    
    # Copy A to avoid modification
    A_work = A.copy()
    
    # We will perform Gaussian Elimination to get RREF.
    # We also track the transformation matrix L such that A_rref = L @ A_orig.
    # Then A_orig = L_inv @ A_rref.
    # The rows of A_rref (non-zero) are the basis vectors.
    # The rows of L_inv (first r columns) give the coefficients for each original row.
    
    # Initialize L as Identity
    L = np.eye(m, dtype=int)
    
    pivot_cols = []
    row = 0
    for col in range(n):
        # Find a pivot in column 'col' from row 'row' downwards
        pivot_idx = -1
        for i in range(row, m):
            if A_work[i, col] == 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
        
        # Swap current row with pivot row
        if pivot_idx != row:
            A_work[[row, pivot_idx], :] = A_work[[pivot_idx, row], :]
            L[[row, pivot_idx], :] = L[[pivot_idx, row], :]
        
        pivot_cols.append(col)
        
        # Eliminate entries above and below the pivot
        for i in range(m):
            if i != row and A_work[i, col] == 1:
                A_work[i, :] = (A_work[i, :] + A_work[row, :]) % 2
                L[i, :] = (L[i, :] + L[row, :]) % 2
        
        row += 1
        if row == m:
            break
    
    r = row  # Rank
    
    # The first r rows of A_work are the basis vectors (in RREF form)
    # The first r rows of L are the transformation applied to get RREF.
    # We need L_inv such that A_orig = L_inv @ A_rref.
    # Since L @ A_orig = A_rref, we have A_orig = L^{-1} @ A_rref.
    # We can compute L_inv by inverting the r x r submatrix of L?
    # Actually, L is m x m. The transformation on the row space is captured by the first r rows of L?
    # Let's be precise.
    # Let A_rref be the matrix with r non-zero rows (the top r rows of A_work).
    # Let L_sub be the top r x m submatrix of L? No.
    # The equation is: [L_top; L_bottom] @ A_orig = [A_rref; 0].
    # So L_top @ A_orig = A_rref.
    # We want to express each row a_i of A_orig as a combo of rows of A_rref.
    # a_i = sum_k c_ik b_k.
    # This is equivalent to: A_orig = C @ A_rref, where C is m x r.
    # We know L_top @ A_orig = A_rref.
    # This doesn't directly give C.
    
    # Alternative: Track the coefficients directly during elimination.
    # Instead of just L, let's track the "history" of each original row.
    # Or simpler: Perform elimination on an augmented matrix [A | I_m].
    # The right side will become L.
    # Then we have L @ A = A_rref (top r rows).
    # We want C such that A = C @ A_rref (top r rows).
    # Note that the non-zero rows of A_rref form a basis.
    # The matrix C is the inverse of the mapping from A_rref rows to A rows.
    
    # Let's re-do the elimination tracking the "coordinates" of the current rows in the original basis?
    # No, let's use the property that if we perform elimination to get RREF,
    # the operations are linear.
    
    # Easier method:
    # 1. Find pivot columns.
    # 2. Extract the submatrix of A consisting of the pivot columns. Call it A_piv.
    # 3. A_piv is r x r and invertible over GF(2).
    # 4. The basis vectors are the rows of A_rref.
    # 5. Each original row a_i can be expressed as a linear combination of the basis vectors.
    #    Let B be the matrix of basis vectors (r x n).
    #    We want C (m x r) such that A = C @ B.
    #    Consider the pivot columns of A. Let A_piv be the r x r matrix formed by these columns.
    #    Let B_piv be the r x r matrix formed by these columns in B (which is A_rref).
    #    In RREF, B_piv is the Identity matrix I_r.
    #    So A_piv = C @ I_r = C.
    #    Therefore, the coefficients C are simply the values in the pivot columns of the original matrix A!
    #    Wait, is this true?
    #    If B is in RREF, its pivot columns form an Identity matrix.
    #    A = C @ B.
    #    Looking at the pivot columns: A_piv = C @ B_piv = C @ I_r = C.
    #    Yes! The coefficients for expressing the original rows in terms of the RREF basis
    #    are just the values in the pivot columns of the original matrix.
    
    # Let's verify this logic.
    # Basis B is RREF of A.
    # Any row a_i is in the row space of B.
    # a_i = sum_k c_ik b_k.
    # Let P be the set of pivot columns.
    # For any pivot column p in P, b_k[p] is 1 if k corresponds to the pivot row for column p, else 0.
    # Specifically, if we order the basis vectors by their pivot columns,
    # b_k has a 1 in column pivot_k and 0 in all other pivot columns.
    # So, looking at column pivot_k:
    # a_i[pivot_k] = sum_j c_ij b_j[pivot_k] = c_ik.
    # Thus, c_ik = a_i[pivot_k].
    
    # So, the coefficient matrix C is simply A[:, pivot_cols].
    
    # Steps:
    # 1. Perform Gaussian Elimination to find pivot_cols and the basis vectors (rows of A_rref).
    # 2. Extract basis vectors B (r x n) from A_work (top r rows).
    # 3. Extract coefficients C (m x r) from A (original) using pivot_cols.
    # 4. Synthesize wires for B.
    # 5. Synthesize outputs using C and the wires for B.
    
    # Re-run elimination to get A_rref and pivot_cols cleanly
    A_work = A.copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot_idx = -1
        for i in range(row, m):
            if A_work[i, col] == 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
        
        if pivot_idx != row:
            A_work[[row, pivot_idx], :] = A_work[[pivot_idx, row], :]
        
        pivot_cols.append(col)
        
        for i in range(m):
            if i != row and A_work[i, col] == 1:
                A_work[i, :] = (A_work[i, :] + A_work[row, :]) % 2
        
        row += 1
        if row == m:
            break
    
    r = row
    # Basis vectors are the top r rows of A_work
    B = A_work[:r, :]
    
    # Coefficients are the original A at the pivot columns
    # C[i, k] = A[i, pivot_cols[k]]
    C = A[:, pivot_cols]
    
    gates = []
    outputs = []
    next_wire = n + 1
    
    # 1. Synthesize basis wires w_0 ... w_{r-1}
    basis_wires = []
    for k in range(r):
        b_k = B[k]
        ones = np.where(b_k == 1)[0]
        
        if len(ones) == 0:
            # This shouldn't happen for a basis vector, but handle it
            basis_wires.append(n)
        elif len(ones) == 1:
            basis_wires.append(int(ones[0]))
        else:
            current_wire = int(ones[0])
            for j in range(1, len(ones)):
                gates.append([current_wire, int(ones[j])])
                current_wire = next_wire
                next_wire += 1
            basis_wires.append(current_wire)
    
    # 2. Synthesize outputs
    for i in range(m):
        c_i = C[i]
        # Find indices k where c_i[k] == 1
        active_basis = np.where(c_i == 1)[0]
        
        if len(active_basis) == 0:
            outputs.append(n)
        elif len(active_basis) == 1:
            outputs.append(basis_wires[int(active_basis[0])])
        else:
            current_wire = basis_wires[int(active_basis[0])]
            for idx in active_basis[1:]:
                gates.append([current_wire, basis_wires[int(idx)]])
                current_wire = next_wire
                next_wire += 1
            outputs.append(current_wire)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
