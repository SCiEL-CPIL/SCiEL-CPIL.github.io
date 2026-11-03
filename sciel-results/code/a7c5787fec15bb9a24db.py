"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            # Try to reduce rank by finding a dependency
            if time.time() < deadline - 5:
                U83, V83, W83 = _try_rank_83_via_dependency(U, V, W, n, m, p, deadline)
                if U83 is not None:
                    if _verify_fast(U83, V83, W83, n, m, p):
                        U = U83
                        V = V83
                        W = W83
                        rank = U.shape[1]
            
            # Convert to integer lists for clean output
            U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
            V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
            W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+cg, af+dh],[ce+dg, cf+gh]]
    W index: W[k*2+i, c] where C[i,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(f-h)
    # m3 = d(g-e)
    # m4 = (a+b)h
    # m5 = (c-d)(g+h)
    # m6 = (a-b)(e+f)
    # C[0,0] = m0 + m3 - m4 + m6
    # C[0,1] = m2 + m4
    # C[1,0] = m1 + m3
    # C[1,1] = m0 - m1 + m2 + m5
    
    # U: A factors (U[i*2+j, c])
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[3, 5] = -1  # -d in m5
    U[0, 6] = 1   # a in m6
    U[1, 6] = -1  # -b in m6
    
    # V: B factors (V[j*2+k, c])
    V[0, 0] = 1   # e in m0
    V[1, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[1, 2] = -1  # f in m2 (already set)
    V[1, 2] = 1   # f in m2
    V[0, 2] = -1  # -h in m2
    V[1, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[1, 4] = 1   # h in m4
    V[1, 5] = 1   # g in m5
    V[0, 5] = 1   # h in m5 (wait, m5 = (c-d)(g+h))
    V[0, 5] = 1   # g in m5
    V[1, 5] = 1   # h in m5
    V[0, 6] = 1   # e in m6
    V[1, 6] = 1   # f in m6
    
    # W: C factors (W[k*2+i, c])
    # C[0,0] = m0 + m3 - m4 + m6
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    # C[0,1] = m2 + m4
    W[1, 2] = 1   # m2 in C[0,1]
    W[1, 4] = 1   # m4 in C[0,1]
    # C[1,0] = m1 + m3
    W[0, 1] = 1   # m1 in C[1,0]
    W[0, 3] = 1   # m3 in C[1,0] (already set)
    # C[1,1] = m0 - m1 + m2 + m5
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Total rank: 3 k-blocks * (2 Strassen blocks + 1 schoolbook for j=4)
    # For each k-block:
    #   j in {0,1}: Strassen (7)
    #   j in {2,3}: Strassen (7) 
    #   j = 4: schoolbook 2x1x2 (4)
    # Per k-block: 7+7+4 = 18
    # For bottom row (i=2):
    #   j in {0,1}: schoolbook 1x2x2 (4)
    #   j in {2,3}: schoolbook 1x2x2 (4)
    #   j = 4: schoolbook 1x1x2 (2)
    # Per k-block for bottom: 4+4+2 = 10
    # Total per k-block: 18+10 = 28
    # 3 k-blocks: 84
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            # U: A[i,j] for i in {0,1}, j in {0,1}
            U[0*5 + 0, col] = sU[0, c]   # i=0, j=0
            U[0*5 + 1, col] = sU[1, c]   # i=0, j=1
            U[1*5 + 0, col] = sU[2, c]   # i=1, j=0
            U[1*5 + 1, col] = sU[3, c]   # i=1, j=1
            # V: B[j,k] for j in {0,1}, k in {k0,k1}
            V[0*6 + k0, col] = sV[0, c]   # j=0, k=k0
            V[0*6 + k1, col] = sV[1, c]   # j=0, k=k1
            V[1*6 + k0, col] = sV[2, c]   # j=1, k=k0
            V[1*6 + k1, col] = sV[3, c]   # j=1, k=k1
            # W: C[k,i] for k in {k0,k1}, i in {0,1}
            W[k0*3 + 0, col] = sW[0, c]   # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]   # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]   # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]   # k=k1, i=1
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        # Truncate or pad
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _try_rank_83_via_dependency(U84, V84, W84, n, m, p, deadline):
    """
    Try to find a linear dependency among the 84 Strassen-partitioned terms
    to reduce rank to 83.
    """
    r84 = U84.shape[1]
    
    # Build the tensor as a 3D array: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We want to check if one column can be expressed as a linear combination of others.
    # For each column c, check if removing it still gives the correct tensor.
    
    # More efficiently: build the "tensor matricization" and check rank.
    # The tensor T has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # Each rank-1 term contributes U[i,j,c]*V[j,k,c]*W[k,i,c]
    
    # Try removing each column and see if we can adjust others to compensate.
    # This is expensive; instead, check if the 84 vectors are linearly dependent
    # in the appropriate tensor space.
    
    # Build a matrix where each column c is the vectorized contribution:
    # vec_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this is the full tensor entry, which should equal 1 for all (i,j,k).
    # Actually, the sum over c of these should equal 1 for each (i,j,k).
    
    # Alternative: check if there's a column c* such that its contribution
    # can be absorbed by modifying other columns.
    
    # For simplicity, try a few candidate columns to remove and solve for adjustments.
    # This is a heuristic approach.
    
    n_entries = n * m * p  # 90 entries
    
    # Build matrix M where M[idx, c] = U[i,j,c]*V[j,k,c]*W[k,i,c] for entry (i,j,k)
    M = np.zeros((n_entries, r84))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r84):
                    M[idx, c] = U84[i*m+j, c] * V84[j*p+k, c] * W84[k*n+i, c]
                idx += 1
    
    # M @ ones should give all 1s (the tensor entries)
    # Check: sum_c M[idx,c] should be 1 for all idx
    
    # Now, for each candidate column c*, try to see if we can find coefficients
    # for other columns that, when added to column c*, give zero net contribution.
    # i.e., is column c* in the span of the other columns?
    
    # This is equivalent to checking if the 84 columns of M are linearly dependent.
    # If rank(M) < 84, then there's a dependency.
    
    # Compute rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except Exception:
        return None
    
    if rank_M >= r84:
        return None
    
    # Find a null vector of M^T (i.e., M^T @ v = 0 means sum_c v[c] * M[:,c] = 0)
    # This gives coefficients v[c] such that sum_c v[c] * (contribution of column c) = 0
    # Then we can remove one column and adjust others.
    
    # Get null space of M^T
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=True)
    except Exception:
        return None
    
    # Null space vectors are rows of Vt_svd corresponding to zero singular values
    null_mask = s_vals < 1e-10
    if not np.any(null_mask):
        return None
    
    # Take the first null vector
    null_vec = Vt_svd[null_mask[0], :]
    
    # Find the column with largest absolute coefficient to remove
    c_star = np.argmax(np.abs(null_vec))
    
    # The dependency is: sum_c null_vec[c] * col_c = 0
    # So col_{c_star} = -sum_{c != c_star} (null_vec[c] / null_vec[c_star]) * col_c
    
    # To remove column c_star, we need to adjust the other columns.
    # New column c (for c != c_star) should become:
    # col_c_new = col_c - (null_vec[c] / null_vec[c_star]) * col_{c_star}
    # But we're removing col_{c_star}, so we just need to verify that
    # the remaining columns with adjusted coefficients still work.
    
    # Actually, the simplest approach: just drop column c_star and see if
    # the remaining 83 columns still sum to the correct tensor.
    # This only works if null_vec[c_star] = 0, which it's not.
    
    # Better: we need to find a different representation.
    # For now, this is a heuristic that may not work.
    
    # Alternative: try to find if any single column has zero contribution
    # (i.e., all entries in that column of M are zero)
    col_sums = np.sum(np.abs(M), axis=0)
    zero_cols = np.where(col_sums < 1e-15)[0]
    
    if len(zero_cols) > 0:
        # Remove a zero column
        c_remove = zero_cols[0]
        mask = np.ones(r84, dtype=bool)
        mask[c_remove] = False
        return U84[:, mask], V84[:, mask], W84[:, mask]
    
    return None


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
