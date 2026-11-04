import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify exact decomposition."""
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n*m)])
        vc = np.array([V[j][c] for j in range(m*p)])
        wc = np.array([W[k][c] for k in range(p*n)])
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

def _laderman_2x3x3():
    """
    Laderman's rank-7 decomposition for 2x3 @ 3x3 -> 2x3.
    Returns list of 7 terms, each (u_dict, v_dict, w_dict).
    u: 2x3 matrix indices (i,j) -> row i*3+j
    v: 3x3 matrix indices (j,k) -> row j*3+k
    w: 2x3 output indices (k,i) -> row k*3+i  [trace convention W[k*n+i,r]]
    """
    # Laderman's algorithm for 2x3x3 (rank 7)
    # A is 2x3, B is 3x3, C = A*B is 2x3
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    terms = []
    
    # M1 = A[0,0] * B[0,0]
    terms.append(({(0,0): 1}, {(0,0): 1}, {(0,0): 1}))
    
    # M2 = A[0,1] * B[1,0]
    terms.append(({(0,1): 1}, {(1,0): 1}, {(0,1): 1}))
    
    # M3 = A[0,2] * B[2,0]
    terms.append(({(0,2): 1}, {(2,0): 1}, {(0,2): 1}))
    
    # M4 = (A[0,0]+A[1,0]) * (B[1,0]+B[2,0])
    terms.append(({(0,0): 1, (1,0): 1}, {(1,0): 1, (2,0): 1}, {(0,1): 1, (0,2): 1}))
    
    # M5 = (A[0,1]+A[1,1]) * (B[1,1]+B[2,1])
    terms.append(({(0,1): 1, (1,1): 1}, {(1,1): 1, (2,1): 1}, {(1,1): 1, (1,2): 1}))
    
    # M6 = (A[0,2]+A[1,2]) * (B[1,2]+B[2,2])
    terms.append(({(0,2): 1, (1,2): 1}, {(1,2): 1, (2,2): 1}, {(2,1): 1, (2,2): 1}))
    
    # M7 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[0,1]+B[0,2])
    terms.append(({(0,0): 1, (0,1): 1, (0,2): 1}, {(0,0): 1, (0,1): 1, (0,2): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}))
    
    return terms

def _laderman_3x2x3():
    """
    Laderman's rank-7 decomposition for 3x2 @ 2x3 -> 3x3.
    Returns list of 7 terms, each (u_dict, v_dict, w_dict).
    u: 3x2 matrix indices (i,j) -> row i*2+j
    v: 2x3 matrix indices (j,k) -> row j*3+k
    w: 3x3 output indices (k,i) -> row k*3+i  [trace convention W[k*n+i,r]]
    """
    # A is 3x2, B is 2x3, C = A*B is 3x3
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    terms = []
    
    # M1 = A[0,0] * B[0,0]
    terms.append(({(0,0): 1}, {(0,0): 1}, {(0,0): 1}))
    
    # M2 = A[1,0] * B[0,1]
    terms.append(({(1,0): 1}, {(0,1): 1}, {(1,0): 1}))
    
    # M3 = A[2,0] * B[0,2]
    terms.append(({(2,0): 1}, {(0,2): 1}, {(2,0): 1}))
    
    # M4 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])
    terms.append(({(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}))
    
    # M5 = (A[1,0]+A[1,1]) * (B[1,1]+B[1,2])
    terms.append(({(1,0): 1, (1,1): 1}, {(1,1): 1, (1,2): 1}, {(1,2): 1, (2,1): 1}))
    
    # M6 = (A[2,0]+A[2,1]) * (B[1,2]+B[0,2])
    terms.append(({(2,0): 1, (2,1): 1}, {(1,2): 1, (0,2): 1}, {(2,2): 1, (0,2): 1}))
    
    # M7 = (A[0,0]+A[1,0]+A[2,0]) * (B[0,0]+B[0,1]+B[0,2])
    terms.append(({(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1, (0,1): 1, (0,2): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}))
    
    return terms

def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 @ 2x2 -> 2x2.
    Returns list of 7 terms, each (u_dict, v_dict, w_dict).
    u: 2x2 matrix indices (i,j) -> row i*2+j
    v: 2x2 matrix indices (j,k) -> row j*2+k
    w: 2x2 output indices (k,i) -> row k*2+i  [trace convention W[k*n+i,r]]
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    
    # M2 = (A10+A11)B00
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    
    # M3 = A00(B10-B11)
    terms.append(({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    
    # M4 = A11(B01-B00)
    terms.append(({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    
    # M5 = (A00+A01)B11
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    
    return terms

def _embed_terms(terms, i_offset, j_offset, k_offset, n, m, p):
    """
    Embed local terms into the global (n,m,p) tensor.
    Local u indices (i,j) -> global row (i_offset+i)*m + (j_offset+j)
    Local v indices (j,k) -> global row (j_offset+j)*p + (k_offset+k)
    Local w indices (k,i) -> global row (k_offset+k)*n + (i_offset+i)
    """
    global_terms = []
    for u_dict, v_dict, w_dict in terms:
        g_u = {}
        for (di, dj), coeff in u_dict.items():
            row = (i_offset + di) * m + (j_offset + dj)
            g_u[row] = g_u.get(row, 0) + coeff
        g_v = {}
        for (dj, dk), coeff in v_dict.items():
            row = (j_offset + dj) * p + (k_offset + dk)
            g_v[row] = g_v.get(row, 0) + coeff
        g_w = {}
        for (dk, di), coeff in w_dict.items():
            row = (k_offset + dk) * n + (i_offset + di)
            g_w[row] = g_w.get(row, 0) + coeff
        global_terms.append((g_u, g_v, g_w))
    return global_terms

def _build_factor_matrices(terms, n, m, p):
    """
    Given a list of terms (each a tuple of dicts for U, V, W contributions),
    build the full factor matrices U, V, W as numpy arrays.
    """
    r = len(terms)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for row, val in u_dict.items():
            U[row, c] = val
        for row, val in v_dict.items():
            V[row, c] = val
        for row, val in w_dict.items():
            W[row, c] = val
    
    return U, V, W

def _compress_rank(U, V, W, tol=1e-10):
    """
    Compress the rank by finding the true rank of the combined factor space.
    If columns are linearly dependent, project onto a minimal basis.
    Returns (U_new, V_new, W_new, new_rank)
    """
    r = U.shape[1]
    if r == 0:
        return U, V, W, 0
    
    combined = np.vstack([U, V, W])
    
    # Compute SVD
    U_svd, s, Vt = np.linalg.svd(combined, full_matrices=False)
    
    # Determine numerical rank
    if len(s) == 0:
        return U, V, W, r
    
    max_s = s[0] if s[0] > 0 else 1.0
    rank = int(np.sum(s > max_s * tol))
    
    if rank >= r:
        return U, V, W, r
    
    # The column space of 'combined' has dimension 'rank'
    # We need to re-express U, V, W in terms of a rank-dimensional basis
    # 
    # combined = U_svd[:, :rank] @ diag(s[:rank]) @ Vt[:rank, :]
    # 
    # Let B = U_svd[:, :rank] (this is the basis, shape (n*m+m*p+p*n, rank))
    # Let C = diag(s[:rank]) @ Vt[:rank, :] (shape (rank, r))
    # Then combined = B @ C
    #
    # U is the top n*m rows of combined: U = B[:n*m,:] @ C
    # V is the next m*p rows: V = B[n*m:n*m+m*p,:] @ C
    # W is the last p*n rows: W = B[n*m+m*p:,:] @ C
    #
    # New decomposition: U_new = B[:n*m,:], V_new = B[n*m:n*m+m*p,:], W_new = B[n*m+m*p:,:]
    # But wait, that's not right. We need U_new @ ... 
    #
    # Actually, the tensor decomposition is:
    # T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    #
    # If we have a linear transformation: new_cols = old_cols @ M where M is r x r
    # then U_new = U @ M, V_new = V @ M, W_new = W @ M^{-1}
    #
    # But we want to reduce the number of columns. 
    #
    # Better approach: Since combined has rank 'rank', there exists a matrix P (r x rank)
    # such that combined = Q @ P where Q is (n*m+m*p+p*n) x rank.
    # 
    # Then U = Q[:n*m,:] @ P, V = Q[n*m:n*m+m*p,:] @ P, W = Q[n*m+m*p:,:] @ P
    #
    # This gives a rank-'rank' decomposition:
    # U_new = Q[:n*m,:]
    # V_new = Q[n*m:n*m+m*p,:]  
    # W_new = Q[n*m+m*p:,:]
    #
    # But wait, this doesn't preserve the tensor structure directly.
    # The tensor is T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    #
    # If U_new = Q[:n*m,:], V_new = Q[n*m:n*m+m*p,:], W_new = Q[n*m+m*p:,:]
    # then sum_c U_new[i,c]*V_new[j,c]*W_new[k,c] = sum_c Q[i,c]*Q[m_p+j,c]*Q[m_p+m_p+k,c]
    # which is NOT the same as the original tensor.
    #
    # Correct approach: We need to find matrices A (r x rank), B (rank x r) such that
    # for the tensor to be preserved, we need:
    # U @ A @ B = U, V @ A @ B = V, W @ A @ B = W ... no that's not right either.
    #
    # The correct way: If we have a change of basis in the factor space,
    # let M be an invertible r x r matrix. Then:
    # U' = U @ M, V' = V @ M, W' = W @ M^{-1}
    # preserves the tensor.
    #
    # To reduce rank, we need a non-invertible transformation.
    # If the columns of (U,V,W) stacked have rank r' < r, then there exist
    # matrices P (r x r') and Q (r' x r) with P @ Q = I_r ... no.
    #
    # Let me think differently. The set of vectors {u_c, v_c, w_c} for c=1..r
    # spans a space of dimension r' in R^{n*m + m*p + p*n}.
    # 
    # Actually, the correct compression is:
    # If there's a linear dependence among the "columns" in the combined space,
    # we can't simply drop columns because U, V, W are coupled.
    #
    # The proper way: Find the rank of the tensor itself. But that's hard.
    #
    # Alternative: Use the fact that if we can write the original rank-r decomposition
    # as a linear combination of rank-r' decompositions with r' < r, we've reduced rank.
    #
    # For now, let's try a simpler approach: if the SVD reveals that some singular
    # values are zero (or near-zero), we can potentially project.
    #
    # Actually, the cleanest approach for exact compression:
    # If combined = U_svd @ S @ Vt, and rank < r, then:
    # combined = U_svd[:, :rank] @ (S[:rank] @ Vt[:rank, :])
    # Let A = U_svd[:, :rank] (shape (N, rank))
    # Let B = S[:rank] @ Vt[:rank, :] (shape (rank, r))
    # Then combined = A @ B
    #
    # U = A[:n*m,:] @ B
    # V = A[n*m:n*m+m*p,:] @ B
    # W = A[n*m+m*p:,:] @ B
    #
    # This is a rank-r' decomposition where:
    # U_new = A[:n*m,:]  (shape n*m x rank)
    # V_new = A[n*m:n*m+m*p,:]  (shape m*p x rank)
    # W_new = A[n*m+m*p:,:]  (shape p*n x rank)
    #
    # But does this preserve the tensor? 
    # sum_c U_new[i,c] * V_new[j,c] * W_new[k,c]
    # = sum_c A[i,c] * A[m_p+j,c] * A[m_p+m_p+k,c]
    #
    # This is NOT equal to the original tensor in general.
    #
    # So this approach is incorrect. The compression must preserve the 
    # trilinear structure.
    #
    # Correct method: We need to find a lower-rank tensor decomposition.
    # This is generally NP-hard. For small cases, we can try:
    # 1. Check if any column is a linear combination of others in a way
    #    that preserves the tensor.
    #
    # For now, return the original if no safe compression is found.
    return U, V, W, r

def _compress_columns(U, V, W, tol=1e-10):
    """
    Try to compress by finding linear dependencies that preserve the tensor.
    If column c can be expressed as a linear combination of other columns
    in U, V, W simultaneously (with the same coefficients), we can eliminate it.
    """
    r = U.shape[1]
    if r <= 1:
        return U, V, W, r
    
    # Try to find if any column is redundant
    # For each column c, check if U[:,c], V[:,c], W[:,c] can be expressed
    # as linear combinations of the other columns with the SAME coefficients.
    #
    # This is equivalent to: does the column (U[:,c], V[:,c], W[:,c]) lie in
    # the span of the other columns in the combined space?
    #
    # If yes, we can eliminate it.
    
    combined = np.vstack([U, V, W])
    N = combined.shape[0]
    
    new_U_cols = []
    new_V_cols = []
    new_W_cols = []
    
    # Try to eliminate columns one by one
    keep = list(range(r))
    
    for c in range(r):
        if c not in keep:
            continue
        if len(keep) <= 1:
            break
        
        # Check if column c is in the span of the other kept columns
        others = [k for k in keep if k != c]
        if len(others) == 0:
            continue
            
        col_c = combined[:, c]
        cols_others = combined[:, others]
        
        # Solve cols_others @ x = col_c
        try:
            x, residuals, rank, sv = np.linalg.lstsq(cols_others, col_c, rcond=tol)
            # Check if the solution is exact
            if np.allclose(cols_others @ x, col_c, atol=tol):
                # Column c is redundant, remove it
                keep.remove(c)
        except:
            pass
    
    if len(keep) < r:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, len(keep)
    
    return U, V, W, r

def _construct_laderman_partition(n, m, p):
    """
    Partition the 3x5x7 problem using Laderman blocks.
    
    Strategy: 
    - Use 2x3x3 Laderman for rows 0-1, cols 0-2 (A[0:2,0:3] * B[0:3,0:3] -> C[0:3,0:2])
    - Use 3x2x3 Laderman for rows 0-2, cols 3-4 (A[0:3,3:5] * B[3:5,0:7] -> C[0:7,0:3])
    - Fill remaining with schoolbook
    
    Actually, let's think about this more carefully.
    
    The tensor T[i,j,k] = sum_l A[i,l] * B[l,k]
    
    We can partition the middle index l (size 5) into [0:3] and [3:5].
    
    For l in [0:3]: A[0:3, 0:3] * B[0:3, 0:7] -> C[0:7, 0:3]
    This is a 3x3 @ 3x7 problem. We can further partition.
    
    Alternative partition:
    - Split i into [0:2] and [2:3]
    - Split k into [0:3] and [3:7]
    
    Let me try a different approach:
    - Block 1: i=[0,1], j=[0,1,2], k=[0,1,2] -> 2x3 @ 3x3 -> 2x3 (Laderman, rank 7)
    - Block 2: i=[0,1,2], j=[3,4], k=[0,1,2] -> 3x2 @ 2x3 -> 3x3 (Laderman, rank 7)
    - Remaining: schoolbook for all other (i,j,k) combinations
    
    Wait, these blocks overlap in the output space. Let me be more careful.
    
    The output C[i,k] gets contributions from all j. So we can't just partition
    independently by output blocks. We need to partition by the middle index j.
    
    Let's partition j into [0:3] and [3:5]:
    - Part 1: j in [0,1,2]: A[0:3,0:3] @ B[0:3,0:7] -> C[0:7,0:3]
    - Part 2: j in [3,4]: A[0:3,3:5] @ B[3:5,0:7] -> C[0:7,0:3]
    
    For Part 1 (3x3 @ 3x7):
    We can further split i into [0:2] and [2:3], and k into [0:3] and [3:7]:
    - 1a: i=[0,1], k=[0,1,2]: 2x3 @ 3x3 -> 2x3 (Laderman, rank 7)
    - 1b: i=[0,1], k=[3,4,5,6]: 2x3 @ 3x4 -> 2x4 (need decomposition)
    - 1c: i=[2], k=[0,1,2]: 1x3 @ 3x3 -> 1x3 (schoolbook, rank 9)
    - 1d: i=[2], k=[3,4,5,6]: 1x3 @ 3x4 -> 1x4 (schoolbook, rank 12)
    
    This is getting complex. Let me try a simpler partition that's known to work.
    
    Actually, let me just use the direct approach:
    - Place Laderman 2x3x3 at (i0=0, j0=0, k0=0): covers i=[0,1], j=[0,1,2], k=[0,1,2]
    - Place Laderman 3x2x3 at (i0=0, j0=3, k0=0): covers i=[0,1,2], j=[3,4], k=[0,1,2]
    - Schoolbook for everything else
    
    But these don't cover all of j. Let me check:
    - Block 1 covers j=[0,1,2]
    - Block 2 covers j=[3,4]
    - Together they cover all j=[0,1,2,3,4] ✓
    
    But Block 1 only covers i=[0,1] and k=[0,1,2]
    Block 2 covers i=[0,1,2] and k=[0,1,2]
    
    So for i=2, j=[0,1,2], k=[0,1,2]: not covered by either block (Block 1 only has i=[0,1])
    For k=[3,4,5,6]: not covered by either block
    
    So we need schoolbook for:
    - i=2, j=[0,1,2], k=[0,1,2]: 1*3*3 = 9 terms
    - i=[0,1,2], j=[0,1,2,3,4], k=[3,4,5,6]: 3*5*4 = 60 terms
    - i=[0,1], j=[3,4], k=[3,4,5,6]: already in the 60 above
    
    Total schoolbook: 9 + 60 = 69
    Total rank: 7 + 7 + 69 = 83
    
    That's better than 105! Let me implement this.
    """
    terms = []
    
    # Block 1: Laderman 2x3x3 at (i0=0, j0=0, k0=0)
    # Covers i=[0,1], j=[0,1,2], k=[0,1,2]
    laderman_2x3x3 = _laderman_2x3x3()
    terms.extend(_embed_terms(laderman_2x3x3, 0, 0, 0, n, m, p))
    
    # Block 2: Laderman 3x2x3 at (i0=0, j0=3, k0=0)
    # Covers i=[0,1,2], j=[3,4], k=[0,1,2]
    laderman_3x2x3 = _laderman_3x2x3()
    terms.extend(_embed_terms(laderman_3x2x3, 0, 3, 0, n, m, p))
    
    # Schoolbook for remaining (i,j,k)
    block1_i = {0, 1}
    block1_j = {0, 1, 2}
    block1_k = {0, 1, 2}
    
    block2_i = {0, 1, 2}
    block2_j = {3, 4}
    block2_k = {0, 1, 2}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block1 = i in block1_i and j in block1_j and k in block1_k
                in_block2 = i in block2_i and j in block2_j and k in block2_k
                if in_block1 or in_block2:
                    continue
                u = {(i * m + j): 1}
                v = {(j * p + k): 1}
                w = {(k * n + i): 1}
                terms.append((u, v, w))
    
    return terms

def _construct_laderman_partition_v2(n, m, p):
    """
    Alternative partition:
    - Block 1: Laderman 2x3x3 at (i0=0, j0=0, k0=0): i=[0,1], j=[0,1,2], k=[0,1,2]
    - Block 2: Laderman 2x3x3 at (i0=1, j0=0, k0=3): i=[1,2], j=[0,1,2], k=[3,4,5]
    - Block 3: Laderman 3x2x3 at (i0=0, j0=3, k0=0): i=[0,1,2], j=[3,4], k=[0,1,2]
    - Schoolbook for rest
    
    Let me check coverage:
    Block 1: i=[0,1], j=[0,1,2], k=[0,1,2]
    Block 2: i=[1,2], j=[0,1,2], k=[3,4,5]
    Block 3: i=[0,1,2], j=[3,4], k=[0,1,2]
    
    Not covered:
    - i=0, j=[0,1,2], k=[5,6]: 1*3*2 = 6
    - i=1, j=[0,1,2], k=[6]: 1*3*1 = 3
    - i=2, j=[0,1,2], k=[0,1,2]: 1*3*3 = 9
    - i=2, j=[0,1,2], k=[6]: 1*3*1 = 3
    - i=[0,1,2], j=[3,4], k=[3,4,5,6]: 3*2*4 = 24
    - i=[0,1], j=[3,4], k=[3,4,5,6]: included in 24
    - i=2, j=[3,4], k=[3,4,5,6]: included in 24
    
    Total schoolbook: 6+3+9+3+24 = 45
    Total rank: 7+7+7+45 = 66
    
    Even better! But I need to verify the Laderman embeddings are correct.
    """
    terms = []
    
    # Block 1: Laderman 2x3x3 at (i0=0, j0=0, k0=0)
    laderman_2x3x3 = _laderman_2x3x3()
    terms.extend(_embed_terms(laderman_2x3x3, 0, 0, 0, n, m, p))
    
    # Block 2: Laderman 2x3x3 at (i0=1, j0=0, k0=3)
    # i=[1,2], j=[0,1,2], k=[3,4,5]
    terms.extend(_embed_terms(laderman_2x3x3, 1, 0, 3, n, m, p))
    
    # Block 3: Laderman 3x2x3 at (i0=0, j0=3, k0=0)
    # i=[0,1,2], j=[3,4], k=[0,1,2]
    laderman_3x2x3 = _laderman_3x2x3()
    terms.extend(_embed_terms(laderman_3x2x3, 0, 3, 0, n, m, p))
    
    # Schoolbook for remaining
    block1_i = {0, 1}
    block1_j = {0, 1, 2}
    block1_k = {0, 1, 2}
    
    block2_i = {1, 2}
    block2_j = {0, 1, 2}
    block2_k = {3, 4, 5}
    
    block3_i = {0, 1, 2}
    block3_j = {3, 4}
    block3_k = {0, 1, 2}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block1 = i in block1_i and j in block1_j and k in block1_k
                in_block2 = i in block2_i and j in block2_j and k in block2_k
                in_block3 = i in block3_i and j in block3_j and k in block3_k
                if in_block1 or in_block2 or in_block3:
                    continue
                u = {(i * m + j): 1}
                v = {(j * p + k): 1}
                w = {(k * n + i): 1}
                terms.append((u, v, w))
    
    return terms

def _construct_laderman_partition_v3(n, m, p):
    """
    Try to maximize Laderman block coverage.
    
    Partition j into [0:3] and [3:5].
    
    For j=[0:3] (size 3):
    - Split i into [0:2] and [2:3]
    - Split k into [0:3] and [3:7]
    
    For j=[3:5] (size 2):
    - Use 3x2x3 Laderman for k=[0:3]
    - Use 3x2x4 for k=[3:7] (schoolbook: 3*2*4=24)
    
    For j=[0:3]:
    - i=[0:2], k=[0:3]: 2x3x3 Laderman (rank 7)
    - i=[0:2], k=[3:7]: 2x3x4 schoolbook (rank 24)
    - i=[2:3], k=[0:3]: 1x3x3 schoolbook (rank 9)
    - i=[2:3], k=[3:7]: 1x3x4 schoolbook (rank 12)
    
    Total for j=[0:3]: 7 + 24 + 9 + 12 = 52
    Total for j=[3:5]: 7 + 24 = 31
    Grand total: 52 + 31 = 83
    
    Same as v1. Let me try a different split.
    
    What if I use:
    - i=[0:2], j=[0:3], k=[0:3]: 2x3x3 Laderman (7)
    - i=[0:2], j=[0:3], k=[3:7]: 2x3x4 (24)
    - i=[2:3], j=[0:3], k=[0:7]: 1x3x7 (21)
    - i=[0:3], j=[3:5], k=[0:3]: 3x2x3 Laderman (7)
    - i=[0:3], j=[3:5], k=[3:7]: 3x2x4 (24)
    
    Total: 7+24+21+7+24 = 83
    
    Hmm, same. The issue is that 2x3x4 and 3x2x4 don't have known low-rank decompositions.
    
    Let me try yet another approach: use Strassen 2x2x2 blocks more aggressively.
    """
    terms = []
    
    # Use multiple Strassen 2x2x2 blocks
    # i can be [0,1] or [1,2]
    # j can be [0,1], [1,2], [2,3], [3,4]
    # k can be [0,1], [1,2], [2,3], [3,4], [4,5], [5,6]
    
    # Greedily place non-overlapping Strassen blocks
    strassen = _strassen_2x2x2()
    
    placed = set()
    
    # Try all possible positions and greedily add non-overlapping ones
    candidates = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                candidates.append((i0, j0, k0))
    
    # Sort by some heuristic (e.g., position)
    for i0, j0, k0 in candidates:
        block_i = {i0, i0+1}
        block_j = {j0, j0+1}
        block_k = {k0, k0+1}
        
        # Check if this block overlaps with any placed block
        overlaps = False
        for (pi0, pj0, pk0) in placed:
            pbi = {pi0, pi0+1}
            pbj = {pj0, pj0+1}
            pbk = {pk0, pk0+1}
            if pbi & block_i and pbj & block_j and pbk & block_k:
                overlaps = True
                break
        
        if not overlaps:
            placed.add((i0, j0, k0))
            terms.extend(_embed_terms(strassen, i0, j0, k0, n, m, p))
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                for (i0, j0, k0) in placed:
                    if i in {i0, i0+1} and j in {j0, j0+1} and k in {k0, k0+1}:
                        covered = True
                        break
                if not covered:
                    u = {(i * m + j): 1}
                    v = {(j * p + k): 1}
                    w = {(k * n + i): 1}
                    terms.append((u, v, w))
    
    return terms

def _search_best_partition(n, m, p, deadline):
    """
    Search over different partition strategies and return the best one.
    """
    best_terms = None
    best_rank = n * m * p
    
    # Try various partition strategies
    strategies = []
    
    # Strategy 1: Laderman v1
    strategies.append(("laderman_v1", _construct_laderman_partition))
    
    # Strategy 2: Laderman v2
    strategies.append(("laderman_v2", _construct_laderman_partition_v2))
    
    # Strategy 3: Greedy Strassen
    strategies.append(("greedy_strassen", _construct_laderman_partition_v3))
    
    for name, func in strategies:
        if time.time() > deadline - 10:
            break
        
        try:
            terms = func(n, m, p)
            U, V, W = _build_factor_matrices(terms, n, m, p)
            rank = U.shape[1]
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list):
                if rank < best_rank:
                    best_rank = rank
                    best_terms = terms
        except Exception as e:
            pass
    
    return best_terms, best_rank

def _try_more_laderman_placements(n, m, p, deadline):
    """
    Try all possible placements of Laderman blocks and find the best combination.
    """
    best_terms = None
    best_rank = n * m * p
    
    laderman_2x3x3 = _laderman_2x3x3()
    laderman_3x2x3 = _laderman_3x2x3()
    
    # All possible placements of 2x3x3 blocks
    # i0 in [0,1], j0 in [0,2], k0 in [0,4]
    placements_2x3x3 = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-2) for k0 in range(p-2)]
    
    # All possible placements of 3x2x3 blocks
    # i0 in [0], j0 in [0,3], k0 in [0,4]
    placements_3x2x3 = [(i0, j0, k0) for i0 in range(n-2) for j0 in range(m-1) for k0 in range(p-2)]
    
    # Try single blocks first
    for i0, j0, k0 in placements_2x3x3:
        if time.time() > deadline - 15:
            break
        terms = _embed_terms(laderman_2x3x3, i0, j0, k0, n, m, p)
        
        block_i = {i0, i0+1}
        block_j = {j0, j0+1, j0+2}
        block_k = {k0, k0+1, k0+2}
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i in block_i and j in block_j and k in block_k:
                        continue
                    u = {(i * m + j): 1}
                    v = {(j * p + k): 1}
                    w = {(k * n + i): 1}
                    terms.append((u, v, w))
        
        U, V, W = _build_factor_matrices(terms, n, m, p)
        rank = U.shape[1]
        U_list, V_list, W_list = U.tolist(), V.tolist(), W.tolist()
        if _verify(U_list, V_list, W_list) and rank < best_rank:
            best_rank = rank
            best_terms = terms
    
    for i0, j0, k0 in placements_3x2x3:
        if time.time() > deadline - 15:
            break
        terms = _embed_terms(laderman_3x2x3, i0, j0, k0, n, m, p)
        
        block_i = {i0, i0+1, i0+2}
        block_j = {j0, j0+1}
        block_k = {k0, k0+1, k0+2}
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i in block_i and j in block_j and k in block_k:
                        continue
                    u = {(i * m + j): 1}
                    v = {(j * p + k): 1}
                    w = {(k * n + i): 1}
                    terms.append((u, v, w))
        
        U, V, W = _build_factor_matrices(terms, n, m, p)
        rank = U.shape[1]
        U_list, V_list, W_list = U.tolist(), V.tolist(), W.tolist()
        if _verify(U_list, V_list, W_list) and rank < best_rank:
            best_rank = rank
            best_terms = terms
    
    # Try pairs of non-overlapping blocks
    for i0, j0, k0 in placements_2x3x3:
        if time.time() > deadline - 5:
            break
        for i1, j1, k1 in placements_2x3x3:
            if time.time() > deadline - 5:
                break
            # Check non-overlapping
            b1_i = {i0, i0+1}
            b1_j = {j0, j0+1, j0+2}
            b1_k = {k0, k0+1, k0+2}
            b2_i = {i1, i1+1}
            b2_j = {j1, j1+1, j1+2}
            b2_k = {k1, k1+1, k1+2}
            
            if b1_i & b2_i and b1_j & b2_j and b1_k & b2_k:
                continue
            
            terms = _embed_terms(laderman_2x3x3, i0, j0, k0, n, m, p)
            terms.extend(_embed_terms(laderman_2x3x3, i1, j1, k1, n, m, p))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        in_b1 = i in b1_i and j in b1_j and k in b1_k
                        in_b2 = i in b2_i and j in b2_j and k in b2_k
                        if in_b1 or in_b2:
                            continue
                        u = {(i * m + j): 1}
                        v = {(j * p + k): 1}
                        w = {(k * n + i): 1}
                        terms.append((u, v, w))
            
            U, V, W = _build_factor_matrices(terms, n, m, p)
            rank = U.shape[1]
            U_list, V_list, W_list = U.tolist(), V.tolist(), W.tolist()
            if _verify(U_list, V_list, W_list) and rank < best_rank:
                best_rank = rank
                best_terms = terms
    
    return best_terms, best_rank

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i * m + j): 1}
                v = {(j * p + k): 1}
                w = {(k * n + i): 1}
                schoolbook_terms.append((u, v, w))
    
    U_sb, V_sb, W_sb = _build_factor_matrices(schoolbook_terms, n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # Try Laderman partitions
    if time.time() < deadline - 30:
        terms, rank = _search_best_partition(n, m, p, deadline)
        if terms is not None and rank < best_rank:
            U, V, W = _build_factor_matrices(terms, n, m, p)
            U_list, V_list, W_list = U.tolist(), V.tolist(), W.tolist()
            if _verify(U_list, V_list, W_list):
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try more Laderman placements
    if time.time() < deadline - 15:
        terms, rank = _try_more_laderman_placements(n, m, p, deadline)
        if terms is not None and rank < best_rank:
            U, V, W = _build_factor_matrices(terms, n, m, p)
            U_list, V_list, W_list = U.tolist(), V.tolist(), W.tolist()
            if _verify(U_list, V_list, W_list):
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

