import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# Precompute the target tensor T with entries T[i*m+j, j*p+k, k*n+i] = 1
n, m, p = DIMENSIONS
def _target_tensor():
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

_T_cache = None
def _get_T():
    global _T_cache
    if _T_cache is None:
        _T_cache = _target_tensor()
    return _T_cache

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    T = _get_T()
    # Compute reconstructed tensor: R_(a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
    # Efficiently: U (nm x R), V (mp x R), W (pn x R)
    # For each r: outer product U[:,r], V[:,r], W[:,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < 1e-9

def _round_half_int(M, tol=1e-6):
    """Round a matrix to nearest half-integers if possible."""
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _als_decompose(T, R, n, m, p, max_iter=200, tol=1e-7):
    """
    Alternating least squares to find U (nm x R), V (mp x R), W (pn x R)
    such that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c].
    Vectorized version for speed.
    """
    nm, mp, pn = n*m, m*p, p*n
    # Structured initialization: start with schoolbook and add small noise
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    # Initialize with a subset of schoolbook columns if R < nm*mp*pn
    schoolbook_cols = min(R, nm * mp * pn)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= schoolbook_cols:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # Add small random perturbation to avoid degenerate solutions
    U += np.random.randn(nm, R) * 0.01
    V += np.random.randn(mp, R) * 0.01
    W += np.random.randn(pn, R) * 0.01

    for it in range(max_iter):
        # Update U: vectorized least squares
        # For each a: solve min_X sum_{b,c} (sum_r X[r] V[b,r] W[c,r] - T[a,b,c])^2
        # This is: (V*W)^T (V*W) X = (V*W)^T T[a]
        # where (V*W)[(b,c),r] = V[b,r]*W[c,r]
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, R)
        MW_flat = MW.reshape(mp * pn, R)
        # Compute Gram matrix and RHS for all a at once
        # Gram = MW_flat.T @ MW_flat (R x R)
        # For each a: rhs_a = MW_flat.T @ T[a].reshape(mp*pn)
        # U = solve(Gram, rhs_all) where rhs_all is (nm, R)
        T_reshaped = T.reshape(nm, mp * pn)
        # rhs_all[a, r] = sum_{b,c} T[a,b,c] * MW_flat[(b,c), r]
        rhs_all = T_reshaped @ MW_flat  # (nm, R)
        Gram = MW_flat.T @ MW_flat  # (R, R)
        # Add small regularization for numerical stability
        Gram += np.eye(R) * 1e-10
        try:
            U = np.linalg.solve(Gram, rhs_all.T).T
        except np.linalg.LinAlgError:
            U, _, _, _ = np.linalg.lstsq(Gram, rhs_all.T, rcond=None)
            U = U.T

        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, R)
        UW_flat = UW.reshape(nm * pn, R)
        T_reshaped_v = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
        rhs_all_v = T_reshaped_v @ UW_flat  # (mp, R)
        Gram_v = UW_flat.T @ UW_flat  # (R, R)
        Gram_v += np.eye(R) * 1e-10
        try:
            V = np.linalg.solve(Gram_v, rhs_all_v.T).T
        except np.linalg.LinAlgError:
            V, _, _, _ = np.linalg.lstsq(Gram_v, rhs_all_v.T, rcond=None)
            V = V.T

        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, R)
        UV_flat = UV.reshape(nm * mp, R)
        T_reshaped_w = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
        rhs_all_w = T_reshaped_w @ UV_flat  # (pn, R)
        Gram_w = UV_flat.T @ UV_flat  # (R, R)
        Gram_w += np.eye(R) * 1e-10
        try:
            W = np.linalg.solve(Gram_w, rhs_all_w.T).T
        except np.linalg.LinAlgError:
            W, _, _, _ = np.linalg.lstsq(Gram_w, rhs_all_w.T, rcond=None)
            W = W.T

        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break

    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        # Try to round to half-integers
        U_r, ok1 = _round_half_int(U, tol=1e-4)
        if ok1:
            V_r, ok2 = _round_half_int(V, tol=1e-4)
            if ok2:
                W_r, ok3 = _round_half_int(W, tol=1e-4)
                if ok3:
                    if _verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
    return None

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
      U[a, r] with a = i*2+j (i,j in {0,1})
      V[b, r] with b = j*2+k (j,k in {0,1})
      W[c, r] with c = k*2+i (k,i in {0,1})
    such that sum_r U[a,r]*V[b,r]*W[c,r] = 1 if (i,j,k) valid else 0.
    """
    # Strassen's 7 intermediate products:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A22)(B11 - B12)
    # M4 = (A11 + A22)(B22 + B11)
    # M5 = (A12 + A22)(B21 + B22)
    # M6 = (A11 - A21)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B11)
    #
    # C11 = M4 + M3 - M5 + M7
    # C12 = M1 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    #
    # Indexing: a = i*2+j, b = j*2+k, c = k*2+i
    # A elements: A[i,j] -> index i*2+j
    # B elements: B[j,k] -> index j*2+k
    # C elements: C[i,k] -> index k*2+i (trace convention: W[k*n+i] with n=2)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11)(B12 - B22): A[0,0]=idx 0, B[1,2]=idx 4, B[2,2]=idx 6 (0-indexed: B[1,1]=idx 3, B[0,1]=idx 1)
    # Let me use 0-indexed: A[i,j], B[j,k], C[i,k]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices (k*2+i): C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    #   C[i,k] with c = k*2+i: C[0,0]: k=0,i=0 -> 0; C[0,1]: k=1,i=0 -> 2; C[1,0]: k=0,i=1 -> 1; C[1,1]: k=1,i=1 -> 3
    
    # M1 = A[0,0] * (B[1,1] - B[0,1])
    U[0][0] = 1  # A[0,0]
    V[3][0] = 1  # B[1,1]
    V[1][0] = -1 # B[0,1]
    # M1 contributes to C[0,1] and C[1,1]
    # C[0,1]: k=1,i=0 -> c=2
    # C[1,1]: k=1,i=1 -> c=3
    W[2][0] = 1
    W[3][0] = 1
    
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    U[0][1] = 1  # A[0,0]
    U[2][1] = 1  # A[1,0]
    V[3][1] = 1  # B[1,1]
    # M2 contributes to C[1,0] and C[1,1]
    # C[1,0]: k=0,i=1 -> c=1
    # C[1,1]: k=1,i=1 -> c=3
    W[1][1] = 1
    W[3][1] = 1
    
    # M3 = A[1,1] * (B[0,0] - B[0,1])
    U[3][2] = 1  # A[1,1]
    V[0][2] = 1  # B[0,0]
    V[1][2] = -1 # B[0,1]
    # M3 contributes to C[0,0] and C[1,1]
    # C[0,0]: k=0,i=0 -> c=0
    # C[1,1]: k=1,i=1 -> c=3
    W[0][2] = 1
    W[3][2] = -1
    
    # M4 = (A[0,0] + A[1,1]) * (B[1,1] + B[0,0])
    U[0][3] = 1  # A[0,0]
    U[3][3] = 1  # A[1,1]
    V[3][3] = 1  # B[1,1]
    V[0][3] = 1  # B[0,0]
    # M4 contributes to C[0,0] and C[1,0]
    # C[0,0]: k=0,i=0 -> c=0
    # C[1,0]: k=0,i=1 -> c=1
    W[0][3] = 1
    W[1][3] = 1
    
    # M5 = (A[0,1] + A[1,1]) * (B[1,0] + B[1,1])
    U[1][4] = 1  # A[0,1]
    U[3][4] = 1  # A[1,1]
    V[2][4] = 1  # B[1,0]
    V[3][4] = 1  # B[1,1]
    # M5 contributes to C[0,0] and C[0,1]
    # C[0,0]: k=0,i=0 -> c=0
    # C[0,1]: k=1,i=0 -> c=2
    W[0][4] = -1
    W[2][4] = 1
    
    # M6 = (A[0,0] - A[1,0]) * (B[0,0] + B[0,1])
    U[0][5] = 1  # A[0,0]
    U[2][5] = -1 # A[1,0]
    V[0][5] = 1  # B[0,0]
    V[1][5] = 1  # B[0,1]
    # M6 contributes to C[1,1]
    # C[1,1]: k=1,i=1 -> c=3
    W[3][5] = -1
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[0,0])
    U[1][6] = 1  # A[0,1]
    U[3][6] = -1 # A[1,1]
    V[2][6] = 1  # B[1,0]
    V[0][6] = 1  # B[0,0]
    # M7 contributes to C[0,0]
    # C[0,0]: k=0,i=0 -> c=0
    W[0][6] = 1
    
    return U, V, W

def _construct_block_3x2x6():
    """
    Construct a rank-36 (schoolbook) decomposition for 3x2x6 matrix multiplication.
    Returns (U, V, W) with shapes (6, 36), (12, 36), (18, 36)
    where U indices: i*2+j (i in 0..2, j in 0..1)
          V indices: j*6+k (j in 0..1, k in 0..5)
          W indices: k*3+i (k in 0..5, i in 0..2)
    """
    n, m, p = 3, 2, 6
    rank = n * m * p  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _construct_block_3x3x6():
    """
    Construct a rank-54 (schoolbook) decomposition for 3x3x6 matrix multiplication.
    Returns (U, V, W) with shapes (9, 54), (18, 54), (18, 54)
    where U indices: i*3+j (i in 0..2, j in 0..2)
          V indices: j*6+k (j in 0..2, k in 0..5)
          W indices: k*3+i (k in 0..5, i in 0..2)
    """
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _construct_block_decomposition():
    """
    Construct a block decomposition for 3x5x6 by splitting the middle dimension
    into 2+3. Uses schoolbook for each block.
    Total rank: 36 + 54 = 90 (same as full schoolbook).
    """
    # Get the two blocks
    U1, V1, W1 = _construct_block_3x2x6()  # shapes (6,36), (12,36), (18,36)
    U2, V2, W2 = _construct_block_3x3x6()  # shapes (9,54), (18,54), (18,54)
    
    rank1 = 36
    rank2 = 54
    total_rank = rank1 + rank2  # 90
    
    n, m, p = 3, 5, 6
    U = [[0] * total_rank for _ in range(n * m)]  # 15 x 90
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 90
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 90
    
    # Embed block 1 (3x2x6): U1 has 6 rows (i*2+j, i in 0..2, j in 0..1)
    # In the full problem, these correspond to i*5+j for j in 0..1
    for i in range(3):
        for j in range(2):
            for r in range(rank1):
                U[i * 5 + j][r] = U1[i * 2 + j][r]
    
    # V1 has 12 rows (j*6+k, j in 0..1, k in 0..5)
    # In the full problem, these are j*6+k for j in 0..1
    for j in range(2):
        for k in range(6):
            for r in range(rank1):
                V[j * 6 + k][r] = V1[j * 6 + k][r]
    
    # W1 has 18 rows (k*3+i, k in 0..5, i in 0..2)
    # Same indexing in full problem
    for k in range(6):
        for i in range(3):
            for r in range(rank1):
                W[k * 3 + i][r] = W1[k * 3 + i][r]
    
    # Embed block 2 (3x3x6): U2 has 9 rows (i*3+j, i in 0..2, j in 0..2)
    # In the full problem, these correspond to i*5+(j+2) for j in 0..2
    for i in range(3):
        for j in range(3):
            for r in range(rank2):
                U[i * 5 + (j + 2)][rank1 + r] = U2[i * 3 + j][r]
    
    # V2 has 18 rows (j*6+k, j in 0..2, k in 0..5)
    # In the full problem, these are (j+2)*6+k for j in 0..2
    for j in range(3):
        for k in range(6):
            for r in range(rank2):
                V[(j + 2) * 6 + k][rank1 + r] = V2[j * 6 + k][r]
    
    # W2 has 18 rows (k*3+i, k in 0..5, i in 0..2)
    for k in range(6):
        for i in range(3):
            for r in range(rank2):
                W[k * 3 + i][rank1 + r] = W2[k * 3 + i][r]
    
    return {"U": U, "V": V, "W": W}

def _construct_strassen_3x3x6():
    """
    Construct a rank-reduced decomposition for 3x3x6 using Strassen on 3x3x3 sub-blocks.
    Split 6 columns into two groups of 3.
    Each 3x3x3 uses schoolbook (rank 27) since Strassen doesn't directly help for 3x3x3.
    Total rank: 54.
    
    For a true rank reduction, we'd need a non-trivial 3x3x3 decomposition,
    which is not known to have rank < 27.
    """
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave margin from 150s limit
    n, m, p = DIMENSIONS
    T = _get_T()
    
    # First, try the block decomposition (rank 90, but structured)
    # Verify it works
    block_result = _construct_block_decomposition()
    U_b = np.array(block_result["U"], dtype=float)
    V_b = np.array(block_result["V"], dtype=float)
    W_b = np.array(block_result["W"], dtype=float)
    if _verify(U_b, V_b, W_b):
        # This is rank 90, same as schoolbook. Not an improvement.
        pass
    
    # Try ranks from low to high using ALS
    for R in range(30, 91):
        if time.time() > deadline - 15.0:
            break
        
        # Try several random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            result = _als_decompose(T, R, n, m, p, max_iter=50, tol=1e-8)
            if result is not None:
                U_r, V_r, W_r = result
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                # Convert to integers if all are integers
                def to_int_or_half(lst):
                    out = []
                    for row in lst:
                        out_row = []
                        for val in row:
                            if abs(val - round(val)) < 1e-9:
                                out_row.append(int(round(val)))
                            elif abs(val * 2 - round(val * 2)) < 1e-9:
                                out_row.append(round(val * 2) / 2.0)
                            else:
                                out_row.append(val)
                        out.append(out_row)
                    return out
                return {"U": to_int_or_half(U_list), "V": to_int_or_half(V_list), "W": to_int_or_half(W_list)}
    
    # Fallback to schoolbook
    return _construct_schoolbook()
