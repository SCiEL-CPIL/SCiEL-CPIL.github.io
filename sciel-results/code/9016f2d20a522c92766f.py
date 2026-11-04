import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _als_step(n, m, p, rank, U, V, W, T, max_iters=2000, tol=1e-12):
    """
    Efficient ALS for tensor decomposition.
    T shape: (nm, mp, pn)
    U shape: (nm, rank)
    V shape: (mp, rank)
    W shape: (pn, rank)
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Pre-compute reshaped tensors for efficiency
    B_u = T.reshape(nm, -1)
    B_v = T.transpose(1, 0, 2).reshape(mp, -1)
    B_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    err = float('inf')
    for it in range(max_iters):
        # Update U
        K_u = np.einsum('br,cr->bcr', V, W)
        A_u = K_u.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            U = np.linalg.lstsq(A_u.T, B_u.T, rcond=None)[0].T
        except:
            pass

        # Update V
        K_v = np.einsum('ar,cr->acr', U, W)
        A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            V = np.linalg.lstsq(A_v.T, B_v.T, rcond=None)[0].T
        except:
            pass

        # Update W
        K_w = np.einsum('ar,br->abr', U, V)
        A_w = K_w.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            W = np.linalg.lstsq(A_w.T, B_w.T, rcond=None)[0].T
        except:
            pass
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
            
    return U, V, W, err

def _polish_decomposition(U, V, W, T, max_iters=5000, tol=1e-14):
    """High-precision polish pass with tighter tolerance."""
    nm, mp, pn = T.shape
    rank = U.shape[1]
    
    B_u = T.reshape(nm, -1)
    B_v = T.transpose(1, 0, 2).reshape(mp, -1)
    B_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    err = float('inf')
    for it in range(max_iters):
        # Update U
        K_u = np.einsum('br,cr->bcr', V, W)
        A_u = K_u.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            U = np.linalg.lstsq(A_u.T, B_u.T, rcond=None)[0].T
        except:
            break

        # Update V
        K_v = np.einsum('ar,cr->acr', U, W)
        A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            V = np.linalg.lstsq(A_v.T, B_v.T, rcond=None)[0].T
        except:
            break

        # Update W
        K_w = np.einsum('ar,br->abr', U, V)
        A_w = K_w.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            W = np.linalg.lstsq(A_w.T, B_w.T, rcond=None)[0].T
        except:
            break
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
            
    return U, V, W, err

def _try_rank_als_fast(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using fast ALS."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    U, V, W, err = _als_step(n, m, p, rank, U, V, W, T, max_iters=max_iters, tol=1e-12)
    
    if err < 1e-10:
        # Polish for higher precision
        U, V, W, err_polished = _polish_decomposition(U, V, W, T, max_iters=2000, tol=1e-14)
        if err_polished < 1e-10:
            return U, V, W, True
    
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _try_rank_with_basis_transform(n, m, p, rank, max_iters=2000, seed=42):
    """
    Try rank-r decomposition after applying a linear change of basis to the output space.
    """
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    pn = p * n  # 20
    
    # Apply random orthogonal transformation to output space
    Q = rng.randn(pn, pn)
    Q, _ = np.linalg.qr(Q)
    
    # Transform tensor: T'[a,b,c'] = sum_c T[a,b,c] * Q[c',c]
    T_transformed = np.einsum('abc,dc->abd', T, Q)
    
    # Run ALS on transformed tensor
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(pn, rank) * 0.5
    
    U, V, W, err = _als_step(n, m, p, rank, U, V, W, T_transformed, max_iters=max_iters, tol=1e-12)
    
    if err < 1e-10:
        # Transform back: W_original = W_transformed @ Q.T
        W_original = W @ Q.T
        
        # Verify against original tensor
        T_recon = np.einsum('ar,br,cr->abc', U, V, W_original)
        err_orig = np.max(np.abs(T_recon - T))
        
        if err_orig < 1e-10:
            # Polish
            U, V, W_original, err_polished = _polish_decomposition(U, V, W_original, T, max_iters=2000, tol=1e-14)
            if err_polished < 1e-10:
                return U, V, W_original, True
        
        if err_orig < 1e-10:
            return U, V, W_original, True
    
    return U, V, W, False

def _get_trivial_decomposition(n, m, p):
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _get_strassen_rank55_decomposition():
    """
    Constructs a rank-55 decomposition for 2x3 by 3x10 matrix multiplication.
    Strategy: Group the 10 columns of B into 5 pairs.
    For each pair, we multiply a 2x3 matrix by a 3x2 matrix.
    We decompose the 2x3 by 3x2 multiplication (rank 11) using:
    - Strassen for the 2x2 sub-blocks (rank 7)
    - Schoolbook for the remaining 2x1 by 1x2 part (rank 4)
    Total rank per block: 7 + 4 = 11.
    Total rank: 5 * 11 = 55.
    """
    n, m, p = 2, 3, 10
    rank_total = 55
    
    U = [[0] * rank_total for _ in range(n * m)] # 6 x 55
    V = [[0] * rank_total for _ in range(m * p)] # 30 x 55
    W = [[0] * rank_total for _ in range(p * n)] # 20 x 55
    
    # We process 5 blocks, each corresponding to 2 columns of B (and thus 2 columns of C)
    # Block b (0 to 4) handles B columns 2b, 2b+1 and C columns 2b, 2b+1
    
    # Strassen coefficients for 2x2 multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B22 - B11)
    # M4 = A22(B11 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the U, V, W vectors.
    # The tensor term is U[a,r] * V[b,r] * W[c,r].
    # For a specific block, we define 11 terms.
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices, coeff)
        # We will add contributions to U, V, W for each term.
        # Since U, V, W are sparse, we just set the values.
        
        # M1: (A11+A22)(B11+B22)
        # A11 is A[0,0], A22 is A[1,1] (in the 2x2 submatrix of A)
        # B11 is B[0, 2b], B22 is B[1, 2b+1] (in the 2x2 submatrix of B)
        # Note: Indices in A are (row, col). Row 0,1. Col 0,1.
        # Note: Indices in B are (row, col). Row 0,1. Col 2b, 2b+1.
        # Note: Indices in C are (row, col). Row 0,1. Col 2b, 2b+1.
        
        # Term 1: M1
        # U: A[0,0] gets +1, A[1,1] gets +1
        # V: B[0, 2b] gets +1, B[1, 2b+1] gets +1
        # W: C[0, 2b] gets +1, C[1, 2b+1] gets +1
        # Wait, Strassen M1 contributes to C11 and C22.
        # C11 += M1, C22 += M1.
        # So W[0, 2b] += 1, W[1, 2b+1] += 1.
        
        # Let's just build the 11 terms explicitly for one block and then replicate.
    ]
    
    # Helper to add a term to the matrices
    def add_term(U, V, W, u_idx, v_idx, w_idx, val):
        U[u_idx][0] += val # Placeholder, we need to handle the column index
        # Actually, let's just use numpy arrays for construction then convert
        pass

    # Let's use numpy for the construction of the 55 columns
    U_np = np.zeros((6, 55))
    V_np = np.zeros((30, 55))
    W_np = np.zeros((20, 55))
    
    for block in range(5):
        col_offset = block * 11
        b_col1 = 2 * block
        b_col2 = 2 * block + 1
        
        # A indices (6 total):
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        
        # B indices (30 total):
        # B[0, 0..9] = 0..9
        # B[1, 0..9] = 10..19
        # B[2, 0..9] = 20..29
        # For this block, we use B[0, b_col1], B[1, b_col1], B[2, b_col1]
        # and B[0, b_col2], B[1, b_col2], B[2, b_col2]
        
        # C indices (20 total):
        # C[0, 0..9] = 0..9
        # C[1, 0..9] = 10..19
        # For this block, we use C[0, b_col1], C[1, b_col1]
        # and C[0, b_col2], C[1, b_col2]
        
        # The 2x2 subproblem uses A[0:2, 0:2] and B[0:2, 2b:2b+2]
        # A_sub:
        # 0 1
        # 3 4
        # B_sub:
        # B[0, 2b] B[0, 2b+1]
        # B[1, 2b] B[1, 2b+1]
        
        # Strassen Terms for 2x2:
        
        # 1. M1 = (A00 + A11)(B00 + B11)
        # U: A00(0)+1, A11(4)+1
        # V: B00(2b)+1, B11(10+2b+1)+1 -> B[1, 2b+1]
        # W: C00(2b)+1, C11(10+2b+1)+1 -> C[1, 2b+1]
        
        U_np[0, col_offset] += 1
        U_np[4, col_offset] += 1
        V_np[2*b_col1, col_offset] += 1
        V_np[10 + 2*b_col1 + 1, col_offset] += 1
        W_np[2*b_col1, col_offset] += 1
        W_np[10 + 2*b_col1 + 1, col_offset] += 1
        
        # 2. M2 = (A10 + A11)B00
        # U: A10(3)+1, A11(4)+1
        # V: B00(2b)+1
        # W: C10(10+2b)+1
        
        U_np[3, col_offset+1] += 1
        U_np[4, col_offset+1] += 1
        V_np[2*b_col1, col_offset+1] += 1
        W_np[10 + 2*b_col1, col_offset+1] += 1
        
        # 3. M3 = A00(B11 - B00)
        # U: A00(0)+1
        # V: B11(10+2b+1)+1, B00(2b)-1
        # W: C01(2b+1)+1
        
        U_np[0, col_offset+2] += 1
        V_np[10 + 2*b_col1 + 1, col_offset+2] += 1
        V_np[2*b_col1, col_offset+2] -= 1
        W_np[2*b_col1 + 1, col_offset+2] += 1
        
        # 4. M4 = A11(B00 - B11)
        # U: A11(4)+1
        # V: B00(2b)+1, B11(10+2b+1)-1
        # W: C00(2b)+1, C10(10+2b)+1
        
        U_np[4, col_offset+3] += 1
        V_np[2*b_col1, col_offset+3] += 1
        V_np[10 + 2*b_col1 + 1, col_offset+3] -= 1
        W_np[2*b_col1, col_offset+3] += 1
        W_np[10 + 2*b_col1, col_offset+3] += 1
        
        # 5. M5 = (A00 + A01)B11
        # U: A00(0)+1, A01(1)+1
        # V: B11(10+2b+1)+1
        # W: C00(2b)-1, C01(2b+1)+1
        
        U_np[0, col_offset+4] += 1
        U_np[1, col_offset+4] += 1
        V_np[10 + 2*b_col1 + 1, col_offset+4] += 1
        W_np[2*b_col1, col_offset+4] -= 1
        W_np[2*b_col1 + 1, col_offset+4] += 1
        
        # 6. M6 = (A10 - A00)(B00 + B01)
        # U: A10(3)+1, A00(0)-1
        # V: B00(2b)+1, B01(2b+1)+1
        # W: C11(10+2b+1)+1
        
        U_np[3, col_offset+5] += 1
        U_np[0, col_offset+5] -= 1
        V_np[2*b_col1, col_offset+5] += 1
        V_np[2*b_col1 + 1, col_offset+5] += 1
        W_np[10 + 2*b_col1 + 1, col_offset+5] += 1
        
        # 7. M7 = (A01 - A11)(B10 + B11)
        # U: A01(1)+1, A11(4)-1
        # V: B10(10+2b)+1, B11(10+2b+1)+1
        # W: C00(2b)+1
        
        U_np[1, col_offset+6] += 1
        U_np[4, col_offset+6] -= 1
        V_np[10 + 2*b_col1, col_offset+6] += 1
        V_np[10 + 2*b_col1 + 1, col_offset+6] += 1
        W_np[2*b_col1, col_offset+6] += 1
        
        # Now the "tail" part: A[0:2, 2] * B[2, 2b:2b+2]
        # This is 2x1 by 1x2.
        # C[0, 2b] += A[0,2] B[2, 2b]
        # C[0, 2b+1] += A[0,2] B[2, 2b+1]
        # C[1, 2b] += A[1,2] B[2, 2b]
        # C[1, 2b+1] += A[1,2] B[2, 2b+1]
        
        # 8. Term for C[0, 2b]
        U_np[2, col_offset+7] += 1 # A[0,2]
        V_np[20 + 2*b_col1, col_offset+7] += 1 # B[2, 2b]
        W_np[2*b_col1, col_offset+7] += 1 # C[0, 2b]
        
        # 9. Term for C[0, 2b+1]
        U_np[2, col_offset+8] += 1 # A[0,2]
        V_np[20 + 2*b_col1 + 1, col_offset+8] += 1 # B[2, 2b+1]
        W_np[2*b_col1 + 1, col_offset+8] += 1 # C[0, 2b+1]
        
        # 10. Term for C[1, 2b]
        U_np[5, col_offset+9] += 1 # A[1,2]
        V_np[20 + 2*b_col1, col_offset+9] += 1 # B[2, 2b]
        W_np[10 + 2*b_col1, col_offset+9] += 1 # C[1, 2b]
        
        # 11. Term for C[1, 2b+1]
        U_np[5, col_offset+10] += 1 # A[1,2]
        V_np[20 + 2*b_col1 + 1, col_offset+10] += 1 # B[2, 2b+1]
        W_np[10 + 2*b_col1 + 1, col_offset+10] += 1 # C[1, 2b+1]
        
    # Convert to lists
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

def _try_structured_decomposition(n, m, p):
    """
    Try to find a lower rank decomposition using fast ALS with basis transformations.
    """
    n, m, p = 2, 3, 10
    
    start_time = time.time()
    
    # Try ranks from 40 to 54 (aiming for < 55)
    ranks_to_try = list(range(40, 55))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        num_seeds = 3
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            U, V, W, success = _try_rank_als_fast(n, m, p, rank, max_iters=500, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            U, V, W, success = _try_rank_with_basis_transform(n, m, p, rank, max_iters=500, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the fast ALS approach
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fall back to the known rank-55 Strassen-based decomposition
    return _get_strassen_rank55_decomposition()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
