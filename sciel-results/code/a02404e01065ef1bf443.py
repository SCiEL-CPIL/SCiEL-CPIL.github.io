"""Tensor decomposition search for 3x4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]
import time
import numpy as np
from fractions import Fraction

def _validate(U, V, W, n, m, p, atol=1e-10):
    """Check that U, V, W form a valid tensor decomposition."""
    rank = len(U[0]) if rank > 0 else U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rng = np.random.RandomState(12345)
    for trial in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB = A @ B
        
        # Compute decomposed product
        # For each c: scalar_U = sum(A * U_c), scalar_V = sum(B * V_c)
        # result += scalar_U * scalar_V * W_c^T
        result = np.zeros((n, p))
        for c in range(rank):
            U_c = U_arr[:, c].reshape(n, m)
            V_c = V_arr[:, c].reshape(m, p)
            W_c = W_arr[:, c].reshape(p, n)
            scalar_U = np.sum(A * U_c)
            scalar_V = np.sum(B * V_c)
            result += scalar_U * scalar_V * W_c.T
        
        if not np.allclose(result, AB, atol=atol):
            return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, rank, max_iter=200, seed=42, tol=1e-12):
    """
    Use ALS (Alternating Least Squares) to find a rank-r decomposition.
    Returns (U, V, W) as numpy arrays or None if it fails.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize
    for c in range(rank):
        norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if norm > 1e-15:
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: for each i,j, solve for U[i*m+j, :]
        # The contribution of term c to entry (i,j,k) is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # For fixed (i,j), this is a linear system in U[i*m+j, :]
        # Let's define: for fixed i,j, the equation is:
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k in range(p)
        # So we have p equations in rank unknowns (U[i*m+j, 0..rank-1])
        
        for i in range(n):
            for j in range(m):
                # Build the system: A_sys * u_vec = b_vec
                # A_sys[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # b_vec[k] = 1
                A_sys = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A_sys[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b_vec = np.ones(p)
                
                if rank >= p:
                    # Least squares
                    u_vec, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                else:
                    # Underdetermined - use minimum norm solution
                    if rank == 0:
                        u_vec = np.zeros(0)
                    else:
                        # Use pseudo-inverse
                        u_vec = np.linalg.pinv(A_sys) @ b_vec
                
                U[i*m+j, :] = u_vec
        
        # Update V: for each j,k, solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # b_vec[i] = 1
                A_sys = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b_vec = np.ones(n)
                
                if rank >= n:
                    v_vec, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                else:
                    if rank == 0:
                        v_vec = np.zeros(0)
                    else:
                        v_vec = np.linalg.pinv(A_sys) @ b_vec
                
                V[j*p+k, :] = v_vec
        
        # Update W: for each k,i, solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # Equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # b_vec[j] = 1
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b_vec = np.ones(m)
                
                if rank >= m:
                    w_vec, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                else:
                    if rank == 0:
                        w_vec = np.zeros(0)
                    else:
                        w_vec = np.linalg.pinv(A_sys) @ b_vec
                
                W[k*n+i, :] = w_vec
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residual += (val - 1.0)**2
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    # Check if solution is valid
    if best_residual < 1e-8:
        return best_U, best_V, best_W, best_residual
    return None

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Rank 7 instead of 8.
    """
    # Strassen's 7 multiplication terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices: A[i,j] -> U[i*m+j, c], B[j,k] -> V[j*p+k, c], C[k,i] -> W[k*n+i, c]
    # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
    # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[2], h=B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    
    # M3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = d(g-e)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # e
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[0, 4] = -1 # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1  # c
    U[0, 5] = -1 # a
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple block decompositions into a full decomposition.
    This is a placeholder - proper combination requires careful index mapping.
    """
    pass

def _try_strassen_based(n, m, p):
    """
    Try to construct a decomposition using Strassen-like techniques.
    For 3x4x6, we can partition as:
    - A (3x4) into blocks: top 2 rows + bottom 1 row, or other splits
    - B (4x6) into blocks: left 4 cols as 2x2 blocks of 2x2
    - C (3x6) as result
    
    One approach: treat the 4x4 middle as 2x2 blocks of 2x2.
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    B = [[B11, B12, B13], [B21, B22, B23]] where B11,B21 are 2x2, etc.
    
    Actually, let's try a simpler approach:
    Split A into A_top (2x4) and A_bot (1x4)
    Split B into B_left (4x3) and B_right (4x3)
    Then C_top = A_top @ B_left, C_top2 = A_top @ B_right
    C_bot = A_bot @ B_left, C_bot2 = A_bot @ B_right
    
    Each of these is a 2x4 x 4x3 or 1x4 x 4x3 multiplication.
    For 2x4 x 4x3: we can further decompose using Strassen on 2x2 blocks.
    """
    pass

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_decomp = sb
    
    # Try various ranks from lower to higher
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But realistically, we'll try ranks that might be achievable
    
    # First, try to see if we can do better than schoolbook with ALS
    # Try ranks: 60, 55, 50, 45, 40, 36, 30, 24
    ranks_to_try = [60, 55, 50, 48, 45, 42, 40, 36, 32, 30, 28, 24]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try multiple seeds
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            
            result = _als_decompose(n, m, p, target_rank, max_iter=50, seed=seed, tol=1e-10)
            
            if result is not None:
                U, V, W, residual = result
                # Convert to list format
                U_list = [[U[i, c] for c in range(target_rank)] for i in range(n * m)]
                V_list = [[V[i, c] for c in range(target_rank)] for i in range(m * p)]
                W_list = [[W[i, c] for c in range(target_rank)] for i in range(p * n)]
                
                # Validate
                if _validate(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_decomp = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found a valid decomposition for this rank, try lower
    
    return best_decomp
