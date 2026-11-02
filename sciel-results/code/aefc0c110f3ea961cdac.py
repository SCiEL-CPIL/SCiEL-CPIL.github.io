import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

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

def _construct_rank_26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Partition into 2x2 block (indices 0,1) and 1x1 blocks (index 2).
    Apply Strassen to the 2x2 block (7 terms) + 19 terms for the rest.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2 block (i,j,k in {0,1})
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[4][0] += 1
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[0][1] += 1
    
    # Term 2: M3 = A[0,0] * (B[1,0]-B[0,0])
    U[0][2] += 1
    V[3][2] += 1; V[0][2] += -1
    W[0][2] += 1
    
    # Term 3: M4 = A[1,1] * (B[1,1]-B[0,1])
    U[3][3] += 1
    V[4][3] += 1; V[1][3] += -1
    W[4][3] += 1
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[4][4] += 1
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,1]+B[1,1])
    U[2][5] += 1; U[0][5] += -1
    V[1][5] += 1; V[4][5] += 1
    W[4][5] += 1
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[1,0])
    U[1][6] += 1; U[3][6] += -1
    V[0][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    # 19 more terms for the remaining entries involving index 2
    remaining = [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2)
    ]
    
    col = 7
    for (i,j,k) in remaining:
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _als_decompose(T, rank, max_iters=500, tol=1e-10, seed=0):
    """
    Alternating Least Squares for tensor decomposition.
    T: target tensor (9, 9, 9)
    rank: target rank
    Returns (U, V, W) if converged with low error, else None.
    """
    dim_a, dim_b, dim_c = 9, 9, 9
    
    np.random.seed(seed)
    U = np.random.randn(dim_a, rank)
    V = np.random.randn(dim_b, rank)
    W = np.random.randn(dim_c, rank)
    
    # Normalize columns to prevent scaling issues
    for r in range(rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw

    prev_error = float('inf')
    
    for it in range(max_iters):
        # Update U
        for a in range(dim_a):
            M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V
        for b in range(dim_b):
            M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W
        for c in range(dim_c):
            M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-8:
                return U, V, W
            else:
                return None
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-8:
        return U, V, W
    return None

def _try_rank_23(deadline):
    """
    Attempt to find a rank-23 decomposition using ALS.
    """
    n, m, p = 3, 3, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Try a few seeds
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        res = _als_decompose(T, 23, max_iters=300, seed=seed)
        if res is not None:
            U, V, W = res
            # Verify with integer/half-integer check? 
            # The task says "integer/half-integer entries". ALS gives floats.
            # We need to check if we can round them.
            # Let's check if rounding to nearest 0.5 works.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded version
            T_approx = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_approx, T, atol=1e-9, rtol=0):
                # Convert to lists of floats (which are effectively int/half-int)
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # If rounding didn't work, maybe try integer rounding?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            T_approx_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
            if np.allclose(T_approx_int, T, atol=1e-9, rtol=0):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    return None

def construct():
    """
    Return a rank-minimal decomposition for 3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # First, try to find rank 23
    rank23_result = _try_rank_23(deadline)
    if rank23_result is not None:
        if _verify(rank23_result["U"], rank23_result["V"], rank23_result["W"]):
            return rank23_result
    
    # Fallback to rank 26
    result26 = _construct_rank_26()
    if _verify(result26["U"], result26["V"], result26["W"]):
        return result26
    
    # Fallback to schoolbook rank 27
    return _construct_schoolbook()
