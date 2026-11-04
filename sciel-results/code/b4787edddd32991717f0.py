import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _build_rank33():
    """Return (U, V, W) as lists for the known rank-33 Strassen solution."""
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for k in range(p):
        U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
        U[3][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
    for x in range(3):
        cs = 2*x
        U[1][col]=1;U[5][col]=1;V[6+cs][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=1;W[2*cs+3][col]=1;col+=1
        U[4][col]=1;U[5][col]=1;V[6+cs][col]=1;W[2*cs+1][col]=1;W[2*cs+3][col]=-1;col+=1
        U[1][col]=1;V[6+cs+1][col]=1;V[12+cs+1][col]=-1;W[2*cs+2][col]=1;W[2*cs+3][col]=1;col+=1
        U[5][col]=1;V[12+cs][col]=1;V[6+cs][col]=-1;W[2*cs][col]=1;W[2*cs+1][col]=1;col+=1
        U[1][col]=1;U[2][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=-1;W[2*cs+2][col]=1;col+=1
        U[4][col]=1;U[1][col]=-1;V[6+cs][col]=1;V[6+cs+1][col]=1;W[2*cs+3][col]=1;col+=1
        U[2][col]=1;U[5][col]=-1;V[12+cs][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=1;col+=1
    return U, V, W

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 2x3*3x6 multiplication tensor."""
    n, m, p = 2, 3, 6
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    return np.max(np.abs(T_est - T_target)) < 1e-9

def _find_2x3x3_rank10(deadline):
    """
    Find a rank-10 decomposition of the 2x3 * 3x3 -> 2x3 multiplication tensor.
    """
    n_sub, m_sub, p_sub = 2, 3, 3
    r = 10
    U = np.zeros((n_sub * m_sub, r))
    V = np.zeros((m_sub * p_sub, r))
    W = np.zeros((p_sub * n_sub, r))
    
    T_target = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for s in range(m_sub):
            for j in range(p_sub):
                a_idx = i * m_sub + s
                b_idx = s * p_sub + j
                c_idx = j * n_sub + i
                T_target[a_idx, b_idx, c_idx] = 1.0
                
    best_cost = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Multiple random restarts
    for restart in range(200):
        if time.time() > deadline - 20:
            break
            
        np.random.seed(42 + restart)
        U = np.random.randn(n_sub * m_sub, r)
        V = np.random.randn(m_sub * p_sub, r)
        W = np.random.randn(p_sub * n_sub, r)
        
        for iteration in range(1000):
            if time.time() > deadline - 20:
                break
                
            # Vectorized ALS Update for U
            num_U = np.einsum('ijk,jr,kr->ir', T_target, V, W)
            den_U = np.einsum('jr,kr->r', V**2, W**2)
            U = num_U / den_U
            
            # Vectorized ALS Update for V
            num_V = np.einsum('ijk,ir,kr->jr', T_target, U, W)
            den_V = np.einsum('ir,kr->r', U**2, W**2)
            V = num_V / den_V
            
            # Vectorized ALS Update for W
            num_W = np.einsum('ijk,ir,jr->kr', T_target, U, V)
            den_W = np.einsum('ir,jr->r', U**2, V**2)
            W = num_W / den_W
            
            # Compute cost
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            cost = np.sum((T_est - T_target) ** 2)
            
            if cost < best_cost:
                best_cost = cost
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if cost < 1e-12:
                break
        
        if best_cost < 1e-12:
            break

    if best_U is not None and best_cost < 1e-9:
        # Round to nearest half-integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        T_est = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(T_est - T_target)) < 1e-9:
            return U_rounded, V_rounded, W_rounded, r
            
    return None, None, None, r

def _try_small_rank_search(deadline):
    """
    Attempt to find a rank < 33 decomposition using sub-tensor decomposition.
    Split 2x3*3x6 into two 2x3*3x3 sub-problems.
    """
    n, m, p = 2, 3, 6
    
    # Try to find rank-10 decomposition of 2x3x3
    U_sub, V_sub, W_sub, r_sub = _find_2x3x3_rank10(deadline)
    
    if U_sub is not None:
        # Combine two sub-decompositions
        total_rank = 2 * r_sub
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # First sub-problem (columns 0-2)
        for col in range(r_sub):
            for i in range(n * m):
                U[i][col] = U_sub[i, col]
            for s in range(m):
                for j in range(3):
                    V[s * p + j][col] = V_sub[s * 3 + j, col]
            for j in range(3):
                for i in range(n):
                    W[j * n + i][col] = W_sub[j * n + i, col]
        
        # Second sub-problem (columns 3-5)
        for col in range(r_sub):
            for i in range(n * m):
                U[i][r_sub + col] = U_sub[i, col]
            for s in range(m):
                for j in range(3):
                    V[s * p + (3 + j)][r_sub + col] = V_sub[s * 3 + j, col]
            for j in range(3):
                for i in range(n):
                    W[(3 + j) * n + i][r_sub + col] = W_sub[j * n + i, col]
        
        # Verify
        if _verify(U, V, W):
            return U, V, W, total_rank
    
    # Fallback to rank 33
    return _build_rank33(), 33

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # First, get the known rank-33 solution
    U, V, W = _build_rank33()
    best_rank = 33
    
    # Verify the solution
    if not _verify(U, V, W):
        # Should not happen, but fallback to naive
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for r in range(n):
            for s in range(m):
                for c in range(p):
                    U[r*m+s][col] = 1
                    V[s*p+c][col] = 1
                    W[c*n+r][col] = 1
                    col += 1
        best_rank = n*m*p
    
    # Try to improve with sub-tensor search
    try:
        U2, V2, W2, rank2 = _try_small_rank_search(deadline)
        if U2 is not None and rank2 < best_rank and _verify(U2, V2, W2):
            U, V, W = U2, V2, W2
            best_rank = rank2
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
