import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 145.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns 7 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 2-vectors.
    The decomposition is:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b21-b22)
    M4 = a22(b12-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1 + M4 - M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M7
    """
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: 0=00, 1=01, 2=10, 3=11
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = a11(b21-b22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    # M4 = a22(b12-b11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = -1; W[2, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = -1
    
    return U, V, W

def _try_strassen_hybrid(n, m, p):
    """
    Try to construct a decomposition using Strassen for 2x4x4 blocks and schoolbook for the rest.
    n=2, m=5, p=5.
    Split m into 4+1, p into 4+1.
    The 2x4x4 part can be decomposed into 4 blocks of 2x2x2, each using Strassen (rank 7).
    Total Strassen rank: 4 * 7 = 28.
    Remaining terms:
    - j=4, k=0..4 (5 terms)
    - j=0..3, k=4 (4 terms)
    Total remaining: 9 terms.
    Total rank: 28 + 9 = 37.
    """
    if time.time() > _DEADLINE - 5:
        return None
    
    U_strassen, V_strassen, W_strassen = _strassen_2x2x2()
    
    # Global dimensions
    n_global = n
    m_global = m
    p_global = p
    
    U_global = np.zeros((n_global * m_global, 37))
    V_global = np.zeros((m_global * p_global, 37))
    W_global = np.zeros((p_global * n_global, 37))
    
    col = 0
    
    # 4 Strassen blocks
    # Block 1: j in {0,1}, k in {0,1}
    # Block 2: j in {2,3}, k in {0,1}
    # Block 3: j in {0,1}, k in {2,3}
    # Block 4: j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0), # j_offset, k_offset
        (2, 0),
        (0, 2),
        (2, 2)
    ]
    
    for j_off, k_off in blocks:
        for c in range(7):
            # Map local Strassen indices to global
            # Local U: (i, j_local) -> global (i, j_local + j_off)
            # Local V: (j_local, k_local) -> global (j_local + j_off, k_local + k_off)
            # Local W: (k_local, i) -> global (k_local + k_off, i)
            
            for i in range(2):
                for j_local in range(2):
                    val = U_strassen[i, j_local, c] if len(U_strassen.shape) == 3 else U_strassen[i*2+j_local, c]
                    if val != 0:
                        U_global[i * m_global + (j_local + j_off), col] += val
            
            for j_local in range(2):
                for k_local in range(2):
                    val = V_strassen[j_local, k_local, c] if len(V_strassen.shape) == 3 else V_strassen[j_local*2+k_local, c]
                    if val != 0:
                        V_global[(j_local + j_off) * p_global + (k_local + k_off), col] += val
            
            for k_local in range(2):
                for i in range(2):
                    val = W_strassen[k_local, i, c] if len(W_strassen.shape) == 3 else W_strassen[k_local*2+i, c]
                    if val != 0:
                        W_global[(k_local + k_off) * n_global + i, col] += val
            
            col += 1
    
    # Schoolbook for remaining terms
    # j=4, k=0..4
    for k in range(p_global):
        for i in range(n_global):
            U_global[i * m_global + 4, col] = 1
            V_global[4 * p_global + k, col] = 1
            W_global[k * n_global + i, col] = 1
        col += 1
    
    # j=0..3, k=4
    for j in range(4):
        for i in range(n_global):
            U_global[i * m_global + j, col] = 1
            V_global[j * p_global + 4, col] = 1
            W_global[4 * n_global + i, col] = 1
        col += 1
    
    # Verify
    if _verify(n, m, p, U_global.tolist(), V_global.tolist(), W_global.tolist()):
        return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
    
    return None

def _try_rank_49(n, m, p):
    if time.time() > _DEADLINE - 10:
        return None

    target_rank = 49
    # Dimensions: n=2, m=5, p=5
    # U: (10, 49), V: (25, 49), W: (10, 49)
    
    try:
        import scipy.optimize as opt
        
        best_error = float('inf')
        best_params = None
        
        # Precompute valid indices
        valid_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    valid_indices.append((u_idx, v_idx, w_idx))
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 2:
                break
            
            np.random.seed(1000 + restart)
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                error = 0.0
                for u_idx, v_idx, w_idx in valid_indices:
                    s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    error += (s - 1.0)**2
                return error
            
            def gradient(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for u_idx, v_idx, w_idx in valid_indices:
                    s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    diff = 2 * (s - 1.0)
                    
                    gU[u_idx] += diff * V[v_idx] * W[w_idx]
                    gV[v_idx] += diff * U[u_idx] * W[w_idx]
                    gW[w_idx] += diff * U[u_idx] * V[v_idx]
                
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            try:
                # Use L-BFGS-B
                res = opt.minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
                    
            except Exception:
                pass
        
        if best_error < 1e-6 and best_params is not None:
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try rounding to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
            # Try rounding to nearest integer
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try rounding to nearest quarter
            U_r = np.round(U * 4) / 4
            V_r = np.round(V * 4) / 4
            W_r = np.round(W * 4) / 4
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try Strassen hybrid (rank 37)
    result = _try_strassen_hybrid(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
