import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 100.0 # 100s limit to be safe
    
    # Try to find rank 12 decomposition
    result = _find_rank_12(start_time)
    if result is not None:
        return result
        
    # Fallback to rank 24 (schoolbook)
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

def _find_rank_12(start_time):
    n, m, p = 2, 3, 4
    rank = 12
    
    # Strategy: Solve 2x3x2 rank 6 subproblem and duplicate
    # Subproblem: n=2, m=3, p=2, rank=6
    n_sub, m_sub, p_sub = 2, 3, 2
    rank_sub = 6
    
    T_sub = np.zeros((n_sub, m_sub, p_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i, j, k] = 1.0 if i == k else 0.0

    U_sub = None
    V_sub = None
    W_sub = None
    
    for seed in range(50):
        if time.time() - start_time > 80:
            break
        try:
            sol = _solve_subproblem(n_sub, m_sub, p_sub, T_sub, rank_sub, seed)
            if sol is not None:
                U_sub, V_sub, W_sub = sol
                break
        except Exception:
            continue
            
    if U_sub is None:
        return None
        
    # Construct full U, V, W
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    # U is the same for both blocks
    U_full[:, 0:rank_sub] = U_sub
    U_full[:, rank_sub:2*rank_sub] = U_sub
    
    # V: first half of k indices (0,1) use first block, second half (2,3) use second block
    # V_sub has shape (m_sub, p_sub, rank_sub) = (3, 2, 6)
    # V_full has shape (m, p, rank) = (3, 4, 12)
    
    # Block 1: k=0,1
    V_full[:, 0:p_sub, 0:rank_sub] = V_sub
    # Block 2: k=2,3
    V_full[:, p_sub:p_sub+p_sub, rank_sub:2*rank_sub] = V_sub
    
    # W: first half of k indices (0,1) use first block, second half (2,3) use second block
    # W_sub has shape (p_sub, n_sub, rank_sub) = (2, 2, 6)
    # W_full has shape (p, n, rank) = (4, 2, 12)
    
    # Block 1: k=0,1
    W_full[0:p_sub, :, 0:rank_sub] = W_sub
    # Block 2: k=2,3
    W_full[p_sub:p_sub+p_sub, :, rank_sub:2*rank_sub] = W_sub
    
    # Verify full decomposition
    if _verify_full(n, m, p, U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
        
    return None

def _solve_subproblem(n, m, p, T, rank, seed):
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T_comp = np.einsum('ijc,jkc,klc->ijkl', U3, V3, W3)
        
        diff = T_comp - T
        return np.sum(diff**2)
        
    res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200})
    
    if res.fun < 1e-8:
        U = res.x[:n*m*rank].reshape(n*m, rank)
        V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify exactness with rounded values
        U3 = U_rounded.reshape(n, m, rank)
        V3 = V_rounded.reshape(m, p, rank)
        W3 = W_rounded.reshape(p, n, rank)
        T_comp = np.einsum('ijc,jkc,klc->ijkl', U3, V3, W3)
        
        if np.allclose(T_comp, T, atol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
    return None

def _verify_full(n, m, p, U, V, W):
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T_comp = np.einsum('ijc,jkc,klc->ijkl', U3, V3, W3)
    
    T = np.zeros((n, m, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, i] = 1.0
                
    return np.allclose(T_comp, T, atol=1e-9)
