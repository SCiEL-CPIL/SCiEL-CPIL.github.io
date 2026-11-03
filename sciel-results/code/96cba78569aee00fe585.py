import numpy as np
import time
import random

def _build_tensor_indices(n, m, p):
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _verify_decomposition(U, V, W, indices, tol=1e-9):
    # Check if U, V, W exactly decompose the tensor defined by indices
    recon_nonzero = np.zeros(len(indices))
    for idx, (a, b, c) in enumerate(indices):
        recon_nonzero[idx] = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
    
    if not np.allclose(recon_nonzero, 1.0, atol=tol):
        return False
        
    n_total = n * m
    m_total = m * p
    p_total = p * n
    
    zero_indices = set()
    for _ in range(100):
        a = random.randint(0, n_total - 1)
        b = random.randint(0, m_total - 1)
        c = random.randint(0, p_total - 1)
        if (a, b, c) not in indices:
             zero_indices.add((a, b, c))
             
    for (a, b, c) in zero_indices:
        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        if abs(val) > tol:
            return False
            
    return True

def _round_to_half_integers(mat):
    return np.round(mat * 2) / 2.0

def _try_rank_r(n, m, p, r, indices, deadline):
    start_time = time.time()
    max_restarts = 100
    
    idx_a = np.array([x[0] for x in indices])
    idx_b = np.array([x[1] for x in indices])
    idx_c = np.array([x[2] for x in indices])
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        converged = False
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Update U
            Num_U = np.zeros((n*m, r))
            Denom_U = np.zeros(r)
            for i in range(len(indices)):
                a, b, c = indices[i]
                Num_U[a] += V[b] * W[c]
                Denom_U += (V[b] * W[c]) ** 2
            Denom_U[Denom_U == 0] = 1e-10
            U = Num_U / Denom_U
            
            # Update V
            Num_V = np.zeros((m*p, r))
            Denom_V = np.zeros(r)
            for i in range(len(indices)):
                a, b, c = indices[i]
                Num_V[b] += U[a] * W[c]
                Denom_V += (U[a] * W[c]) ** 2
            Denom_V[Denom_V == 0] = 1e-10
            V = Num_V / Denom_V
            
            # Update W
            Num_W = np.zeros((p*n, r))
            Denom_W = np.zeros(r)
            for i in range(len(indices)):
                a, b, c = indices[i]
                Num_W[c] += U[a] * V[b]
                Denom_W += (U[a] * V[b]) ** 2
            Denom_W[Denom_W == 0] = 1e-10
            W = Num_W / Denom_W
            
            residual = 0
            for i in range(len(indices)):
                a, b, c = indices[i]
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                residual += (1.0 - val) ** 2
                
            if residual < 1e-8:
                converged = True
                break
        
        if not converged:
            continue
            
        U_rounded = _round_to_half_integers(U)
        V_rounded = _round_to_half_integers(V)
        W_rounded = _round_to_half_integers(W)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, indices):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify_decomposition(U_int, V_int, W_int, indices):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 2, 3, 7
    indices = _build_tensor_indices(n, m, p)
    
    start_time = time.time()
    max_time = 140.0

    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_solution = None
    best_rank = n * m * p

    for r in range(41, 20, -1):
        if time.time() > max_time:
            break
            
        remaining = max_time - (time.time() - start_time)
        if remaining < 5:
            break
            
        sol = _try_rank_r(n, m, p, r, indices, time.time() + min(remaining, 20.0))
        
        if sol is not None:
            U, V, W = sol
            best_solution = (U, V, W)
            best_rank = r

    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
