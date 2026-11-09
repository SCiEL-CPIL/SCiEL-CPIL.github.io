"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_trivial = n * m * p
    triples = [(i, k, j) for i in range(n) for k in range(m) for j in range(p)]
    T = len(triples)
    
    idx_U = np.array([i*m+k for i,k,j in triples])
    idx_V = np.array([k*p+j for i,k,j in triples])
    idx_W = np.array([j*n+i for i,k,j in triples])
    
    def _verify_fast(U, V, W, tol=1e-9):
        r = U.shape[1]
        for t in range(T):
            val = np.dot(U[idx_U[t], :], V[idx_V[t], :] * W[idx_W[t], :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def _is_half_integer(arr, tol=1e-6):
        return bool(np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol))
    
    def _to_output(U, V, W):
        U_out = [[float(U[i,r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
        V_out = [[float(V[i,r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
        W_out = [[float(W[i,r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    best_result = None
    best_rank = rank_trivial
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, n*p)
    
    for rank in range(lower_bound, rank_trivial + 1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 15 if rank <= 60 else 8
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            lr = 0.005
            for it in range(120):
                if time.time() > deadline:
                    break
                
                U_exp = U[idx_U, :]
                V_exp = V[idx_V, :]
                W_exp = W[idx_W, :]
                
                errors = np.sum(U_exp * V_exp * W_exp, axis=1) - 1
                
                grad_U_exp = 2 * errors[:, None] * (V_exp * W_exp)
                grad_V_exp = 2 * errors[:, None] * (U_exp * W_exp)
                grad_W_exp = 2 * errors[:, None] * (U_exp * V_exp)
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                np.add.at(grad_U, idx_U, grad_U_exp)
                np.add.at(grad_V, idx_V, grad_V_exp)
                np.add.at(grad_W, idx_W, grad_W_exp)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 25 == 0:
                    if np.max(np.abs(errors)) < 1e-4:
                        if _is_half_integer(U, 1e-2) and _is_half_integer(V, 1e-2) and _is_half_integer(W, 1e-2):
                            U_r = np.round(U*2)/2
                            V_r = np.round(V*2)/2
                            W_r = np.round(W*2)/2
                            if _verify_fast(U_r, V_r, W_r, 1e-8):
                                best_result = _to_output(U_r, V_r, W_r)
                                best_rank = rank
                                found = True
                                break
                if found:
                    break
            
            if found:
                break
        
        if found:
            break
    
    if best_result:
        return best_result
    
    # Return trivial
    U_out = [[0]*rank_trivial for _ in range(n*m)]
    V_out = [[0]*rank_trivial for _ in range(m*p)]
    W_out = [[0]*rank_trivial for _ in range(p*n)]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i*(m*p) + k*p + j
                U_out[i*m+k][r] = 1
                V_out[k*p+j][r] = 1
                W_out[j*n+i][r] = 1
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
