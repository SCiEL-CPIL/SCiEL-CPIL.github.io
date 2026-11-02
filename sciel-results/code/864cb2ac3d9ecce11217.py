"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    target_ranks = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(12345 + target_rank * 1000 + attempt * 7)
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            def loss(params):
                U = params[:n*m*target_rank].reshape(n*m, target_rank)
                V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Test with random matrices
                total_err = 0.0
                for trial in range(10):
                    A = np.random.randn(n, m)
                    B = np.random.randn(m, p)
                    C_true = A @ B
                    
                    # Compute via decomposition
                    # C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k' V[j*p+k',c]*B[j,k'])
                    # Wait, this isn't right. Let me think again.
                    #
                    # The decomposition: C[i,k] = sum_c U_i_c * V_j_c * W_k_c ... no
                    #
                    # Standard: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+l, c] * B[j,l])
                    # This doesn't work because j appears in both sums.
                    #
                    # Correct: The tensor T[a,b,c] with a=i*m+j, b=j'*p+k, c=k'*n+i'
                    # C[c] = sum_{a,b} T[a,b,c] * A[a] * B[b]
                    # = sum_c_idx sum_{a,b} U[a,c_idx]*V[b,c_idx]*W[c,c_idx] * A[a]*B[b]
                    # = sum_c_idx W[c,c_idx] * (sum_a U[a,c_idx]*A[a]) * (sum_b V[b,c_idx]*B[b])
                    #
                    # So: let alpha[c_idx] = sum_a U[a,c_idx]*A[a] = sum_i sum_j U[i*m+j,c_idx]*A[i,j]
                    #     beta[c_idx] = sum_b V[b,c_idx]*B[b] = sum_j sum_k V[j*p+k,c_idx]*B[j,k]
                    # Then C[i,k] where c = k*n+i: C[i,k] = sum_c_idx W[k*n+i, c_idx] * alpha[c_idx] * beta[c_idx]
                    
                    alpha = np.zeros(target_rank)
                    for c_idx in range(target_rank):
                        alpha[c_idx] = np.sum(U[:, c_idx] * A.flatten())
                    
                    beta = np.zeros(target_rank)
                    for c_idx in range(target_rank):
                        beta[c_idx] = np.sum(V[:, c_idx] * B.flatten())
                    
                    C_comp = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            c = k * n + i
                            C_comp[i, k] = np.sum(W[c, :] * alpha * beta)
                    
                    total_err += np.sum((C_comp - C_true) ** 2)
                
                return total_err / 10.0
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            res = minimize(loss, params_init, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e-8:
                U_f = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_f = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_f = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                best_rank = target_rank
                best_U, best_V, best_W = U_f, V_f, W_f
                break
            elif res.fun < 1e-6 and target_rank < best_rank:
                U_f = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_f = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_f = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                best_rank = target_rank
                best_U, best_V, best_W = U_f, V_f, W_f
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
