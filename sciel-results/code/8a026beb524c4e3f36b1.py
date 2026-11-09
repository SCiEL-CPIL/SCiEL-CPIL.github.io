import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # Build the target tensor T of shape (n*m, m*p, p*n) ?
    # No, the tensor is T[i, j, k] = 1 if i,j,k correspond to A[i,j] * B[j,k] -> C[i,k]
    # Indices:
    # U index: i*m + j (0 to n*m-1)
    # V index: j*p + k (0 to m*p-1)
    # W index: k*n + i (0 to p*n-1)
    
    # The tensor T is defined such that T[u_idx, v_idx, w_idx] = 1 if they match.
    # Actually, the decomposition is sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
    # T[u,v,w] is 1 if u=(i,j), v=(j,k), w=(k,i) for some i,j,k.
    # Note: u determines i,j. v determines j,k. w determines k,i.
    # So for a valid triple (i,j,k), u=i*m+j, v=j*p+k, w=k*n+i.
    # T[u,v,w] = 1. All other entries 0.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Current best rank is 44.
    # We try to find a rank 43 decomposition.
    
    def try_rank(rank):
        # Random initialization
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We use a sparse initialization based on the known 44 rank solution, 
        # but we need to generate a 43 rank one.
        # Let's just use random dense initialization for the search.
        
        best_solution = None
        best_resid = np.inf
        
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to prevent overflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_curr = x[:n*m*rank].reshape(n*m, rank)
                V_curr = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_curr = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute tensor sum
                # T_calc[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # This is a tensor product.
                # We can compute this efficiently using einsum or loops.
                # T_calc = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                # But T is sparse. We only need to match the non-zero entries.
                # Non-zero entries are at (i*m+j, j*p+k, k*n+i)
                
                # Let's compute the full tensor and compare.
                T_calc = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                
                # Residual is the difference at the non-zero positions
                # And we also want the zero positions to be zero?
                # The tensor T has zeros everywhere else.
                # So we minimize || T_calc - T ||_F
                diff = T_calc - T
                return diff.flatten()
            
            try:
                result = least_squares(objective, x0, method='lm', max_nfev=1000)
                resid = np.linalg.norm(result.fun)
                
                if resid < 1e-8:
                    # Found a solution
                    U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Check if we can round to integers/half-integers
                    # Scale the solution
                    # Find the GCD of the coefficients?
                    # Try scaling by powers of 2
                    for scale in [1, 2, 4, 8, 16, 32, 64, 128]:
                        U_s = U_sol * scale
                        V_s = V_sol * scale
                        W_s = W_sol * scale
                        
                        U_r = np.round(U_s * 2) / 2
                        V_r = np.round(V_s * 2) / 2
                        W_r = np.round(W_s * 2) / 2
                        
                        # Verify exactness
                        T_calc_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                        if np.allclose(T_calc_r, T, atol=1e-9):
                            # Check if entries are actually half-integers
                            if np.allclose(U_r * 2, np.round(U_r * 2)) and \
                               np.allclose(V_r * 2, np.round(V_r * 2)) and \
                               np.allclose(W_r * 2, np.round(W_r * 2)):
                                return U_r, V_r, W_r
            except Exception:
                pass
                
        return None

    U43, V43, W43 = try_rank(43)
    
    if U43 is not None:
        return {"U": U43.tolist(), "V": V43.tolist(), "W": W43.tolist()}
        
    # Fallback to rank 44
    # Re-construct the rank 44 solution from the current program
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    schoolbook_terms_122 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1}, {(1,0): 1}),
        ({(0,1): 1}, {(1,1): 1}, {(1,0): 1})
    ]
    
    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    rank_col = 0
    
    def add_block(i_off, j_off, k_off, terms):
        nonlocal rank_col
        for term in terms:
            u_dict, v_dict, w_dict = term
            
            for (li, lj), coeff in u_dict.items():
                idx_u = (i_off + li) * m + (j_off + lj)
                U[idx_u, rank_col] += coeff
                
            for (lj, lk), coeff in v_dict.items():
                idx_v = (j_off + lj) * p + (k_off + lk)
                V[idx_v, rank_col] += coeff
                
            for (lk, li), coeff in w_dict.items():
                idx_w = (k_off + lk) * n + (i_off + li)
                W[idx_w, rank_col] += coeff
                
            rank_col += 1
            
    add_block(0, 0, 0, strassen_terms)
    add_block(0, 2, 0, strassen_terms)
    add_block(0, 0, 2, strassen_terms)
    add_block(0, 2, 2, strassen_terms)
    
    add_block(2, 0, 0, schoolbook_terms_122)
    add_block(2, 2, 0, schoolbook_terms_122)
    add_block(2, 0, 2, schoolbook_terms_122)
    add_block(2, 2, 2, schoolbook_terms_122)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
