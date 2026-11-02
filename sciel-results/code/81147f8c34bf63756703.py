import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 6, 6, 6
    # Target tensor T
    # T[u, v, w] = 1 if u=i*m+k, v=k*p+j, w=j*n+i
    # Else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Known valid decomposition rank 189
    # We will try to beat this
    
    deadline = time.time() + 110.0
    
    # Try to find a decomposition with rank < 189
    # We'll try ranks 188, 187, etc.
    
    def evaluate_decomposition(U, V, W):
        # Returns error norm
        # T_recon = sum_r U[:, r] V[:, r] W[:, r]
        # Efficient computation:
        # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Using einsum: 'ir, jr, kr -> ijk'
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(T_recon - T)

    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        # Initialize U, V, W
        # Strategy: Random initialization or perturbation of known solution
        # Since we don't have the 189 solution vectors easily accessible as a list 
        # (we'd have to reconstruct them), we can just use random init.
        # But random init for rank 188 is unlikely to converge to an integer solution.
        
        # Better: Use the structure of the problem.
        # However, without the 189 coefficients, we can't easily perturb.
        
        # Let's try a simple random search with optimization.
        # This is a shot in the dark, but it's the only way to "search".
        
        # To increase chances, we can use multiple random restarts.
        # But 120s is short.
        
        # Let's try 1 restart for rank 188.
        
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Flatten parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            if time.time() > deadline:
                return float('inf')
            U_f = x[:n*m*rank].reshape(n*m, rank)
            V_f = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_f = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            T_recon = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
            err = T_recon - T
            return np.sum(err * err)
        
        # Use L-BFGS-B or similar
        # This might be too slow for 20k vars.
        # Let's try.
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
        except Exception:
            return None
            
        if time.time() > deadline:
            return None
            
        x_opt = res.x
        U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
        V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = x_opt[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        # Snap to half-integers
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
        # Check exactness
        T_recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
        if np.allclose(T_recon, T, atol=1e-9):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
        return None

    # Try rank 188
    result = try_rank(188)
    if result:
        return result
        
    # Fallback to rank 189 (Block Strassen)
    # Re-implement the known 189 solution
    
    rank = 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                    r_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
