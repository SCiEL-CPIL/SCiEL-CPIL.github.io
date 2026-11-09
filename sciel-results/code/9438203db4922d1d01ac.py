import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # Shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Pre-compute indices for U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Indices:
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    # Function to compute error
    def compute_error(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reconstruct tensor
        # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can do this efficiently using einsum or loops
        # r = U.shape[1]
        # U_expanded = U.reshape(n, m, r)
        # V_expanded = V.reshape(m, p, r)
        # W_expanded = W.reshape(p, n, r)
        # T_recon = np.einsum('ijr,jkr,kir->ijk', U_expanded, V_expanded, W_expanded)
        # return np.linalg.norm(T_recon - T)
        
        # More efficient:
        r = U.shape[1]
        # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3rd order tensor contraction.
        
        # Let's use loops for clarity and to avoid huge memory if r is large, 
        # but n,m,p are small.
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Dot product of columns
                    T_recon[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        return np.linalg.norm(T_recon - T)
    
    def verify_solution(U, V, W):
        # Check if entries are integers or half-integers
        for mat in [U, V, W]:
            for val in mat:
                # Check if val is close to integer or half-integer
                if abs(val - round(val)) > 1e-6 and abs(val - (round(val*2)/2.0)) > 1e-6:
                    return False
        # Check exact decomposition
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_recon[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        return np.allclose(T_recon, T, atol=1e-6)

    def snap_to_half_integers(mat):
        return np.round(mat * 2) / 2.0

    # Fallback solution (Rank 22)
    # We can reuse the logic from the current program or hardcode it.
    # To save time, I'll implement a simple rank 22 generator or just return the known one.
    # Actually, I'll just try to find a better one. If fail, I need a valid one.
    # I will construct the rank 22 solution as in the prompt.
    
    def get_rank22_solution():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, coeff in u_coeffs.items():
                U[idx][col] += coeff
            for idx, coeff in v_coeffs.items():
                V[idx][col] += coeff
            for idx, coeff in w_coeffs.items():
                W[idx][col] += coeff
            col += 1

        # ... (Copy the rank 22 construction from the prompt) ...
        # I will paste the construction code here to ensure a valid fallback.
        
        # Block 1
        add_term({0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
        add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
        add_term({0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
        add_term({4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
        add_term({0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
        add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        add_term({1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})
        
        # Block 2
        add_term({0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
        add_term({3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
        add_term({0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
        add_term({4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
        add_term({0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
        add_term({3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
        add_term({1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})
        
        # Block 3
        add_term({2: 1}, {8: 1}, {0: 1})
        add_term({2: 1}, {9: 1}, {2: 1})
        add_term({2: 1}, {10: 1}, {4: 1})
        add_term({2: 1}, {11: 1}, {6: 1})
        add_term({5: 1}, {8: 1}, {1: 1})
        add_term({5: 1}, {9: 1}, {3: 1})
        add_term({5: 1}, {10: 1}, {5: 1})
        add_term({5: 1}, {11: 1}, {7: 1})
        
        return {"U": U, "V": V, "W": W}

    # Search for lower rank
    best_solution = None
    
    for rank in [16, 17, 18, 19, 20, 21]:
        if time.time() > deadline - 20.0:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
                
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Simple gradient descent or Nelder-Mead
            # Nelder-Mead is derivative-free and might handle the non-smoothness better initially
            # But it's slow. Let's try L-BFGS-B with a penalty for non-integer values?
            # Actually, just minimize error, then snap.
            
            # Use scipy.optimize.minimize
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    U_flat = params[:n*m*rank].reshape(n*m, rank)
                    V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    return compute_error(U_flat, V_flat, W_flat)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'disp': False})
                
                if res.success:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    # Snap
                    U_snap = snap_to_half_integers(U_opt)
                    V_snap = snap_to_half_integers(V_opt)
                    W_snap = snap_to_half_integers(W_opt)
                    
                    if verify_solution(U_snap, V_snap, W_snap):
                        best_solution = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        return best_solution
                        
            except Exception as e:
                pass

    # If no better solution found, return rank 22
    return get_rank22_solution()
