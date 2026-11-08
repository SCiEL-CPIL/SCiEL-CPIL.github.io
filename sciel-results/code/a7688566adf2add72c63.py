import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10

    # Precompute valid indices and target values
    # Target is 1 for all valid (i, j, k)
    # Map (i, j, k) to (u_idx, v_idx, w_idx)
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                u_indices.append(u_idx)
                v_indices.append(v_idx)
                w_indices.append(w_idx)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        terms = U[u_indices] * V[v_indices] * W[w_indices]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=tol)
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_solution = get_schoolbook()
    best_rank = n * m * p

    # Known lower bound for R(2,3,5)
    # The tensor rank must be at least max(n*m, m*p, p*n) = max(6, 15, 10) = 15?
    # Actually for matrix multiplication, rank >= max(n*m, m*p, p*n) is NOT true in general.
    # But we can use Strassen-like bounds. Let's just search from a reasonable lower bound.
    
    # For 2x3x5, the schoolbook rank is 30.
    # We know that R(2,2,2) = 7 (Strassen).
    # For 2x3x5, we can try to use Strassen blocks or other known decompositions.
    # Let's start searching from rank 10 upwards, but we know it must be at least 
    # the maximum of the dimension products in some sense.
    # Actually, a simple lower bound is: rank >= max(n, m, p) * min(n, m, p) is not right.
    # Let's just use: the rank must be sufficient to express all 30 independent entries.
    # But they're not all independent in the tensor sense.
    
    # Start from a lower bound. For 2x3x5, it's known that rank <= 24 (via Strassen-like).
    # Let's try to find something better than schoolbook (30).
    
    # Try ranks from 12 to best_rank-1
    # We'll use a more efficient search strategy
    
    for R in range(12, best_rank):
        if time.time() > deadline:
            break
            
        found_solution = None
        
        # Use multiple random restarts with different strategies
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            np.random.seed(42 + R * 1000 + restart)
            
            if restart % 3 == 0:
                # Random normal
                U_init = np.random.randn(N_U, R)
                V_init = np.random.randn(N_V, R)
                W_init = np.random.randn(N_W, R)
            elif restart % 3 == 1:
                # Sparse-ish initialization
                U_init = np.random.randn(N_U, R)
                U_init[np.abs(U_init) < 0.3] = 0
                V_init = np.random.randn(N_V, R)
                V_init[np.abs(V_init) < 0.3] = 0
                W_init = np.random.randn(N_W, R)
                W_init[np.abs(W_init) < 0.3] = 0
            else:
                # Binary-ish initialization
                U_init = np.sign(np.random.randn(N_U, R))
                V_init = np.sign(np.random.randn(N_V, R))
                W_init = np.sign(np.random.randn(N_W, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(x):
                U = x[:N_U * R].reshape(N_U, R)
                V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W = x[N_U * R + N_V * R :].reshape(N_W, R)
                
                terms = U[u_indices] * V[v_indices] * W[w_indices]
                sums = terms.sum(axis=1)
                
                errors = sums - 1.0
                return np.sum(errors ** 2)
            
            try:
                # Use L-BFGS-B for better performance
                res = minimize(loss, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'gtol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Potential solution found, try to round to half-integers
                x_opt = res.x
                U_opt = x_opt[:N_U * R].reshape(N_U, R)
                V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Try rounding to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify_solution(U_round, V_round, W_round):
                    found_solution = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break
                else:
                    # Try integer rounding
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    
                    if verify_solution(U_int, V_int, W_int):
                        found_solution = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        break
                    else:
                        # Try quarter-integer rounding
                        U_q = np.round(U_opt * 4) / 4
                        V_q = np.round(V_opt * 4) / 4
                        W_q = np.round(W_opt * 4) / 4
                        
                        if verify_solution(U_q, V_q, W_q):
                            found_solution = {
                                "U": U_q.tolist(),
                                "V": V_q.tolist(),
                                "W": W_q.tolist()
                            }
                            break
        
        if found_solution:
            best_solution = found_solution
            best_rank = R
            # Since we iterate R increasing, return immediately
            return best_solution
            
    return best_solution
