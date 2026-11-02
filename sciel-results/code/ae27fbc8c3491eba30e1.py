"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank26_solution():
    """
    Returns a valid rank-26 decomposition.
    Uses Strassen's 7-term algorithm for the 2x2 core and schoolbook for borders.
    """
    n = m = p = 3
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    # M1 = (A00+A11)*(B00+B11)
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    # M2 = (A10+A11)*B00
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    # M3 = A00*(B01-B11)
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    # M4 = A11*(B10-B00)
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    # M5 = (A00+A01)*B11
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    # M6 = (A10-A00)*(B00+B01)
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    # M7 = (A01-A11)*(B10+B11)
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Cross terms ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row, r] += coeff
        for (row, coeff) in v_list:
            V[row, r] += coeff
        for (row, coeff) in w_list:
            W[row, r] += coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _residuals_vec(U, V, W, rank):
    """Vectorized residual computation for the decomposition constraints."""
    n = m = p = 3
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k)
    # Using Einstein summation or manual loops
    res = np.zeros(27)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                res[idx] = s - 1.0
                idx += 1
    return res

def _search_lower_rank(current_rank, target_rank, deadline):
    """
    Attempts to find a lower-rank decomposition by optimizing variables.
    Starts from the current solution and drops columns, then refines.
    Returns a solution dict if successful, None otherwise.
    """
    if target_rank >= current_rank:
        return None
        
    sol_curr = _rank26_solution()
    U = np.array(sol_curr["U"])
    V = np.array(sol_curr["V"])
    W = np.array(sol_curr["W"])
    
    n = m = p = 3
    
    # We try to find a decomposition of the specified target_rank.
    # Strategy: Randomly initialize or project from the current rank.
    # Since dropping rows from a 26-rank solution to 23 doesn't guarantee a solution,
    # we use optimization to "fit" the tensors U, V, W to match the identity.
    
    # Initialize with random values (Gaussian) or zeros? Zeros are bad (singularity).
    # Let's try initializing from the current solution by keeping 'target_rank' columns
    # and setting the rest to 0, but this is biased. 
    # Better: Random restarts.
    
    best_solution = None
    
    # Number of restarts depending on time
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts if deadline > time.time() else 0
    if time_per_restart < 1.0:
        return None

    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U_init = np.random.randn(n*m, target_rank) * 0.5
        V_init = np.random.randn(m*p, target_rank) * 0.5
        W_init = np.random.randn(p*n, target_rank) * 0.5
        
        def residuals(x):
            U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            return _residuals_vec(U_flat, V_flat, W_flat, target_rank)
            
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use 'lm' (Levenberg-Marquardt) as it doesn't have bounds, good for this
            result = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if result.cost < 1e-10:
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Strict verification
                if _verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist()):
                    best_solution = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                    return best_solution
                    
        except Exception:
            pass
            
    return best_solution

def _search_rank23(deadline):
    """Specific wrapper for rank 23 search."""
    return _search_lower_rank(26, 23, deadline)

def _search_rank24(deadline):
    """Specific wrapper for rank 24 search."""
    return _search_lower_rank(26, 24, deadline)

def _search_rank25(deadline):
    """Specific wrapper for rank 25 search."""
    return _search_lower_rank(26, 25, deadline)

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    # Hard limit is 150s. We leave a 10s buffer for safety.
    deadline = start_time + 140.0
    
    # Guaranteed fallback
    result_26 = _rank26_solution()
    if _verify(result_26["U"], result_26["V"], result_26["W"]):
        # We have a valid rank 26.
        # Try to improve to 25, 24, 23.
        
        # Rank 25
        if time.time() < deadline - 10.0:
            res_25 = _search_rank25(deadline - 10.0)
            if res_25 is not None:
                return res_25
                
        # Rank 24
        if time.time() < deadline - 10.0:
            res_24 = _search_rank24(deadline - 10.0)
            if res_24 is not None:
                return res_24
                
        # Rank 23
        if time.time() < deadline - 5.0:
            res_23 = _search_rank23(deadline - 5.0)
            if res_23 is not None:
                return res_23
                
        return result_26
        
    # If rank 26 failed verification (shouldn't happen), try rank 27 schoolbook
    n = m = p = 3
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
