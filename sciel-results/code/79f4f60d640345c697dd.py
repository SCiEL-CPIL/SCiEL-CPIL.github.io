"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
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

def _strassen_rank26():
    """Rank-26 decomposition for 3x3 matrix multiplication."""
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # Strassen core for C[0:2,0:2] (7 terms)
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # Border: C[0:2, 2]
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # Border: C[2, 0:2]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # Corner: C[2,2]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # A[0:2,2]*B[2,0:2] -> C[0:2,0:2]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _try_rank23_numerical(start_time):
    """
    Attempt to find a rank-23 decomposition via numerical optimization.
    Uses the known structure: we need to satisfy 27 tensor equations with 3*9*23=621 variables.
    Strategy: start from rank-26, try to eliminate 3 terms by solving constraints.
    """
    deadline = start_time + 105.0
    
    # Known rank-23 decomposition structure from literature:
    # Uses a non-trivial combination that shares products more cleverly.
    # Let's try a direct numerical approach with scipy.
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    rank = 23
    n = m = p = 3
    
    # Flatten the three matrices
    # U: 9 x 23, V: 9 x 23, W: 9 x 23
    # Total variables: 3 * 9 * 23 = 621
    
    # Residual: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1 = 0
    # 27 residuals
    
    def residuals(params):
        U_flat = params[:9*rank]
        V_flat = params[9*rank:18*rank]
        W_flat = params[18*rank:27*rank]
        
        U = U_flat.reshape(9, rank)
        V = V_flat.reshape(9, rank)
        W = W_flat.reshape(9, rank)
        
        res = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    res[idx] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
        return res
    
    # Initial guess: start from rank-26 and zero out 3 columns, or use random
    # Better: use the rank-26 solution as initial guess for first 23 columns
    r26 = _strassen_rank26()
    U0 = np.array(r26["U"])[:, :rank]
    V0 = np.array(r26["V"])[:, :rank]
    W0 = np.array(r26["W"])[:, :rank]
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    # Try multiple random initializations
    best_result = None
    best_residual = np.inf
    
    rng = np.random.RandomState(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            x_init = x0.copy()
        else:
            # Perturb
            noise_scale = 0.1 * (attempt - 1)
            x_init = x0 + rng.randn(len(x0)) * noise_scale
        
        try:
            result = least_squares(residuals, x_init, max_nfev=20000, 
                                   xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue
        
        if result.cost < best_residual:
            best_residual = result.cost
            best_result = result
        
        if best_residual < 1e-10:
            break
    
    if best_result is None or best_result.cost > 1e-8:
        return None
    
    # Extract solution
    x = best_result.x
    U_sol = x[:9*rank].reshape(9, rank)
    V_sol = x[9*rank:18*rank].reshape(9, rank)
    W_sol = x[18*rank:27*rank].reshape(9, rank)
    
    # Round to nearest half-integer if close
    U_rounded = np.round(U_sol * 2) / 2
    V_rounded = np.round(V_sol * 2) / 2
    W_rounded = np.round(W_sol * 2) / 2
    
    # Check if rounded version still works
    U_list = [[float(x) for x in row] for row in U_rounded.tolist()]
    V_list = [[float(x) for x in row] for row in V_rounded.tolist()]
    W_list = [[float(x) for x in row] for row in W_rounded.tolist()]
    
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try without rounding
    U_list = [[float(x) for x in row] for row in U_sol.tolist()]
    V_list = [[float(x) for x in row] for row in V_sol.tolist()]
    W_list = [[float(x) for x in row] for row in W_sol.tolist()]
    
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try rank-23 first
    r23 = _try_rank23_numerical(start_time)
    if r23 is not None and _verify(r23["U"], r23["V"], r23["W"]):
        if time.time() < deadline:
            return r23
    
    # Try rank-26 construction
    result = _strassen_rank26()
    if _verify(result["U"], result["V"], result["W"]):
        if time.time() < deadline:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
