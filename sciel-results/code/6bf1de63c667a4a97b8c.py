import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W):
    """
    Verify the decomposition by checking if it computes matrix multiplication
    for a set of random matrices.
    """
    n = m = p = 3
    rank = len(U[0]) if U else 0
    
    # Convert to numpy arrays for efficient multiplication
    U_np = np.array(U, dtype=float).reshape(n, m, rank)
    V_np = np.array(V, dtype=float).reshape(m, p, rank)
    W_np = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Test with random matrices
    for _ in range(10):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_sum = np.dot(U_np[i, :, r], A[i, :])
                    v_sum = np.dot(V_np[:, k, r], B[:, k])
                    val += W_np[k, i, r] * u_sum * v_sum
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True

def _solve_rank_r(R, time_limit=60.0):
    """
    Attempt to find a rank-R decomposition using numerical optimization.
    Returns (U, V, W) if found, else None.
    """
    n = m = p = 3
    # Variables: U (9*R), V (9*R), W (9*R) -> Total 27*R
    total_vars = 27 * R
    
    def unpack(params):
        U_flat = params[:9*R].reshape(9, R)
        V_flat = params[9*R:18*R].reshape(9, R)
        W_flat = params[18*R:].reshape(9, R)
        return U_flat, V_flat, W_flat

    def objective(params):
        U_flat, V_flat, W_flat = unpack(params)
        # U_flat[i*m+j, r] -> index i*m+j
        # V_flat[j*p+k, r] -> index j*p+k
        # W_flat[k*n+i, r] -> index k*n+i
        
        # We need to check constraints:
        # sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
        
        # Efficient calculation using einsum or loops
        # Let's use loops for clarity, 81 constraints
        error = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(p): # j2 is the index for B, which is m*p, so 0..2
                        # Target: 1 if j1==j2 else 0
                        target = 1.0 if j1 == j2 else 0.0
                        # Sum over r
                        s = 0.0
                        for r in range(R):
                            w_val = W_flat[k*n + i, r]
                            u_val = U_flat[i*m + j1, r]
                            v_val = V_flat[j2*p + k, r]
                            s += w_val * u_val * v_val
                        diff = s - target
                        error += diff * diff
        return error

    # Initial guess: Random
    np.random.seed(42)
    x0 = np.random.randn(total_vars) * 0.1
    
    start_time = time.time()
    
    # Try a few different initializations or methods
    best_result = None
    best_error = float('inf')
    
    # Use Nelder-Mead or Powell for small dimensions? 621 dims is large.
    # L-BFGS-B is better if gradients are available, but we don't have analytic gradients easily.
    # We can use finite difference gradients (slow) or just a derivative-free method.
    # Powell is derivative-free but slow in high dimensions.
    # Let's try COBYLA or Nelder-Mead with a tight bound?
    # Actually, for 621 variables, standard scipy minimizers might struggle.
    
    # Alternative: Use a gradient-based method with finite differences.
    # Or, simpler: Use the fact that we can fix one matrix to simplify?
    # No, let's try `minimize` with method='Nelder-Mead' might be too slow.
    # Let's try 'Powell'.
    
    # Better idea: Use a random search / hill climbing with numpy vectorization for the objective.
    
    def objective_vec(params):
        U_flat, V_flat, W_flat = unpack(params)
        # U_flat: (9, R)
        # V_flat: (9, R)
        # W_flat: (9, R)
        
        # We want to compute T_ikj1j2 = sum_r W[kn+i, r] U[im+j1, r] V[j2p+k, r]
        # Let's reshape to 3D tensors
        # U: (i, j, r)
        # V: (j, k, r)
        # W: (k, i, r)
        
        U_3d = U_flat.reshape(3, 3, R)
        V_3d = V_flat.reshape(3, 3, R)
        W_3d = W_flat.reshape(3, 3, R) # k, i, r
        
        # We need to sum over r: W[k,i,r] * U[i,j1,r] * V[j2,k,r]
        # This is a tensor contraction.
        
        # Let's compute the full 4D tensor of coefficients
        # C[i, k, j1, j2] = sum_r W[k, i, r] * U[i, j1, r] * V[j2, k, r]
        
        # einsum: 'kir,ijr,jkr->ikjj' ? No.
        # W[k, i, r]
        # U[i, j1, r]
        # V[j2, k, r]
        
        # We can do:
        # Temp1[i, k, j1, r] = U[i, j1, r] * W[k, i, r] ? No, W depends on k.
        
        # Let's use np.einsum
        # 'kir,ijr,jkr->ikjj' where j is j1 and j is j2?
        # Indices:
        # W: k, i, r
        # U: i, j1, r
        # V: j2, k, r
        
        # Result: i, k, j1, j2
        # C = np.einsum('kir,ijr,jkr->ikjj', W_3d, U_3d, V_3d)
        # Wait, U_3d is (i, j1, r). V_3d is (j2, k, r).
        # 'kir,ijr,jkr->ikjj'
        # k from W, i from W/U, j1 from U, j2 from V, k from V.
        # Yes.
        
        C_tensor = np.einsum('kir,ijr,jkr->ikjj', W_3d, U_3d, V_3d)
        
        # Target: C[i, k, j1, j2] = delta(j1, j2)
        # Target tensor:
        Target = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    Target[i, k, j, j] = 1.0
        
        diff = C_tensor - Target
        return np.sum(diff * diff)

    # Optimize
    # Since R=23, 621 variables.
    # We'll use a few restarts with Nelder-Mead or Powell.
    # Powell is generally robust for derivative-free.
    
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
            
        x0 = np.random.randn(total_vars) * (0.5 if restart == 0 else 1.0)
        
        res = minimize(objective_vec, x0, method='Powell', options={'maxiter': 100, 'disp': False})
        
        if res.fun < best_error:
            best_error = res.fun
            best_result = res.x
            
        if best_error < 1e-8:
            break
            
    if best_result is not None and best_error < 1e-8:
        U_flat, V_flat, W_flat = unpack(best_result)
        # Convert back to list of lists
        U = U_flat.T.tolist() # U is (9, R) -> we need (9, R) rows?
        # Prompt: U is (n*m, rank). So 9 rows, R columns.
        # U_flat is (9, R). So U_flat.T is (R, 9). We want U[i][r].
        # So U = U_flat.tolist() is correct if U_flat is (9, R).
        U = U_flat.tolist()
        V = V_flat.tolist()
        W = W_flat.tolist()
        
        # Verify
        if _verify_decomposition(U, V, W):
            return U, V, W
            
    return None

def _strassen_rank26():
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    # 2x2 Strassen core (7 terms)
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # Border terms (19 terms)
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Try to find a Rank 23 solution
    # If found, return it.
    # Else, return Rank 26.
    
    # We can try Rank 23 first.
    # If it takes too long, we abort and return Rank 26.
    
    U23, V23, W23 = None, None, None
    try:
        U23, V23, W23 = _solve_rank_r(23, time_limit=120.0)
    except Exception as e:
        print(f"Error solving rank 23: {e}")
        
    if U23 is not None:
        return {"U": U23, "V": V23, "W": W23}
        
    # Fallback to Rank 26
    result = _strassen_rank26()
    
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
        
    # Fallback to Schoolbook Rank 27
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
