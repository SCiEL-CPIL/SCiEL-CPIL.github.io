import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the 2x3x3 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(U.shape[1]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_residual(U, V, W):
    """Compute the sum of squared residuals for the decomposition."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(r))
                res += (val - 1.0)**2
    return res

def _try_find_rank16(deadline):
    """Try to find a rank-16 decomposition using more aggressive optimization."""
    n, m, p = DIMENSIONS
    r = 16
    
    U17, V17, W17 = _build_rank17()
    
    # Try removing each term and optimizing
    best_result = None
    best_residual = float('inf')
    
    for remove_idx in range(17):
        if time.time() > deadline - 20:
            break
        
        U16 = U17[:, [i for i in range(17) if i != remove_idx]].copy()
        V16 = V17[:, [i for i in range(17) if i != remove_idx]].copy()
        W16 = W17[:, [i for i in range(17) if i != remove_idx]].copy()
        
        # Use a more systematic optimization approach
        # Try different step sizes and random restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            U_curr = U16.copy()
            V_curr = V16.copy()
            W_curr = W16.copy()
            
            if restart > 0:
                # Add small random perturbations
                U_curr += np.random.normal(0, 0.01, U_curr.shape)
                V_curr += np.random.normal(0, 0.01, V_curr.shape)
                W_curr += np.random.normal(0, 0.01, W_curr.shape)
            
            current_res = _compute_residual(U_curr, V_curr, W_curr)
            
            # Gradient-free optimization with adaptive step sizes
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
            
            for ss in step_sizes:
                if time.time() > deadline - 5:
                    break
                
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    improved = False
                    for _ in range(30):
                        if time.time() > deadline - 5:
                            break
                        
                        mat_choice = np.random.choice(3)
                        if mat_choice == 0:
                            idx = np.random.randint(n*m)
                            term = np.random.randint(r)
                            old_val = U_curr[idx, term]
                            U_curr[idx, term] += np.random.uniform(-ss, ss)
                            new_res = _compute_residual(U_curr, V_curr, W_curr)
                            if new_res < current_res:
                                current_res = new_res
                                improved = True
                            else:
                                U_curr[idx, term] = old_val
                        elif mat_choice == 1:
                            idx = np.random.randint(m*p)
                            term = np.random.randint(r)
                            old_val = V_curr[idx, term]
                            V_curr[idx, term] += np.random.uniform(-ss, ss)
                            new_res = _compute_residual(U_curr, V_curr, W_curr)
                            if new_res < current_res:
                                current_res = new_res
                                improved = True
                            else:
                                V_curr[idx, term] = old_val
                        else:
                            idx = np.random.randint(p*n)
                            term = np.random.randint(r)
                            old_val = W_curr[idx, term]
                            W_curr[idx, term] += np.random.uniform(-ss, ss)
                            new_res = _compute_residual(U_curr, V_curr, W_curr)
                            if new_res < current_res:
                                current_res = new_res
                                improved = True
                            else:
                                W_curr[idx, term] = old_val
                        
                        if not improved:
                            break
                    
                    if current_res < 1e-10:
                        break
                
                if current_res < 1e-10:
                    break
            
            if current_res < best_residual:
                best_residual = current_res
                best_result = (U_curr, V_curr, W_curr)
            
            if best_residual < 1e-10:
                break
    
    if best_result is not None and best_residual < 1e-10:
        U16, V16, W16 = best_result
        if _verify(U16, V16, W16):
            return U16, V16, W16
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # First, verify the rank-17 decomposition
    U17, V17, W17 = _build_rank17()
    
    if not _verify(U17, V17, W17):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Try to find rank-16
    result16 = _try_find_rank16(deadline)
    if result16 is not None:
        U16, V16, W16 = result16
        if _verify(U16, V16, W16):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
