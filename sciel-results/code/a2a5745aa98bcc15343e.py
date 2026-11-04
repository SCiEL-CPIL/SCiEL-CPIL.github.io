"""Matrix multiplication 3,4,8 tensor decomposition - combined approach."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Combines structured search with gradient-based refinement.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    def schoolbook():
        """Return the standard rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build true tensor
        T_true = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m + j, j*p + k, k*n + i] = 1.0
        
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_recon, T_true, atol=tol):
            return False
        return True
    
    def try_snap_and_verify(U, V, W, scales=[1, 2, 4]):
        """Try to snap to integer/half-integer and verify."""
        for scale in scales:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r, V_r, W_r
        return None
    
    # Check time - if running low, return schoolbook
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Build true tensor once
    T_true = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Lower bound: slice rank arguments
    # Fixing j: need at least n*p = 24
    # Fixing i: need at least m*p = 32  
    # Fixing k: need at least n*m = 12
    # So lower bound is max(24, 32, 12) = 32
    lb = max(n*p, m*p, n*m)  # 32
    
    # Strategy 1: Try structured decomposition based on convolutional/FFT ideas
    # For 3x4 * 4x8, we can think of the inner dimension m=4 as a kernel
    # and try to exploit structure in the p=8 dimension
    
    # Try ranks from lower bound up, with gradient descent
    candidate_ranks = []
    for r in range(lb, best_rank + 1, 2):
        candidate_ranks.append(r)
    if best_rank not in candidate_ranks:
        candidate_ranks.append(best_rank)
    
    found = False
    best_U = None
    best_V = None
    best_W = None
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        if r >= best_rank:
            break
        
        # Try multiple initializations and seeds
        for seed in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
            
            rng = np.random.RandomState(seed * 1000 + r)
            
            # Initialize with small random values
            U = rng.randn(nm, r) * 0.5
            V = rng.randn(mp, r) * 0.5
            W = rng.randn(pn, r) * 0.5
            
            # Also try structured initialization based on schoolbook patterns
            # but with some mixing to allow for rank reduction
            if seed == 0:
                # Partial schoolbook structure with some columns shared
                U_init = np.zeros((nm, r))
                V_init = np.zeros((mp, r))
                W_init = np.zeros((pn, r))
                
                # Fill first min(r, n*m*p) columns with schoolbook pattern
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U_init[i*m + j, col] = 1
                                V_init[j*p + k, col] = 1
                                W_init[k*n + i, col] = 1
                                col += 1
                            else:
                                break
                        if col >= r:
                            break
                    if col >= r:
                        break
                
                # Add small random perturbation to break symmetry
                U_init += rng.randn(nm, r) * 0.01
                V_init += rng.randn(mp, r) * 0.01
                W_init += rng.randn(pn, r) * 0.01
                
                U = U_init
                V = V_init
                W = W_init
            
            # Gradient descent with adaptive learning rate
            eta = 0.01
            best_err = np.inf
            patience = 0
            
            for it in range(3000):
                if time.time() - START_TIME > TIME_LIMIT - 20:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T_true - T_recon
                err = np.max(np.abs(R))
                
                if err < best_err:
                    best_err = err
                    patience = 0
                    if err < 1e-8:
                        # Try to snap
                        result = try_snap_and_verify(U, V, W)
                        if result is not None:
                            best_U, best_V, best_W = result
                            best_rank = r
                            found = True
                            break
                else:
                    patience += 1
                    if patience > 500:
                        break
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = np.einsum('abc,ar,br->cr', R, U, V)
                
                # Adaptive learning rate
                if err < 1e-4:
                    eta *= 0.5
                elif err < 1e-2:
                    eta *= 0.8
                
                U += eta * grad_U
                V += eta * grad_V
                W += eta * grad_W
                
                # Occasional snap attempt
                if it % 200 == 0 and err < 1e-6:
                    result = try_snap_and_verify(U, V, W)
                    if result is not None:
                        best_U, best_V, best_W = result
                        best_rank = r
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    if found:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        if verify(U_final, V_final, W_final):
            return {"U": U_final, "V": V_final, "W": W_final}
    
    return best_result
# EVOLVE-BLOCK-END
