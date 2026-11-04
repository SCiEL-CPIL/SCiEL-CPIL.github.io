import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build the target tensor T[a,b,c] where:
    # a = i*m + j (input A index)
    # b = j*p + k (input B index)  
    # c = k*n + i (output C index, trace convention)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p  # 45
    
    # Build schoolbook decomposition as fallback
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank
    
    # Lower bound: max of the three mode dimensions
    lower_bound = max(n*m, m*p, p*n)  # max(9, 15, 15) = 15
    
    # Precompute tensor as list of nonzero entries for faster verification
    nz_indices = np.argwhere(T != 0)
    nz_values = T[nz_indices[:,0], nz_indices[:,1], nz_indices[:,2]]
    
    def verify_decomposition(U, V, W, atol=1e-10):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T"""
        # Only check nonzero entries of T
        for idx in nz_indices:
            a, b, c = int(idx[0]), int(idx[1]), int(idx[2])
            val = 0.0
            for r in range(U.shape[1]):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val - 1.0) > atol:
                return False
        # Also check that zero entries remain zero
        # Sample some zero entries
        zero_mask = (T == 0)
        zero_indices = np.argwhere(zero_mask)
        # Check a subset of zero entries
        check_count = min(200, len(zero_indices))
        if check_count > 0:
            sample_idx = np.random.choice(len(zero_indices), check_count, replace=False)
            for si in sample_idx:
                a, b, c = int(zero_indices[si,0]), int(zero_indices[si,1]), int(zero_indices[si,2])
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > atol:
                    return False
        return True
    
    def try_rank(target_rank, max_restarts=8):
        """Try to find a decomposition of given rank."""
        nonlocal best_result, best_rank
        if time.time() > deadline - 15:
            return False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return False
            
            # Different initialization strategies
            if restart == 0:
                # Structured: use first target_rank columns of schoolbook
                U_init = U_sch[:, :target_rank].copy()
                V_init = V_sch[:, :target_rank].copy()
                W_init = W_sch[:, :target_rank].copy()
            elif restart == 1:
                # Random with small scale
                np.random.seed(42 + restart)
                U_init = np.random.randn(n*m, target_rank) * 0.3
                V_init = np.random.randn(m*p, target_rank) * 0.3
                W_init = np.random.randn(p*n, target_rank) * 0.3
            elif restart == 2:
                # Block-structured: try to group by j index
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                # Assign some columns to each j
                cols_per_j = target_rank // m
                for j in range(m):
                    for r in range(cols_per_j):
                        col_idx = j * cols_per_j + r
                        if col_idx >= target_rank:
                            break
                        for i in range(n):
                            U_init[i*m+j, col_idx] = 1.0
                        for k in range(p):
                            V_init[j*p+k, col_idx] = 1.0
                        for i in range(n):
                            for k in range(p):
                                W_init[k*n+i, col_idx] += 1.0 / (n*p)
            else:
                # Random with various scales
                np.random.seed(100 + restart * 7)
                scale = 0.2 + 0.1 * restart
                U_init = np.random.randn(n*m, target_rank) * scale
                V_init = np.random.randn(m*p, target_rank) * scale
                W_init = np.random.randn(p*n, target_rank) * scale
            
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute residual only on nonzero entries for speed
                residual = 0.0
                for idx in nz_indices:
                    a, b, c = int(idx[0]), int(idx[1]), int(idx[2])
                    val = 0.0
                    for r in range(target_rank):
                        val += U_r[a, r] * V_r[b, r] * W_r[c, r]
                    residual += (val - 1.0) ** 2
                
                # Add penalty for zero entries (sampled)
                zero_mask = (T == 0)
                zero_indices = np.argwhere(zero_mask)
                if len(zero_indices) > 0:
                    sample_size = min(100, len(zero_indices))
                    sample_idx = np.random.choice(len(zero_indices), sample_size, replace=False)
                    for si in sample_idx:
                        a, b, c = int(zero_indices[si,0]), int(zero_indices[si,1]), int(zero_indices[si,2])
                        val = 0.0
                        for r in range(target_rank):
                            val += U_r[a, r] * V_r[b, r] * W_r[c, r]
                        residual += val * val
                
                return residual
            
            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-12})
                
                if result.fun < 1e-10:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                        best_rank = target_rank
                        return True
                    else:
                        # Try without rounding - check if already close enough
                        if verify_decomposition(U_r, V_r, W_r):
                            # Need to convert to half-integers; try scaling
                            # Check if values are already close to half-integers
                            max_dev = max(
                                np.max(np.abs(U_r - np.round(U_r * 2) / 2)),
                                np.max(np.abs(V_r - np.round(V_r * 2) / 2)),
                                np.max(np.abs(W_r - np.round(W_r * 2) / 2))
                            )
                            if max_dev < 1e-6:
                                best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                                best_rank = target_rank
                                return True
            except Exception:
                pass
        
        return False
    
    # Search from lower bound up to schoolbook
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        if try_rank(target_rank):
            # Found a decomposition at this rank, continue to try lower
            pass
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
