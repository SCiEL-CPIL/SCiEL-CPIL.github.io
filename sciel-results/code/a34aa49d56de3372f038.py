"""Matrix multiplication 4x4 x 4x6 - rank-minimal decomposition via Strassen-based construction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 x 4x6 matrix mult."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 4, 6
    
    # Strategy: Split B into two 4x4 blocks: B = [B1, B2] where B1 is columns 0-3, B2 is columns 4-5
    # C = A*B = [A*B1, A*B2]
    # We need to compute two 4x4 matrix multiplications: A*B1 and A*B2
    
    # For each 4x4 multiplication, we can use Strassen's algorithm (rank 7 for 2x2, rank 49 for 4x4 via tensor product)
    # Actually, let's use the standard 4x4x4 decomposition with rank 49 (Strassen squared)
    # Or we can use a simpler approach: schoolbook for 4x4x4 has rank 16, so two of them give 32
    
    # Let's use schoolbook for simplicity and correctness, then try to optimize
    # Schoolbook 4x4x4: rank 16
    # Two 4x4x4 multiplications: rank 32
    # But wait, A is the same in both, so we can't just add ranks directly
    # Actually, the tensor decomposition is:
    # T[i,j,k] = 1 for all valid i,j,k
    # For k in 0..3: C[i,k] = sum_j A[i,j]*B[j,k]
    # For k in 4..5: C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The full tensor has 96 entries (4*4*6)
    # Schoolbook: rank = 96 (one term per (i,j,k) triple)
    # But we can do better by sharing the A factors
    
    # Better approach: For each k (0..5), C[:,k] = A * B[:,k]
    # So we need 6 matrix-vector products, each with 16 multiplications
    # Total: 96 multiplications in schoolbook
    
    # Let's construct the schoolbook decomposition first as a baseline
    rank = n * m * p  # 96
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to find a better decomposition using optimization
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try ranks from low to high
    def try_optimization(rank, max_iter=3000, n_restarts=20):
        """Try to find a solution for a given rank using optimization."""
        if time.time() > deadline:
            return None
        
        total_size = rank * (n * m + m * p + p * n)
        best_obj = float('inf')
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + restart)
            params = np.random.randn(total_size) * 0.5
            
            def objective(params, rank=rank):
                um_size = n * m * rank
                vp_size = m * p * rank
                U_r = params[:um_size].reshape(n * m, rank)
                V_r = params[um_size:um_size + vp_size].reshape(m * p, rank)
                W_r = params[um_size + vp_size:].reshape(p * n, rank)
                
                # Compute residuals using vectorized operations
                residuals = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :]
                            residuals += (1.0 - np.sum(prod)) ** 2
                return residuals
            
            from scipy.optimize import minimize
            result = minimize(objective, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        
        if best_params is not None and best_obj < 1e-8:
            um_size = n * m * rank
            vp_size = m * p * rank
            U_r = best_params[:um_size].reshape(n * m, rank)
            V_r = best_params[um_size:um_size + vp_size].reshape(m * p, rank)
            W_r = best_params[um_size + vp_size:].reshape(p * n, rank)
            
            # Snap to half-integers
            U_snap = np.round(U_r * 2) / 2
            V_snap = np.round(V_r * 2) / 2
            W_snap = np.round(W_r * 2) / 2
            
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U_snap[i*m+j, :] * V_snap[j*p+k, :] * W_snap[k*n+i, :]
                        err = abs(1.0 - np.sum(prod))
                        max_err = max(max_err, err)
            
            if max_err < 1e-9:
                return (U_snap, V_snap, W_snap)
            
            # Try integer snapping
            U_int = np.round(U_r)
            V_int = np.round(V_r)
            W_int = np.round(W_r)
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :]
                        err = abs(1.0 - np.sum(prod))
                        max_err = max(max_err, err)
            
            if max_err < 1e-9:
                return (U_int, V_int, W_int)
        
        return None
    
    # Try various ranks
    # Theoretical lower bound is around 96/3 = 32, but realistically higher
    for rank in [32, 40, 48, 56, 64, 72, 80, 88, 96]:
        if time.time() > deadline:
            break
        
        result = try_optimization(rank)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
