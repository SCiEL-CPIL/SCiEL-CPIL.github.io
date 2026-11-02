import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    def residual_fn(params, R):
        size_U = n * m * R
        size_V = m * p * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R)).T  # (R, n*m)
        V = V_flat.reshape((m * p, R)).T  # (R, m*p)
        W = W_flat.reshape((p * n, R)).T  # (R, p*n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        # Vectorized reconstruction using einsum
        # U: (R, n, m), V: (R, m, p), W: (R, p, n)
        # We need: sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
        # Output shape: (n, m, m, p, n, p)
        recon = np.einsum('rni,rnj,rmk->injm', U_reshaped, V_reshaped, W_reshaped)
        # Wait, that's not right. Let me think again.
        # U_reshaped: (R, n, m) -> indices r, i, j
        # V_reshaped: (R, m, p) -> indices r, j, k  
        # W_reshaped: (R, p, n) -> indices r, k, i
        # We want T[i, j, j2, k, i2, k2] where j==j2, i==i2, k==k2
        # So T[i, j, j, k, i, k] = sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
        
        # Using einsum: 'rij,rjk,rki->ijjik' doesn't work directly because we need to match indices
        # Let's use: U_reshaped is (R,n,m), V_reshaped is (R,m,p), W_reshaped is (R,p,n)
        # We want output (n,m,m,p,n,p)
        # T[i,j,j2,k,i2,k2] = sum_r U[r,i,j] * V[r,j2,k] * W[r,k2,i2]
        # But we need j==j2 and i==i2 and k==k2 for the target
        
        # Actually the target is: T[i,j,j,k,i,k] = 1
        # And our reconstruction should be: sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
        # This gives a tensor of shape (n,m,p) not (n,m,m,p,n,p)
        
        # I need to be more careful. The tensor T has 6 indices.
        # T[i, j, j2, k, i2, k2]
        # The matrix multiplication tensor is defined such that:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is: T[i,j,j2,k,i2,k2] = delta_{j,j2} * delta_{i,i2} * delta_{k,k2}
        # This means T[i,j,j,k,i,k] = 1 for all valid i,j,k
        
        # The decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
        # Where U_r is (n,m), V_r is (m,p), W_r is (p,n)
        # T[i,j,j2,k,i2,k2] = sum_r U[r,i,j] * V[r,j2,k] * W[r,i2,k2]
        # Wait, that's not quite right either.
        
        # Standard matrix multiplication tensor:
        # T_{i,j,k} = sum_j A_{i,j} B_{j,k}
        # The full tensor in 6 indices: T[i,j,j',k,i',k'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
        
        # The rank decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r ∈ R^{n×m}, v_r ∈ R^{m×p}, w_r ∈ R^{p×n}
        # T[i,j,j',k,i',k'] = sum_r u_r[i,j] * v_r[j',k] * w_r[i',k']
        
        # So: recon[i,j,j',k,i',k'] = sum_r U[r,i,j] * V[r,j',k] * W[r,i',k']
        
        # Using einsum:
        # U: (R,n,m) -> r,i,j
        # V: (R,m,p) -> r,j',k
        # W: (R,p,n) -> r,i',k'  but W is (p,n) so indices are k',i' -> we need W[r,k',i']
        # W_reshaped is (R,p,n) = (R,k',i')
        
        recon = np.einsum('rij,rjk,rki->ijjik', U_reshaped, V_reshaped, W_reshaped)
        # Output: (n, m, m, p, n, p)
        # i from U, j from U, j from V, k from V, i from W, k from W
        # This matches T[i,j,j,k,i,k]
        
        return (recon.flatten() - T_flat)
    
    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact"""
        R = U.shape[1]
        U_reshaped = U.reshape(R, n, m)
        V_reshaped = V.reshape(R, m, p)
        W_reshaped = W.reshape(R, p, n)
        
        recon = np.einsum('rij,rjk,rki->ijjik', U_reshaped, V_reshaped, W_reshaped)
        return np.max(np.abs(recon - T)) < tol
    
    def snap_to_half_integers(arr, tol=1e-6):
        """Try to snap values to half-integers"""
        snapped = np.round(arr * 2) / 2
        return snapped
    
    def try_optimize(R, n_restarts=3, warm_start=None):
        """Try to find a solution of rank R"""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        best_result = None
        best_cost = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            if warm_start is not None and restart == 0:
                x0 = warm_start
            else:
                # Use a structured initialization
                x0 = np.random.randn(total_size) * 0.3
            
            try:
                res = least_squares(
                    residual_fn, x0, args=(R,),
                    method='lm', max_nfev=300,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res
                    
                    if res.cost < 1e-10:
                        break
                        
            except Exception as e:
                continue
        
        if best_result is None:
            return None
        
        if best_result.cost < 1e-10:
            U_flat = best_result.x[:size_U]
            V_flat = best_result.x[size_U:size_U + size_V]
            W_flat = best_result.x[size_U + size_V:]
            
            U = U_flat.reshape((n * m, R))
            V = V_flat.reshape((m * p, R))
            W = W_flat.reshape((p * n, R))
            
            # Try snapping to half-integers
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)
            
            if verify_solution(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
            
            # If snapping doesn't work, check original
            if verify_solution(U, V, W):
                return U, V, W
        
        return None
    
    def get_schoolbook_solution():
        """Return the standard schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    # Try to find the minimum rank
    best_solution = None
    best_rank = n * m * p
    
    # The rank must be at least max(n*p, m*p, n*m) for non-trivial cases
    # For 3x4x7: max(21, 28, 12) = 28
    # But we can do better with algebraic decompositions
    # Start from a reasonable lower bound
    
    lower_bound = max(n * p, m * p, n * m)  # 28
    # Add some margin
    start_rank = max(40, lower_bound + 5)
    
    # Try ranks from start_rank up to schoolbook
    for R in range(start_rank, n * m * p + 1):
        if time.time() > deadline:
            break
            
        result = try_optimize(R, n_restarts=3)
        
        if result is not None:
            U, V, W = result
            best_rank = R
            best_solution = (U, V, W)
            break  # We found the minimum rank
    
    # If no better solution found, use schoolbook
    if best_solution is None:
        U, V, W = get_schoolbook_solution()
    else:
        U, V, W = best_solution
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
