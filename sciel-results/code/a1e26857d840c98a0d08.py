import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Shape: (n, m, m, p, n, p)
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    def residual_fn(params, R):
        """Compute residual of tensor decomposition."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        # Reshape to (R, n, m), (R, m, p), (R, p, n)
        U_3d = U_flat.reshape((R, n, m))
        V_3d = V_flat.reshape((R, m, p))
        W_3d = W_flat.reshape((R, p, n))
        
        # Reconstruct: T_recon[i,j,j2,k,i2,k2] = sum_r U[r,i,j]*V[r,j2,k]*W[r,k2,i2]
        # Using einsum: 'rij,rab,rcd->ijabdc'
        T_recon = np.einsum('rij,rab,rcd->ijabdc', U_3d, V_3d, W_3d)
        
        return (T_recon.flatten() - T_flat)
    
    def verify_solution(U, V, W, R):
        """Verify that the decomposition is exact."""
        U_3d = U.reshape((R, n, m))
        V_3d = V.reshape((R, m, p))
        W_3d = W.reshape((R, p, n))
        T_recon = np.einsum('rij,rab,rcd->ijabdc', U_3d, V_3d, W_3d)
        return np.max(np.abs(T_recon.flatten() - T_flat)) < 1e-9
    
    def snap_to_half_integers(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def polish_solution(x0, R, max_iter=500):
        """High-precision polish pass after initial convergence."""
        try:
            res = least_squares(
                residual_fn, x0, args=(R,),
                method='lm', max_nfev=max_iter,
                ftol=1e-14, xtol=1e-14, gtol=1e-14
            )
            return res
        except Exception:
            return None
    
    # Schoolbook fallback
    def get_schoolbook_solution():
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
    
    def get_near_schoolbook_init(R, rng):
        """Create initialization near schoolbook solution with random perturbations."""
        # Start with schoolbook pattern but only use R columns
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Use first R of the n*m*p schoolbook terms
        for col in range(min(R, n * m * p)):
            i = col // (m * p)
            rem = col % (m * p)
            j = rem // p
            k = rem % p
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
        
        # Add small random perturbations
        U += rng.randn(n * m, R) * 0.3
        V += rng.randn(m * p, R) * 0.3
        W += rng.randn(p * n, R) * 0.3
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 70 to 83 (below schoolbook = 84)
    np.random.seed(42)
    rng = np.random.RandomState(12345)
    
    for R in range(70, 84):
        if time.time() > deadline:
            break
            
        # Number of restarts: more for lower ranks (harder)
        n_restarts = 30 if R > 78 else 50
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                    break
                
            # Mix initialization strategies
            if restart % 3 == 0:
                # Pure random
                scale = 0.5
                x0 = rng.randn(n*m*R + m*p*R + p*n*R) * scale
            elif restart % 3 == 1:
                # Near schoolbook
                x0 = get_near_schoolbook_init(R, rng)
            else:
                # Random with larger scale
                scale = 1.0
                x0 = rng.randn(n*m*R + m*p*R + p*n*R) * scale
            
            try:
                res = least_squares(
                    residual_fn, x0, args=(R,), 
                    method='lm', max_nfev=800,
                    ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
            except Exception:
                continue
            
            # Check if the solution is close to exact
            if res.cost < 1e-8:
                # Polish pass for higher precision
                polished = polish_solution(res.x, R, max_iter=300)
                if polished is not None and polished.cost < res.cost:
                    res = polished
                
                # Extract and snap
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U_snap = snap_to_half_integers(U_flat)
                V_snap = snap_to_half_integers(V_flat)
                W_snap = snap_to_half_integers(W_flat)
                
                # Verify snapped solution
                if verify_solution(U_snap, V_snap, W_snap, R):
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((n*m, R)).tolist(),
                        "V": V_snap.reshape((m*p, R)).tolist(),
                        "W": W_snap.reshape((p*n, R)).tolist()
                    }
                    break
        
        if best_solution:
            break
    
    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook
        U, V, W = get_schoolbook_solution()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
