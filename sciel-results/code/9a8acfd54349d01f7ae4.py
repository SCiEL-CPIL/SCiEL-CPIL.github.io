import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _snap_and_verify(U_flat, V_flat, W_flat, R, n, m, p, T_flat, compute_residual):
    """Try to snap to integers or half-integers and verify."""
    # Try integer snapping
    U_int = np.round(U_flat)
    V_int = np.round(V_flat)
    W_int = np.round(W_flat)
    x_snapped = np.concatenate([U_int, V_int, W_int])
    res_snapped = compute_residual(x_snapped, R)
    if np.max(np.abs(res_snapped)) < 1e-9:
        return {
            "U": U_int.reshape((n * m, R)).tolist(),
            "V": V_int.reshape((m * p, R)).tolist(),
            "W": W_int.reshape((p * n, R)).tolist()
        }

    # Try half-integer snapping
    U_half = np.round(U_flat * 2) / 2
    V_half = np.round(V_flat * 2) / 2
    W_half = np.round(W_flat * 2) / 2
    x_snapped = np.concatenate([U_half, V_half, W_half])
    res_snapped = compute_residual(x_snapped, R)
    if np.max(np.abs(res_snapped)) < 1e-9:
        return {
            "U": U_half.reshape((n * m, R)).tolist(),
            "V": V_half.reshape((m * p, R)).tolist(),
            "W": W_half.reshape((p * n, R)).tolist()
        }
    
    return None

def _mps_strassen_decomposition(n, m, p, T_flat, compute_residual, deadline):
    """
    Try to find a decomposition using a tensor network / MPS-like structure.
    This exploits the hierarchical nature of matrix multiplication.
    """
    # We'll try to find a decomposition where the rank-R tensors have a
    # structured form that can be represented as a product of smaller tensors.
    
    # Strategy: Use alternating minimization on a constrained parameterization.
    # We'll represent U, V, W as having a Kronecker-like structure.
    
    # For a 3x4x7 multiplication, we can try to decompose it into smaller blocks.
    # Let's try a rank that is a product of small factors.
    
    # Candidate ranks to try with structured initialization
    candidate_ranks = [12, 15, 18, 20, 21, 24, 27, 30, 32, 35, 36, 40, 42, 45, 48]
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
            
        total_dim = n * m * R + m * p * R + p * n * R
        
        # Try structured initialization based on tensor network decomposition
        # We'll use a simple Kronecker product structure as initialization
        
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            if attempt == 0:
                # Structured initialization: try to use a Kronecker-like structure
                # This is a heuristic initialization that might help
                x0 = np.zeros(total_dim)
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                # Initialize with a simple pattern
                for r in range(R):
                    U_block = np.random.randn(n, m) * 0.1
                    V_block = np.random.randn(m, p) * 0.1
                    W_block = np.random.randn(p, n) * 0.1
                    x0[r*n*m:(r+1)*n*m] = U_block.flatten()
                    x0[size_U + r*m*p:size_U + (r+1)*m*p] = V_block.flatten()
                    x0[size_U + size_V + r*p*n:size_U + size_V + (r+1)*p*n] = W_block.flatten()
            elif attempt % 2 == 1:
                x0 = np.random.randn(total_dim) * 0.3
            else:
                x0 = np.random.randn(total_dim) * 0.05
            
            try:
                res = least_squares(
                    compute_residual, x0, args=(R,),
                    method='lm', max_nfev=500, ftol=1e-14, xtol=1e-14, gtol=1e-14
                )
            except Exception:
                continue
            
            if res.cost < 1e-12:
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                sol = _snap_and_verify(U_flat, V_flat, W_flat, R, n, m, p, T_flat, compute_residual)
                if sol is not None:
                    return sol, R
        
        # If direct optimization didn't work, try a multi-stage approach
        # First optimize with looser tolerance, then polish
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(total_dim) * 0.2
            
            try:
                # Stage 1: coarse optimization
                res1 = least_squares(
                    compute_residual, x0, args=(R,),
                    method='lm', max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8
                )
                
                if res1.cost > 1e-6:
                    continue
                    
                # Stage 2: fine optimization from coarse solution
                res2 = least_squares(
                    compute_residual, res1.x, args=(R,),
                    method='lm', max_nfev=1000, ftol=1e-14, xtol=1e-14, gtol=1e-14
                )
            except Exception:
                continue
            
            if res2.cost < 1e-12:
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U_flat = res2.x[:size_U]
                V_flat = res2.x[size_U:size_U + size_V]
                W_flat = res2.x[size_U + size_V:]
                
                sol = _snap_and_verify(U_flat, V_flat, W_flat, R, n, m, p, T_flat, compute_residual)
                if sol is not None:
                    return sol, R
    
    return None, None

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s safety margin for 150s limit

    n, m, p = DIMENSIONS

    # Target tensor T_flat
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
    T_flat = T.flatten()

    def compute_residual(params, R):
        """Compute residual for given parameters and rank, vectorized."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R

        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]

        U = U_flat.reshape((R, n, m))
        V = V_flat.reshape((R, m, p))
        W = W_flat.reshape((R, p, n))

        # Compute reconstruction
        uv = np.einsum('rij,rjk->rijjk', U, V)
        recon = np.einsum('rijjk,rki->ijkiki', uv, W)

        return (recon.flatten() - T_flat)

    def try_rank_general(R, n_restarts=10, max_nfev=200):
        """Try to find a valid decomposition at rank R using general search."""
        total_dim = n * m * R + m * p * R + p * n * R

        for attempt in range(n_restarts):
            if time.time() > deadline:
                return None

            if attempt % 2 == 0:
                x0 = np.random.randn(total_dim) * 0.5
            else:
                x0 = np.random.randn(total_dim) * 0.1

            try:
                res = least_squares(
                    compute_residual, x0, args=(R,),
                    method='lm', max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
            except Exception:
                continue

            if res.cost < 1e-10:
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R

                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]

                sol = _snap_and_verify(U_flat, V_flat, W_flat, R, n, m, p, T_flat, compute_residual)
                if sol is not None:
                    return sol
                
        return None

    best_solution = None
    best_rank = n * m * p

    # First, try the structured MPS/Strassen-like approach
    mps_sol, mps_rank = _mps_strassen_decomposition(n, m, p, T_flat, compute_residual, deadline)
    if mps_sol is not None:
        best_solution = mps_sol
        best_rank = mps_rank
    else:
        # Fall back to general search
        # Try ranks from 20 to 84
        for R in range(20, min(85, n * m * p)):
            if time.time() > deadline:
                break

            n_restarts = 15 if R <= 40 else 10
            max_nfev = 300 if R <= 40 else 200

            sol = try_rank_general(R, n_restarts=n_restarts, max_nfev=max_nfev)
            if sol is not None:
                best_rank = R
                best_solution = sol
                break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
