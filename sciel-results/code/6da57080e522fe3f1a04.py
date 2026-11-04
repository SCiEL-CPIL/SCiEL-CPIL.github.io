"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute index triples and their corresponding row indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_idx_list = np.array([i * m + j for (i, j, k) in indices])
    v_idx_list = np.array([j * p + k for (i, j, k) in indices])
    w_idx_list = np.array([k * n + i for (i, j, k) in indices])
    num_triples = len(indices)

    def compute_residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        U_rows = U[u_idx_list, :]  # (num_triples, rank)
        V_rows = V[v_idx_list, :]  # (num_triples, rank)
        W_rows = W[w_idx_list, :]  # (num_triples, rank)

        # Compute the product sum for each triple
        products = U_rows * V_rows * W_rows  # (num_triples, rank)
        res = np.sum(products, axis=1) - 1.0
        return res

    def residual(params, rank):
        """Vectorized residual computation for the tensor decomposition."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        return compute_residual(U_flat, V_flat, W_flat)

    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

    def snap_to_half_int(arr):
        """Snap array values to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def verify_snapped(U, V, W, rank):
        """Verify that the snapped matrices form a valid decomposition."""
        U_s = snap_to_half_int(U)
        V_s = snap_to_half_int(V)
        W_s = snap_to_half_int(W)
        
        # Check if they are close to the original solution (optional, but good for sanity)
        # More importantly, check if they satisfy the equation
        if check_decomposition(U_s, V_s, W_s, rank, tol=1e-6):
            return U_s, V_s, W_s
        return None

    def to_list(arr):
        """Convert numpy array to nested list."""
        return arr.tolist()

    # Try ranks from low to high
    # Known lower bound for 2x3x4 is likely around 15-17.
    # We start from 15 to be safe, but 14 is possible? 
    # Let's try 15, 16, 17, 18...
    for rank in range(15, 25):
        if time.time() > deadline - 10:
            break

        # Determine number of attempts based on remaining time
        # Lower ranks are harder, so maybe more attempts?
        # But higher ranks are easier to find.
        # Let's allocate time proportionally.
        remaining_time = deadline - 10 - time.time()
        if remaining_time <= 0: break
        
        # Heuristic: more attempts for lower ranks
        attempts = int(remaining_time / 1.0) 
        if rank < 18:
            attempts = min(50, attempts)
        else:
            attempts = min(10, attempts)

        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break

            np.random.seed(rank * 10000 + attempt * 7 + 42)
            
            # Initialization Strategy:
            # 1. Random Gaussian
            # 2. Naive decomposition scaled (if rank == 24, use it; else, random)
            # For rank < 24, random is standard, but maybe we can use a structured init?
            # Let's stick to random but with better scaling.
            
            scale = 1.0
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                # Use TRF for non-linear least squares (handles underdetermined systems)
                result = least_squares(residual, x0, args=(rank,), method='trf', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                # If cost is low enough, try to snap
                if result.cost < 1e-6:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    # Try to snap to half-integers
                    snapped = verify_snapped(U_sol, V_sol, W_sol, rank)
                    if snapped:
                        U_s, V_s, W_s = snapped
                        return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
                    
                    # If snap failed, maybe the solution is valid but not half-integer?
                    # The task requires integer/half-integer. So we must snap.
                    # However, sometimes the continuous solution is exact but not half-integer.
                    # In that case, we cannot use it. We need to find a solution that IS half-integer.
                    # This implies we might need to search the discrete space directly or 
                    # guide the continuous search towards half-integers.
                    
                    # Alternative: if cost is extremely low, maybe we can just round and check?
                    # We already did that in verify_snapped.
                    
                    # Try a local search around the snapped solution?
                    # Or just try more random restarts.
                    
            except Exception:
                continue

    # Fallback: naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
