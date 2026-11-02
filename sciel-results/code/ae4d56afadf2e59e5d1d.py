"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave margin for return and overhead
    
    # Schoolbook as fallback (rank 24)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Theoretical lower bound
    min_rank = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook - 1
    # We prioritize lower ranks
    ranks_to_try = list(range(min_rank, best_rank))
    
    # If time is tight, maybe try a subset? 
    # For 2x3x4, min_rank=12, best_rank=24. 12 ranks to try.
    # 140s / 12 ranks ~ 11s per rank.
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Allocate time for this rank
        # We want to spend more time on lower ranks
        # Simple allocation: proportional to remaining time / remaining ranks
        remaining_ranks = best_rank - r
        time_left = deadline - time.time()
        if time_left <= 0:
            break
        
        # Let's give each rank a fixed budget, but ensure we don't overshoot
        # If we have 12 ranks and 140s, ~11s each.
        rank_time_limit = min(15.0, time_left / max(1, remaining_ranks))
        
        # Try to find a solution for this rank
        solution = _search_rank_optimization(n, m, p, r, rank_time_limit)
        
        if solution is not None:
            return solution
            
    # If no better solution found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

def _search_rank_optimization(n, m, p, rank, time_limit):
    """
    Try to find a rank decomposition using continuous optimization and snapping.
    """
    start_time = time.time()
    
    # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Total variables: (n*m + m*p + p*n) * rank
    # For 2,3,4, rank=12: (6+12+8)*12 = 312 variables.
    
    # Objective function
    def objective(x):
        # Unpack x into U, V, W
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u + size_v]
        W_flat = x[size_u + size_v:size_u + size_v + size_w]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        # Calculate error: sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
        # Vectorized calculation
        # We need to sum over r for each (i,j,k)
        # This is a trilinear form.
        # Efficient way: 
        # For each r, compute outer product of U[:,r] and V[:,r]? No, indices are coupled.
        
        # Let's loop over r and accumulate
        error = 0.0
        for r_idx in range(rank):
            # u_col: (n*m, 1) -> (n, m)
            u_col = U[:, r_idx].reshape((n, m))
            # v_col: (m*p, 1) -> (m, p)
            v_col = V[:, r_idx].reshape((m, p))
            # w_col: (p*n, 1) -> (p, n)
            w_col = W[:, r_idx].reshape((p, n))
            
            # The term is sum_{i,j,k} u_col[i,j] * v_col[j,k] * w_col[k,i]
            # This is trace( u_col * v_col * w_col )? 
            # Let's check: 
            # (u_col @ v_col) is (n, p). 
            # Let M = u_col @ v_col. M[i,k] = sum_j u[i,j]v[j,k].
            # We want sum_{i,k} M[i,k] * w_col[k,i].
            # This is the Frobenius inner product of M and w_col.T.
            # Or trace(M @ w_col).
            # Let's verify: trace(M @ w_col) = sum_i (M @ w_col)[i,i] = sum_i sum_k M[i,k] w_col[k,i].
            # Yes.
            
            M = u_col @ v_col
            term = np.trace(M @ w_col)
            error += (term - 1.0) ** 2 # Wait, this sums over all i,k?
            # No, the target is 1 for EACH (i,j,k).
            # The tensor T[i,j,k] = 1.
            # My formulation above sums over i,j,k for a fixed r.
            # This is incorrect. The constraint is per (i,j,k).
            
        # Correct Objective:
        # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for ALL i,j,k.
        # Let's build the tensor of the decomposition.
        # T_decomp[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want T_decomp to be all ones.
        
        # Vectorized construction of T_decomp:
        # T_decomp = np.zeros((n, m, p))
        # for r in range(rank):
        #     u = U[:, r].reshape(n, m)
        #     v = V[:, r].reshape(m, p)
        #     w = W[:, r].reshape(p, n)
        #     # How to combine u, v, w to get a 3D tensor contribution?
        #     # The term is u[i,j] * v[j,k] * w[k,i].
        #     # This is not a simple outer product of 3 vectors.
        #     # It is a specific contraction.
        #     # Let's use einsum or manual loops if rank is small.
        #     # Since n,m,p are small (2,3,4), manual loops inside the objective
        #     # might be slow if called many times.
        #     # However, rank is ~12-24. n*m*p is 24.
        #     # Total ops per objective eval: rank * n*m*p ~ 24*24 = 576 multiplications.
        #     # This is fast enough for scipy minimize.
            
        # Let's implement the objective with explicit loops for clarity and correctness,
        # relying on numpy for the inner products if possible.
        
        # Actually, we can pre-compute the indices.
        # But let's just use the loop.
        
        # Re-initialize error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over r
                    s = 0.0
                    for r_idx in range(rank):
                        s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    err_sum += (s - 1.0) ** 2
        return err_sum

    # Initial guess
    # Random small values
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    total_size = size_u + size_v + size_w
    
    # Try a few restarts
    max_restarts = 5
    best_err = float('inf')
    best_x = None
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Random initialization
        # Scale matters. If we start with 0, gradient is 0.
        # Start with small random values.
        x0 = np.random.normal(0, 0.1, total_size)
        
        # Optimization
        # Use Nelder-Mead for small problems without gradient, or L-BFGS-B if we provide gradient.
        # Providing gradient is faster.
        def gradient(x):
            # Numerical gradient is expensive.
            # Let's try Nelder-Mead first, it's robust for small dims.
            # Or use L-BFGS-B with numerical gradient? SciPy does that if jac is not provided.
            pass
        
        # Let's use L-BFGS-B with numerical gradient.
        # It's usually efficient.
        options = {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options=options)
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception as e:
            pass
            
        # Check time
        if time.time() > start_time + time_limit * 0.8:
            break

    if best_x is None:
        return None
        
    # Check if the solution is good enough
    # If best_err is very small, try to snap to half-integers
    if best_err < 1e-6:
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U_flat = best_x[:size_u]
        V_flat = best_x[size_u:size_u + size_v]
        W_flat = best_x[size_u + size_v:]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify snapped solution
        if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
            # Convert to list of lists
            U_list = [[float(U_snap[i, r]) for r in range(rank)] for i in range(n * m)]
            V_list = [[float(V_snap[i, r]) for r in range(rank)] for i in range(m * p)]
            W_list = [[float(W_snap[i, r]) for r in range(rank)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
            
        # If snapping failed, maybe the solution is slightly off.
        # We could try a local search around the snapped solution?
        # Or just accept if the error is extremely small and hope the evaluator is lenient?
        # No, "exactly decompose". Must be exact.
        
        # Try to refine the snapped solution by checking neighbors?
        # This is a discrete search.
        # Given the time, we might just return None and let the outer loop try the next rank.
        
    return None

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Check if s is 1.0
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
