"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify_decomposition(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank_optimization(rank, seed=42, time_limit=10.0):
    """
    Attempt to find a rank-`rank` decomposition using non-linear optimization.
    We solve the system: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    """
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    # Total variables: n*m*rank + m*p*rank + p*n*rank
    n_u = n * m * rank
    n_v = m * p * rank
    n_w = p * n * rank
    total_vars = n_u + n_v + n_w
    
    # Residual function
    def residual(x):
        U_flat = x[:n_u]
        V_flat = x[n_u:n_u + n_v]
        W_flat = x[n_u + n_v:]
        
        U = U_flat.reshape((n, m, rank))
        V = V_flat.reshape((m, p, rank))
        W = W_flat.reshape((p, n, rank))
        
        # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # We want T[i, j, k] == 1 for all i, j, k
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i,j,:] is shape (rank,)
                    # V[j,k,:] is shape (rank,)
                    # W[k,i,:] is shape (rank,)
                    dot = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                    residuals[i, j, k] = dot - 1.0
        return residuals.flatten()

    # Initialization
    # Random initialization often works better than zeros for these symmetric problems
    # We use a fixed seed for reproducibility but can vary it in a loop if needed
    rng = np.random.RandomState(seed)
    x0 = rng.randn(total_vars) * 0.5
    
    # Symmetry breaking: fix one variable to 1 to prevent trivial zero solution scaling
    # The system is homogeneous in U, V, W (if U,V,W is a solution, cU, cV, c/c^2 W is not, wait.
    # sum (cU)(cV)(cW) = c^3 sum UVW. So scaling is not a symmetry of the equation = 1.
    # However, if we had sum UVW = 0, scaling would be a symmetry.
    # Since target is 1, the scale is fixed. But there are local minima.
    # Let's just try random start.
    
    try:
        result = least_squares(residual, x0, method='lm', max_nfev=10000)
    except Exception as e:
        # LM might fail, try TRF
        try:
            result = least_squares(residual, x0, method='trf', max_nfev=10000)
        except:
            return None

    if time.time() - t0 > time_limit:
        return None

    # Check if solution is good
    if result.cost < 1e-12:
        # Extract U, V, W
        U_flat = result.x[:n_u]
        V_flat = result.x[n_u:n_u + n_v]
        W_flat = result.x[n_u + n_v:]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        # The problem requires integer/half-integer entries.
        # Optimization usually gives floats. We need to check if they are close to half-integers.
        # Or, we can try to snap to nearest 0.5 and verify.
        
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify the snapped version
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
            
        # If snapping failed, maybe the optimal solution is exact integers but slightly off?
        # Or maybe it's not half-integer? The task says "integer/half-integer".
        # If the optimizer found a valid real solution, but not half-integer, it's not a valid output.
        # However, often these decompositions have small integer entries.
        # Let's try snapping to integers as well? No, half-integers are more general.
        
        # If the cost is low but snapping fails, it might be a non-rational solution.
        # We return None in that case.
        
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Schoolbook is rank 54.
    # We search for lower ranks.
    # Lower bound: max(3*3, 3*6, 6*3) = 18.
    # However, Strassen-like algorithms might reduce it.
    # For 3x3x3, rank is 23.
    # For 3x3x6, we can treat it as 3x3x3 + 3x3x3? No, that's not how it works.
    # We can use the fact that 3x3x6 can be embedded in larger blocks.
    # Let's try ranks from 53 down to 18.
    # We prioritize higher ranks first as they are more likely to have solutions.
    # We allocate time to each rank attempt.
    
    # We will try a few random seeds for each rank to escape local minima.
    ranks_to_try = list(range(53, 17, -1))
    
    # Time budget per rank: 115s / 36 ranks ~ 3s.
    # But optimization might be slow. Let's do 2s per seed, 2 seeds per rank = 4s.
    # 36 * 4 = 144s > 115s.
    # So we can only try about 28 ranks with 2 seeds each.
    # Or fewer ranks with more seeds.
    # Let's try ranks 53 down to 20. That's 34 ranks.
    # If we spend 3s per rank (1 seed), that's 102s.
    
    best_solution = None
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Try 2 different seeds for this rank
        found = False
        for seed in [42, 123]:
            if time.time() > deadline - 5:
                break
                
            res = _try_rank_optimization(rank, seed=seed, time_limit=3.0)
            if res is not None:
                best_solution = res
                found = True
                break
        
        if found:
            # If we found a solution, we can return it immediately if we want the first one.
            # But the task asks for rank-minimal.
            # So we should continue searching for lower ranks?
            # "Find a rank-minimal exact tensor decomposition"
            # If we find rank 50, we should keep looking for 49, 48...
            # But if we find rank 50, and then fail to find 49, we return 50.
            # So we should not return immediately. We should store it and continue.
            pass
            
    if best_solution is not None:
        return best_solution
        
    return _schoolbook()
# EVOLVE-BLOCK-END
