import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the target tensor T[i*m+j, j*p+k] -> k*n+i as a linear map.
    Actually, the tensor is: T[i, j, k] = 1 if output index is (k, i).
    We represent the bilinear form: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
    So we need: for each r, the outer product U_r (n x m) * V_r (m x p) should contribute to W_r (p x n).
    Specifically: (U_r @ V_r).T should equal W_r (reshaped as p x n).
    """
    pass

def _tensor_residuals(flat_params, n, m, p, rank):
    """
    flat_params contains rank * (n*m + m*p + p*n) entries.
    Reshape into U (n*m, rank), V (m*p, rank), W (p*n, rank).
    For each rank component r:
        U_r reshaped as (n, m), V_r reshaped as (m, p), W_r reshaped as (p, n)
        We need: (U_r @ V_r).T == W_r
    So residual for component r is: (U_r @ V_r).T - W_r, flattened.
    Total residual size: rank * p * n
    """
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    U = flat_params[:rank * U_size].reshape(rank, U_size)
    V = flat_params[rank * U_size : rank * U_size + rank * V_size].reshape(rank, V_size)
    W = flat_params[rank * U_size + rank * V_size :].reshape(rank, W_size)
    
    residuals = np.zeros(rank * p * n)
    for r in range(rank):
        Ur = U[r].reshape(n, m)
        Vr = V[r].reshape(m, p)
        Wr = W[r].reshape(p, n)
        # (Ur @ Vr) is (n, p), transpose to (p, n)
        prod = (Ur @ Vr).T
        residuals[r * p * n : (r + 1) * p * n] = prod - Wr
    
    return residuals

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify exact decomposition with integer/half-integer entries."""
    r = len(U[0]) if U else 0
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n * m)])
        vc = np.array([V[j][c] for j in range(m * p)])
        wc = np.array([W[k][c] for k in range(p * n)])
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod.T, w_mat, atol=1e-9):
            return False
    # Check that all entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _optimize_rank(rank, n, m, p, deadline, seed=0):
    """
    Try to find a rank-`rank` decomposition by continuous optimization.
    Returns (U_list, V_list, W_list) if found, else None.
    """
    if time.time() > deadline:
        return None
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    total_params = rank * (U_size + V_size + W_size)
    
    # Initialize with schoolbook-like pattern for first rank components, random for rest
    rng = np.random.RandomState(seed)
    x0 = np.zeros(total_params)
    
    # Fill in schoolbook pattern for min(rank, n*m*p) components
    n_schoolbook = min(rank, n * m * p)
    for r in range(n_schoolbook):
        # Assign one (i,j,k) triple to each rank component
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        if i < n and j < m and k < p:
            # U: row i*m+j, col r
            U_idx = r * U_size + i * m + j
            V_idx = r * U_size + r * V_size + j * p + k
            W_idx = r * U_size + r * V_size + r * W_size + k * n + i
            x0[U_idx] = 1.0
            x0[V_idx] = 1.0
            x0[W_idx] = 1.0
    
    # Add small random perturbation to break symmetry
    x0 += rng.randn(total_params) * 0.1
    
    # Run optimization
    result = least_squares(
        _tensor_residuals,
        x0,
        args=(n, m, p, rank),
        method='lm',  # Levenberg-Marquardt, good for small problems
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )
    
    if time.time() > deadline:
        return None
    
    # Check if solution is close enough
    final_res = np.max(np.abs(result.fun))
    
    if final_res < 1e-8:
        # Round to nearest half-integer
        x_rounded = np.round(result.x * 2) / 2.0
        
        # Verify the rounded solution
        U = x_rounded[:rank * U_size].reshape(rank, U_size).T  # (U_size, rank)
        V = x_rounded[rank * U_size : rank * U_size + rank * V_size].reshape(rank, V_size).T
        W = x_rounded[rank * U_size + rank * V_size :].reshape(rank, W_size).T
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _search_ranks(n, m, p, deadline):
    """
    Search for the minimal rank by trying decreasing rank values.
    """
    best_result = _construct_schoolbook()
    best_rank = n * m * p
    
    # Theoretical lower bounds: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But we need to be realistic. Try a range of ranks.
    # For 3x5x7, known best might be around 50-70.
    
    # Try ranks from a reasonable range
    # Start from schoolbook and go down
    for rank in range(best_rank - 1, max(30, best_rank - 40), -1):
        if time.time() > deadline - 5:
            break
        
        result = _optimize_rank(rank, n, m, p, deadline, seed=rank)
        if result is not None:
            best_rank = rank
            best_result = result
            # Continue trying lower ranks
        else:
            # Try a few more seeds for this rank
            for seed in range(1, 4):
                if time.time() > deadline - 5:
                    break
                result = _optimize_rank(rank, n, m, p, deadline, seed=seed + 100)
                if result is not None:
                    best_rank = rank
                    best_result = result
                    break
    
    return best_result, best_rank

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook()
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try continuous optimization for lower ranks
    if time.time() < deadline - 10:
        result, rank = _search_ranks(n, m, p, deadline)
        if rank < best_rank:
            best_rank = rank
            best_result = result
    
    return best_result
