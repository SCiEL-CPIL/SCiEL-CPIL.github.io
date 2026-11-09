"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array.
    
    T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    T[a,b,c] = 1 if j=j' and k'=k'' and i''=i, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries (should be 0)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for ip in range(n):
                        if jp == j and kp == k and ip == i:
                            continue
                        val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+kp, :] * W_arr[kp*n+ip, :])
                        if abs(val) > tol:
                            return False
    return True


def _sa_decomposition(T, n, m, p, rank, time_budget=10.0, seed=42):
    """Use Simulated Annealing to find a rank-r decomposition with integer/half-integer entries."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random integers
    # Scale by 2 to allow half-integers later? Or just integers first.
    # Let's try integers in {-1, 0, 1} initially.
    
    U = rng.randint(-1, 2, size=(n * m, rank))
    V = rng.randint(-1, 2, size=(m * p, rank))
    W = rng.randint(-1, 2, size=(p * n, rank))
    
    # Precompute indices for non-zero target entries to speed up error calculation
    # T is sparse. Only n*m*p entries are 1.
    # But we need to check ALL entries for error?
    # The error is sum over all a,b,c of (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
    # This is expensive.
    
    # Alternative: Only check the 1s and a random subset of 0s?
    # For exactness, we must check all.
    # n*m = 9, m*p = 24, p*n = 24. Total entries = 9*24*24 = 5184.
    # Rank ~ 30-50.
    # 5184 * 50 operations per error calc. ~ 250k ops.
    # This is fast enough for SA steps.
    
    def calc_error(U, V, W):
        # Compute tensor product
        # T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this efficiently?
        # For each r, add outer product of U[:,r], V[:,r], W[:,r]
        # This is rank-1 update.
        # Let's just do it directly for simplicity, it's small.
        err = 0.0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    target = T[a, b, c]
                    err += (val - target) ** 2
        return err

    current_error = calc_error(U, V, W)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    T_start = 1.0
    T_end = 0.01
    alpha = 0.99
    
    current_temp = T_start
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Generate a neighbor
        # Pick a random matrix and a random entry
        mat_idx = rng.randint(0, 3)
        if mat_idx == 0:
            row, col = rng.randint(0, n * m), rng.randint(0, rank)
            mat = U
        elif mat_idx == 1:
            row, col = rng.randint(0, m * p), rng.randint(0, rank)
            mat = V
        else:
            row, col = rng.randint(0, p * n), rng.randint(0, rank)
            mat = W
            
        old_val = mat[row, col]
        # Change to a nearby integer
        # Try -1, 0, 1, 2, -2
        new_val = old_val + rng.choice([-1, 1])
        # Keep values small
        if new_val > 5: new_val = 5
        if new_val < -5: new_val = -5
        
        mat[row, col] = new_val
        
        new_error = calc_error(U, V, W)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_error - current_error
        if delta < 0 or rng.rand() < np.exp(-delta / current_temp):
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error < 1e-10:
                    break
        else:
            mat[row, col] = old_val
            
        current_temp *= alpha
        
    # Return the best found
    if best_error < 1e-10:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None


def _try_rank(T, n, m, p, rank, time_budget=10.0, seed=42):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    try:
        U, V, W = _sa_decomposition(T, n, m, p, rank, time_budget=time_budget, seed=seed)
        if time.time() - start > time_budget:
            return None
        if U is not None and _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            return U, V, W
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        pass
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    if remaining < 15.0:
        return best_result
    
    # Try SA with various ranks
    # Lower bound is 24. Try 25, 26, ..., 40
    # Start with lower ranks to see if we can find a solution
    ranks_to_try = [25, 26, 27, 28, 30, 32, 35, 40]
    
    for rank in ranks_to_try:
        remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
        if remaining < 15.0:
            break
        
        time_budget = min(remaining * 0.2, 15.0)
        result = _try_rank(T, n, m, p, rank, time_budget=time_budget, seed=42 + rank)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = rank
            # If we found a very low rank, we might stop or try even lower
            # But let's keep trying to see if we can go lower
            if rank <= 25:
                break

    return best_result
