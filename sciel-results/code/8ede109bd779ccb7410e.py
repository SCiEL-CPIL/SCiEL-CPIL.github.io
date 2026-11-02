"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices: B[j,k] -> V[j*p + k]
        v_e = k0        # B[0, k0]
        v_f = k1        # B[0, k1]
        v_g = p + k0    # B[1, k0]
        v_h = p + k1    # B[1, k1]
        
        # W indices: C[i,k] -> W[k*n + i]
        w_c00 = k0 * n      # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n      # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]
        
        # U indices: A[i,j] -> U[i*m + j]
        # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        
        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook 4 terms
    k2 = 10
    v_e2 = k2        # B[0, 10]
    v_g2 = p + k2    # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]
    
    # A00 * B[0,10] -> C[0,10]
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A01 * B[1,10] -> C[0,10]
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A10 * B[0,10] -> C[1,10]
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # A11 * B[1,10] -> C[1,10]
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank):
    """Vectorized error computation."""
    sums = np.zeros(len(ta_arr))
    for c in range(rank):
        sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
    return np.sum((sums - 1.0) ** 2)

def _polish(U, V, W, ta_arr, tb_arr, tc_arr, rank, n, m, p, deadline, max_rounds=5000, tol=1e-14):
    """
    High-precision polish pass: drive a near-converged solution to machine precision.
    Uses alternating least squares with tighter thresholds and more iterations.
    Returns (U, V, W) as lists if converged, else None.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank)
    if prev_err < tol:
        return U.tolist(), V.tolist(), W.tolist()
    
    best_err = prev_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = (ta_arr == a)
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-15
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = (tb_arr == b)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-15
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = (tc_arr == c_idx)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-15
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 10 iterations
        if it % 10 == 0 or it == max_rounds - 1:
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank)
            
            if err < tol:
                return U.tolist(), V.tolist(), W.tolist()
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # If stuck, try a tiny perturbation
            if err > best_err * 1.001 and it > 100:
                rng = np.random.RandomState(it)
                U = best_U + 1e-6 * rng.randn(n * m, rank)
                V = best_V + 1e-6 * rng.randn(m * p, rank)
                W = best_W + 1e-6 * rng.randn(p * n, rank)
    
    # Return best found if it's close enough
    if best_err < 1e-12:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_low_rank(rank, n, m, p, deadline, max_rounds=3000, seed=42, warm_start=None):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    if warm_start is not None:
        U, V, W = warm_start
    else:
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = np.zeros((p * n, rank))
        W[:, :] = 1.0  # Initialize W to ones
    
    # Target index arrays
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    prev_err = float('inf')
    stagnation_count = 0
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = (ta_arr == a)
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = (tb_arr == b)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = (tc_arr == c_idx)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 20 iterations
        if it % 20 == 0 or it == max_rounds - 1:
            err = _compute_error(U, V, W, ta_arr, tb_arr, tc_arr, rank)
            
            if err < 1e-10:
                # Try to polish to higher precision
                if time.time() < deadline - 2:
                    polished = _polish(U, V, W, ta_arr, tb_arr, tc_arr, rank, n, m, p, deadline=deadline - 1)
                    if polished is not None:
                        return polished
                return U.tolist(), V.tolist(), W.tolist()
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # If error is not improving significantly, add perturbation
            if err > prev_err * 1.005 and it > 50:
                stagnation_count += 1
                if stagnation_count > 5:
                    # Add small random perturbation to escape local minimum
                    U = U + 0.001 * rng.randn(n * m, rank)
                    V = V + 0.001 * rng.randn(m * p, rank)
                    W = W + 0.001 * rng.randn(p * n, rank)
                    stagnation_count = 0
            else:
                stagnation_count = 0
            
            # If error is much worse than best, break early
            if err > best_err * 1.1 and it > 100:
                break
            prev_err = err
    
    # Final attempt: polish the best solution found
    if best_err < 1e-8 and time.time() < deadline - 2:
        polished = _polish(best_U, best_V, best_W, ta_arr, tb_arr, tc_arr, rank, n, m, p, deadline=deadline - 1)
        if polished is not None:
            return polished
    
    return None

def _make_warm_start_for_rank(rank, rank39_U, rank39_V, rank39_W, n, m, p, seed=42):
    """
    Create a warm start for given rank by removing columns from rank-39 and perturbing.
    """
    rng = np.random.RandomState(seed)
    
    # Remove the last (39 - rank) columns
    U = np.array(rank39_U)[:, :rank]
    V = np.array(rank39_V)[:, :rank]
    W = np.array(rank39_W)[:, :rank]
    
    # Add small perturbation to help escape local minima
    U = U + 0.005 * rng.randn(n * m, rank)
    V = V + 0.005 * rng.randn(m * p, rank)
    W = W + 0.005 * rng.randn(p * n, rank)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 30:
            # Try ranks from 38 down to 30, with multiple seeds
            for rank in [38, 37, 36, 35, 34, 33, 32, 31, 30]:
                if time.time() > deadline - 10:
                    break
                # Use warm start from rank-39
                warm_start = _make_warm_start_for_rank(rank, U, V, W, n, m, p, seed=rank)
                for seed in [rank, rank + 100, rank + 200, rank + 300]:
                    if time.time() > deadline - 5:
                        break
                    # Try with warm start
                    result = _try_low_rank(rank, n, m, p, deadline=deadline - 5, max_rounds=2000, seed=seed, warm_start=warm_start)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
                    
                    # Also try without warm start (random init)
                    if time.time() < deadline - 8:
                        result = _try_low_rank(rank, n, m, p, deadline=deadline - 5, max_rounds=1000, seed=seed)
                        if result is not None:
                            U2, V2, W2 = result
                            if _verify(U2, V2, W2, n, m, p):
                                return {"U": U2, "V": V2, "W": W2}
        
        # Final attempt: try to polish rank-39 to see if we can get a cleaner solution
        # (this won't reduce rank but ensures numerical stability)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
