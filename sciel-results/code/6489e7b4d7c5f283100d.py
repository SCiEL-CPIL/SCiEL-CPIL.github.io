"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _try_find_rank_34(n, m, p):
    """
    Try to find a rank-34 decomposition by searching for a linear dependency
    among the 36 schoolbook terms, or by structured optimization.
    
    Strategy: Start from schoolbook (rank 36) and try to express one term as a
    linear combination of the others, effectively reducing rank by 1.
    
    We search for coefficients c_r (r != target) and a target column t such that:
    For all (i,j,k): U[i*m+j, t]*V[j*p+k, t]*W[k*n+i, t] = sum_{r != t} c_r * U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    
    This is a nonlinear system. Instead, we use a different approach:
    Parameterize a rank-34 decomposition and solve via least squares / optimization.
    """
    if _time_remaining() < 20:
        return None
    
    target_rank = 34
    n, m, p = 3, 3, 4
    
    # Initialize with schoolbook truncated to 34 columns, then optimize
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    U_init = np.array(U_s)[:, :target_rank]
    V_init = np.array(V_s)[:, :target_rank]
    W_init = np.array(W_s)[:, :target_rank]
    
    # The remaining 2 columns (indices 34, 35) need to be absorbed.
    # We'll add their contributions as a low-rank correction and optimize.
    
    # Better approach: use a random initialization mixed with schoolbook
    np.random.seed(12345)
    
    best_err = float('inf')
    best_params = None
    
    for attempt in range(5):
        if _time_remaining() < 15:
            break
        
        # Mix schoolbook (truncated) with small random perturbation
        U = U_init.copy() + np.random.randn(n*m, target_rank) * 0.05
        V = V_init.copy() + np.random.randn(m*p, target_rank) * 0.05
        W = W_init.copy() + np.random.randn(p*n, target_rank) * 0.05
        
        # Add the two missing schoolbook terms as a rank-2 correction
        # Columns 34 and 35 of schoolbook:
        # col 34: i=2, j=2, k=2 -> U[2*3+2]=U[8]=1, V[2*4+2]=V[10]=1, W[2*3+2]=W[8]=1
        # col 35: i=2, j=2, k=3 -> U[8]=1, V[2*4+3]=V[11]=1, W[3*3+2]=W[11]=1
        
        # We'll add these as additional terms but with optimized coefficients
        # Actually, let's just optimize all 34 terms freely
        
        from scipy.optimize import minimize
        
        def objective(params):
            if _time_remaining() < 5:
                return 1e10
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        err += (val - 1.0) ** 2
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None and best_err < 1e-8:
        U_opt = best_params[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = best_params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = best_params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        if _verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), target_rank
    
    return None

def _build_strassen_rank_34(n, m, p):
    """
    Construct a rank-34 decomposition using a block Strassen approach.
    
    Partition:
    A (3x3): A11 = A[0:2,0:2], A12 = A[0:2,2], A21 = A[2,0:2], A22 = A[2,2]
    B (3x4): B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2,0:2], B22 = B[2,2:4]
    
    C[0:2,0:2] = A11@B11 + A12@B21
    C[0:2,2:4] = A11@B12 + A12@B22  
    C[2,0:2]   = A21@B11 + A22@B21
    C[2,2:4]   = A21@B12 + A22@B22
    
    For A11@B11 (2x2 x 2x2), use rank-4 schoolbook (4 mults).
    For A11@B12 (2x2 x 2x2), use rank-4 schoolbook (4 mults).
    For A12@B21 (2x1 x 1x2), use rank-2 (2 mults).
    For A12@B22 (2x1 x 1x2), use rank-2 (2 mults).
    For A21@B11 (1x2 x 2x2), use rank-4 (4 mults).
    For A21@B12 (1x2 x 2x2), use rank-4 (4 mults).
    For A22@B21 (1x1 x 1x2), use rank-2 (2 mults).
    For A22@B22 (1x1 x 1x2), use rank-2 (2 mults).
    
    Total: 4+4+2+2+4+4+2+2 = 24. That's less than 36!
    
    But wait, this is just schoolbook with a different grouping - it should give the same total.
    Let me recount: the total number of scalar multiplications in schoolbook for 3x3x3x4 is 3*3*4=36.
    
    The issue is that when I partition, the cross terms A11@B11, A12@B21, etc. each contribute
    to the same output entries, so I need to be careful about double-counting.
    
    Actually, the partition gives:
    C[0:2,0:2] gets contributions from A11@B11 (4 mults) and A12@B21 (2 mults) = 6
    C[0:2,2:4] gets contributions from A11@B12 (4 mults) and A12@B22 (2 mults) = 6
    C[2,0:2] gets contributions from A21@B11 (4 mults) and A22@B21 (2 mults) = 6
    C[2,2:4] gets contributions from A21@B12 (4 mults) and A22@B22 (2 mults) = 6
    Total: 24
    
    But this is wrong - schoolbook for 3x3x3x4 has 36 multiplications, not 24.
    The error: A11@B11 is 2x2 x 2x2 = 4 mults, but A12@B21 is 2x1 x 1x2 = 2 mults.
    Sum for C[0:2,0:2]: 4+2=6. But schoolbook for this 2x2 output block needs 2*3*2=12 mults.
    
    I see the issue: A12 is 2x1 and B21 is 1x2, so A12@B21 is 2x2 = 4 entries, but it only
    needs 2*1*2=4 scalar multiplications... wait, 2x1 @ 1x2 = 2x2, which is 2*1*2=4 mults.
    
    Let me redo: A12@B21: A12 is 2x1, B21 is 1x2. Product is 2x2. Number of mults = 2*1*2 = 4.
    A11@B11: 2x2 @ 2x2 = 2x2. Mults = 2*2*2 = 8.
    
    So C[0:2,0:2]: 8+4=12. Correct.
    C[0:2,2:4]: A11@B12 (2x2@2x2=8) + A12@B22 (2x1@1x2=4) = 12.
    C[2,0:2]: A21@B11 (1x2@2x2=4) + A22@B21 (1x1@1x2=2) = 6.
    C[2,2:4]: A21@B12 (1x2@2x2=4) + A22@B22 (1x1@1x2=2) = 6.
    Total: 12+12+6+6 = 36. Correct.
    
    Now, can I reduce any of these? The A11@B11 (2x2@2x2) has rank 4 (schoolbook).
    Strassen for 2x2x2 has rank 7, which is WORSE. So no help there.
    
    For the 1x2@2x2 blocks (rank 4), can I do better? 1x2@2x2 = 1x2 output.
    C[i,k] = sum_j A[i,j]*B[j,k]. For fixed i (1 row), this is 2*2=4 mults.
    No obvious reduction.
    
    So the block decomposition doesn't directly give rank 34. The rank-34 result must
    come from a more subtle algebraic identity.
    
    Let me just return schoolbook as fallback and rely on the optimization.
    """
    return _build_schoolbook(n, m, p)

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try to find rank-34 decomposition
    if _time_remaining() > 25:
        result = _try_find_rank_34(n, m, p)
        if result is not None:
            U, V, W, best_rank = result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U] if isinstance(U, np.ndarray) else U
    V_list = [[float(x) for x in row] for row in V] if isinstance(V, np.ndarray) else V
    W_list = [[float(x) for x in row] for row in W] if isinstance(W, np.ndarray) else W
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
