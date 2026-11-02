import time
import numpy as np
from fractions import Fraction
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _verify_strict(U, V, W, tol=1e-9):
    """
    Strictly verify that U, V, W decompose the matrix multiplication tensor.
    Checks both the non-zero entries (should sum to 1) and zero entries (should sum to 0).
    """
    n, m, p = 2, 2, 11
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Build target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    # Otherwise 0.
    
    # Efficient check using einsum for the full tensor
    # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Create target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Check allclose
    if not np.allclose(T_decomp, T_target, atol=tol, rtol=0):
        return False
        
    return True

def _construct_rank39():
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
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
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank, f"Rank mismatch: {r} != {rank}"
    
    return {"U": U, "V": V, "W": W}

def _try_low_rank(target_rank):
    """
    Attempt to find a decomposition of target_rank using scipy.optimize.
    We minimize the sum of squares of errors over ALL entries of the tensor.
    """
    if time.time() > _DEADLINE - 20.0:
        return None

    n, m, p = 2, 2, 11
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Precompute target tensor
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    best_err = float('inf')
    best_solution = None
    
    # Try multiple random initializations and warm starts
    inits = []
    
    # 1. Random init
    np.random.seed(42)
    U_rand = np.random.randn(dim_u, target_rank) * 0.1
    V_rand = np.random.randn(dim_v, target_rank) * 0.1
    W_rand = np.random.randn(dim_w, target_rank) * 0.1
    inits.append((U_rand, V_rand, W_rand))
    
    # 2. Warm start from rank 39 (drop columns)
    if target_rank < 39:
        base = _construct_rank39()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)
        
        # Try dropping different sets of columns
        # We can just take the first target_rank columns, or random subsets
        # For now, simple: take first target_rank
        U_ws = U_base[:, :target_rank]
        V_ws = V_base[:, :target_rank]
        W_ws = W_base[:, :target_rank]
        inits.append((U_ws, V_ws, W_ws))
        
        # Maybe try dropping the "schoolbook" part if target_rank is significantly lower?
        # But rank 38 is only 1 less.
        
    for init_idx, (U_init, V_init, W_init) in enumerate(inits):
        if time.time() > _DEADLINE - 10.0:
            break
            
        def objective(x):
            U = x[:dim_u*target_rank].reshape(dim_u, target_rank)
            V = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
            W = x[dim_u*target_rank + dim_v*target_rank :].reshape(dim_w, target_rank)
            
            # Compute T_decomp
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Error is sum of squared differences
            err = np.sum((T_decomp - T_target)**2)
            return err
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use L-BFGS-B for bounded optimization, or Nelder-Mead for unbounded
            # Since we don't have explicit bounds, Nelder-Mead is robust but slow.
            # L-BFGS-B is faster. Let's try L-BFGS-B with large bounds.
            res = scipy.optimize.minimize(
                objective, x0, method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.success or res.fun < best_err:
                best_err = res.fun
                x_best = res.x
                U_best = x_best[:dim_u*target_rank].reshape(dim_u, target_rank)
                V_best = x_best[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                W_best = x_best[dim_u*target_rank + dim_v*target_rank :].reshape(dim_w, target_rank)
                
                # Check strict verification
                if _verify_strict(U_best, V_best, W_best, tol=1e-8):
                    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
                    
        except Exception as e:
            pass
            
    return None

def construct():
    """Main entry point. Returns the rank-minimal decomposition."""
    global _START_TIME, _DEADLINE
    
    try:
        # 1. Get the verified rank-39 solution
        result = _construct_rank39()
        
        # 2. Verify it strictly
        if not _verify_strict(result["U"], result["V"], result["W"]):
            raise ValueError("Rank-39 verification failed")
        
        # 3. Try to find a lower-rank solution (Rank 38)
        if time.time() < _DEADLINE - 30.0:
            try:
                better = _try_low_rank(38)
                if better is not None:
                    result = better
            except Exception:
                pass
        
        # 4. Try Rank 37 if time permits and 38 failed?
        # Unlikely to jump 2 ranks with simple search, but worth a shot if fast.
        if time.time() < _DEADLINE - 15.0:
            try:
                better37 = _try_low_rank(37)
                if better37 is not None:
                    result = better37
            except Exception:
                pass

        # 5. Final verification
        if not _verify_strict(result["U"], result["V"], result["W"]):
            # Fallback to the known good solution
            result = _construct_rank39()
            if not _verify_strict(result["U"], result["V"], result["W"]):
                raise RuntimeError("No valid decomposition found")
        
        return result
        
    except Exception:
        # Absolute fallback: return the hardcoded rank-39
        return _construct_rank39()
