"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Check if the decomposition satisfies all n*m*p tensor equations."""
    rank = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-8:
                    return False
    return True

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _rank_11_solution():
    """Return the known rank-11 decomposition (Strassen + schoolbook for col 2)."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen part (terms 0-6) for 2x2 * 2x2 (cols 0,1 of B)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[4,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    U[0,2]=1; V[1,2]=1; V[4,2]=-1
    U[3,3]=1; V[3,3]=1; V[0,3]=-1
    U[0,4]=1; U[1,4]=1; V[4,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    U[1,6]=1; U[3,6]=-1; V[3,6]=1; V[4,6]=1
    
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    W[2,2]=1; W[2,4]=1
    W[1,1]=1; W[1,3]=1
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    
    # Schoolbook for col 2 (terms 7-10)
    U[0,7]=1; V[2,7]=1; W[4,7]=1
    U[1,8]=1; V[5,8]=1; W[4,8]=1
    U[2,9]=1; V[2,9]=1; W[5,9]=1
    U[3,10]=1; V[5,10]=1; W[5,10]=1
    
    return U, V, W

def _try_rank_10(deadline):
    """Try to find a rank-10 decomposition via structured optimization."""
    n, m, p = DIMENSIONS
    rank = 10
    
    U11, V11, W11 = _rank_11_solution()
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    def objective(params):
        U_flat = params[0:4*rank]
        V_flat = params[4*rank:4*rank+6*rank]
        W_flat = params[4*rank+6*rank:4*rank+6*rank+6*rank]
        
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    obj += (val - 1)**2
        return obj
    
    # Try multiple initializations
    inits = []
    
    # Init 1: drop last column, scale remaining
    inits.append((U11[:, :10].copy().flatten(), V11[:, :10].copy().flatten(), W11[:, :10].copy().flatten()))
    
    # Init 2-11: drop each column
    for skip in range(11):
        if time.time() > deadline:
            break
        U_s = np.delete(U11, skip, axis=1)
        V_s = np.delete(V11, skip, axis=1)
        W_s = np.delete(W11, skip, axis=1)
        inits.append((U_s.flatten(), V_s.flatten(), W_s.flatten()))
    
    best_result = None
    best_obj = float('inf')
    
    for init_idx, (uf, vf, wf) in enumerate(inits):
        if time.time() > deadline:
            break
        
        params0 = np.concatenate([uf, vf, wf])
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 300, 'gtol': 1e-14, 'ftol': 1e-16})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is not None and best_obj < 1e-10:
        U_flat = best_result[0:4*rank]
        V_flat = best_result[4*rank:4*rank+6*rank]
        W_flat = best_result[4*rank+6*rank:4*rank+6*rank+6*rank]
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_10_randomized(deadline):
    """Try randomized search for rank-10 decomposition."""
    n, m, p = DIMENSIONS
    rank = 10
    
    U11, V11, W11 = _rank_11_solution()
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    def objective(params):
        U_flat = params[0:4*rank]
        V_flat = params[4*rank:4*rank+6*rank]
        W_flat = params[4*rank+6*rank:4*rank+6*rank+6*rank]
        
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    obj += (val - 1)**2
        return obj
    
    # Random perturbations of the rank-11 solution
    rng = np.random.default_rng(42)
    best_result = None
    best_obj = float('inf')
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Randomly perturb the rank-11 solution
        U_p = U11[:, :10].copy()
        V_p = V11[:, :10].copy()
        W_p = W11[:, :10].copy()
        
        # Add small random perturbations
        U_p += rng.standard_normal(U_p.shape) * 0.1
        V_p += rng.standard_normal(V_p.shape) * 0.1
        W_p += rng.standard_normal(W_p.shape) * 0.1
        
        params0 = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B',
                             options={'maxiter': 100, 'gtol': 1e-14, 'ftol': 1e-16})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is not None and best_obj < 1e-10:
        U_flat = best_result[0:4*rank]
        V_flat = best_result[4*rank:4*rank+6*rank]
        W_flat = best_result[4*rank+6*rank:4*rank+6*rank+6*rank]
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First try rank 10 with structured search
    result_10 = _try_rank_10(deadline)
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Try randomized rank 10 search
    if time.time() < deadline - 10:
        result_10r = _try_rank_10_randomized(deadline)
        if result_10r is not None:
            U, V, W = result_10r
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 11 (Strassen + schoolbook)
    U, V, W = _rank_11_solution()
    
    # Verify the decomposition
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
