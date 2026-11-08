"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, trials=200, seed=42):
    """Verify that the decomposition exactly reconstructs matrix multiplication."""
    rng = np.random.RandomState(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        
        C_recon = np.zeros((n, p))
        for c in range(rank):
            u_c = U_arr[:, c]
            v_c = V_arr[:, c]
            w_c = W_arr[:, c]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
                    C_recon[i,k] += s
        
        if not np.allclose(C, C_recon, atol=1e-9):
            return False
    return True

def _verify_tensor_exact(U, V, W, n, m, p):
    """Check that the tensor entries are exactly 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_optimization(r, n, m, p, deadline, max_iters=5000):
    """Try to find a decomposition of rank r using optimization."""
    # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
    # Total variables: r*(n*m + m*p + p*n)
    total_vars = r * (n*m + m*p + p*n)
    
    if total_vars > 50000:  # Too many variables for practical optimization
        return None
    
    rng = np.random.RandomState(123 + r)
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        nm = n*m
        mp = m*p
        pn = p*n
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        return U, V, W
    
    def _objective(x):
        if time.time() > deadline:
            return 1e10
        U, V, W = _unflatten(x)
        # Compute tensor residual
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        return err
    
    # Try multiple random initializations
    best_result = None
    n_attempts = 5
    time_per_attempt = (deadline - time.time()) / max(n_attempts, 1)
    
    for attempt in range(n_attempts):
        if time.time() > deadline - 2:
            break
        
        # Random initialization with small values
        x0 = rng.randn(total_vars) * 0.5
        
        try:
            result = minimize(
                _objective, x0, method='L-BFGS-B', 
                options={'maxiter': max_iters, 'maxfun': max_iters * 3, 'time_limit': max(time_per_attempt, 1)}
            )
            
            if result.fun < 1e-8:
                U, V, W = _unflatten(result.x)
                # Check if we can round to integers/half-integers
                # Try rounding to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_tensor_exact(U_rounded, V_rounded, W_rounded, n, m, p):
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    if best_result is None or r < best_result[0]:
                        best_result = (r, U_list, V_list, W_list)
                        return best_result
                elif _verify_tensor_exact(U, V, W, n, m, p):
                    # Check if already close to half-integers
                    if np.allclose(U, np.round(U*2)/2, atol=1e-6) and \
                       np.allclose(V, np.round(V*2)/2, atol=1e-6) and \
                       np.allclose(W, np.round(W*2)/2, atol=1e-6):
                        U_list = (np.round(U*2)/2).tolist()
                        V_list = (np.round(V*2)/2).tolist()
                        W_list = (np.round(W*2)/2).tolist()
                        if best_result is None or r < best_result[0]:
                            best_result = (r, U_list, V_list, W_list)
                            return best_result
        except Exception:
            continue
    
    return best_result

def _try_rank_greedy(r, n, m, p, deadline):
    """Try a greedy construction for rank r."""
    # For rank r, try to build the decomposition greedily
    # Start with zero matrices and add one term at a time
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # For each term c, we want to maximize the reduction in residual
    # This is complex, so let's try a simpler structured approach
    
    # Construction: For n=2, m=3, p=5
    # Use the fact that we can write the MM as a sum of rank-1 tensors
    
    # Try: for c in range(r), assign U[:,c], V[:,c], W[:,c] to cover certain (i,j,k) triples
    # This is essentially finding a covering with rank-1 tensors
    
    # Simple greedy: for each (i,j,k), if not yet covered, assign a new term
    # But this gives rank = n*m*p = 30
    
    # Better: try to share terms
    # For fixed j, the terms involving A[i,j] and B[j,k] can be shared across i,k
    
    # Let's try: for each j (3 choices), use n*p = 10 terms, but try to merge
    # Total: 3*10 = 30, can we do better?
    
    # Alternative: use the structure that for each (i,k), we need sum_j A[i,j]*B[j,k]
    # If we precompute for each j: vector v_j of length n*p where v_j[i*p+k] = A[i,j]*B[j,k]
    # Then C = sum_j v_j (as a matrix)
    # But v_j depends on A and B, so this doesn't directly help
    
    # Let's try a different known construction
    # For M_{2,3,5}, try rank 12
    if r >= 12:
        # Construction: use 10 terms for "diagonal" and 2 for "off-diagonal" corrections
        # This is heuristic, so let's skip and rely on optimization
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin
    
    # Try ranks from low to high using optimization
    for r in range(10, 25):
        if time.time() > deadline - 10:
            break
        result = _try_rank_optimization(r, n, m, p, deadline)
        if result is not None:
            r_found, U_list, V_list, W_list = result
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, trials=100):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
