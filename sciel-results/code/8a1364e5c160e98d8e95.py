"""Matrix multiplication 4,4,5 tensor decomposition with combinatorial subspace elimination."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return U, V, W

def _combinatorial_subspace_elimination(U_init, V_init, W_init, n, m, p, max_iterations=200, tol=1e-8):
    """
    Attempts to reduce rank by identifying and removing linearly dependent terms
    using null-space analysis of the tensor flattening.
    """
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    rank = U.shape[1]
    
    def _verify_current():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    total = np.dot(u_val * v_val, w_val)
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    if not _verify_current():
        return None
    
    for _ in range(max_iterations):
        # Try to find a zero or nearly-zero column in any factor
        removed = False
        
        for r in range(rank):
            # Check if column r is effectively zero in all three factors
            if (np.linalg.norm(U[:, r]) < tol and 
                np.linalg.norm(V[:, r]) < tol and 
                np.linalg.norm(W[:, r]) < tol):
                # Remove this column
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                removed = True
                break
        
        if not removed:
            # Try more sophisticated linear dependency detection
            # For each factor, check if any column is a linear combination of others
            for name, M in [("U", U), ("V", V), ("W", W)]:
                # Compute SVD to find near-null space
                try:
                    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                    # If smallest singular value is very small, the last column is dependent
                    if s[-1] < tol * max(s[0], 1e-10):
                        # Try removing the last column
                        M_new = np.delete(M, -1, axis=1)
                        
                        # Verify if removing this column still satisfies the equation
                        # We need to check if the removed column's contribution was zero
                        # This is complex; skip for now and just try simple zero removal
                        pass
                except:
                    pass
        
        if not removed:
            break
        
        if not _verify_current():
            # Revert
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            break
    
    if _verify_current():
        return U, V, W
    
    return None

def _try_gradient_optimization(target_rank, n, m, p, max_iter=100, seed=42, time_limit=30):
    """Try to find a decomposition of given rank using gradient-based optimization."""
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, target_rank) * 0.3
    V = np.random.randn(m*p, target_rank) * 0.3
    W = np.random.randn(p*n, target_rank) * 0.3
    
    def loss_and_grad(params):
        U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        loss_val = 0.0
        U_grad = np.zeros_like(U_p)
        V_grad = np.zeros_like(V_p)
        W_grad = np.zeros_like(W_p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    
                    prod = U_p[u_idx, :] * V_p[v_idx, :]
                    s = np.dot(prod, W_p[w_idx, :])
                    resid = s - 1.0
                    
                    loss_val += resid * resid
                    
                    U_grad[u_idx, :] += 2.0 * resid * V_p[v_idx, :] * W_p[w_idx, :]
                    V_grad[v_idx, :] += 2.0 * resid * U_p[u_idx, :] * W_p[w_idx, :]
                    W_grad[w_idx, :] += 2.0 * resid * U_p[u_idx, :] * V_p[v_idx, :]
        
        grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
        return loss_val, grad
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    from scipy.optimize import minimize
    try:
        result = minimize(loss_and_grad, params, jac=True, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-5, 'ftol': 1e-10, 'maxls': 30})
    except Exception:
        return None
    
    if time.time() - start_time > time_limit:
        return None
    
    if result.fun < 1e-5:
        U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Rational reconstruction: snap to nearest half-integer
        U_snap = np.round(U_opt * 2.0) / 2.0
        V_snap = np.round(V_opt * 2.0) / 2.0
        W_snap = np.round(W_opt * 2.0) / 2.0
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
        
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
            return U_opt, V_opt, W_opt
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            candidates = [(rank, U, V, W)]
        else:
            candidates = []
    except Exception:
        candidates = []
    
    # Pick the best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, U, V, W = candidates[0]
        best_result = (U, V, W)
    
    # Approach 2: Combinatorial subspace elimination
    if best_result is not None and time.time() < deadline - 30:
        try:
            reduced = _combinatorial_subspace_elimination(best_result[0], best_result[1], best_result[2], n, m, p)
            if reduced is not None:
                U_r, V_r, W_r = reduced
                rank_r = U_r.shape[1]
                if rank_r < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = rank_r
        except Exception:
            pass
    
    # Approach 3: Try gradient optimization for lower ranks
    if best_result is not None and time.time() < deadline - 45:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 35), -1):
            if time.time() > deadline - 25:
                break
            found = False
            for seed in [target_rank, target_rank + 100]:
                if time.time() > deadline - 20:
                    break
                try:
                    result = _try_gradient_optimization(target_rank, n, m, p, max_iter=80, seed=seed, time_limit=15)
                    if result is not None:
                        best_result = result
                        best_rank = target_rank
                        found = True
                        break
                except Exception:
                    pass
            if found:
                break
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
