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

def _riemannian_descent(U_init, V_init, W_init, n, m, p, max_iterations=200, tol=1e-8, max_time=30):
    """
    Attempts to reduce rank using Riemannian gradient descent on the Stiefel manifold.
    """
    start_time = time.time()
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    rank = U.shape[1]
    
    # Project to Stiefel manifold (orthonormal columns)
    def _stiefel_projection(X):
        Q, R = np.linalg.qr(X)
        if np.any(np.diag(R) < 0):
            Q *= np.sign(np.diag(R))
        return Q
    
    U = _stiefel_projection(U)
    V = _stiefel_projection(V)
    W = _stiefel_projection(W)
    
    # Initial residual
    def _compute_residual():
        res = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.linalg.norm(res - np.ones((n, m, p)))
    
    current_res = _compute_residual()
    if current_res < tol:
        return U, V, W
    
    # Riemannian gradient descent
    lr = 0.01
    for it in range(max_iterations):
        if time.time() > start_time + max_time:
            break
        
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        target = np.ones((n, m, p))
        res = np.zeros((n, m, p))
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        res = res - target
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U[i*m+j, r] += res[i, j, k] * v[j, k] * w[k, i]
                        grad_V[j*p+k, r] += res[i, j, k] * u[i, j] * w[k, i]
                        grad_W[k*n+i, r] += res[i, j, k] * u[i, j] * v[j, k]
        
        # Project gradient to tangent space
        def _project_tangent(X, G):
            return G - X @ (X.T @ G)
        
        grad_U = _project_tangent(U, grad_U)
        grad_V = _project_tangent(V, grad_V)
        grad_W = _project_tangent(W, grad_W)
        
        # Update
        U = U - lr * grad_U
        V = V - lr * grad_V
        W = W - lr * grad_W
        
        # Re-project to Stiefel manifold
        U = _stiefel_projection(U)
        V = _stiefel_projection(V)
        W = _stiefel_projection(W)
        
        # Check convergence
        new_res = _compute_residual()
        if new_res < tol:
            return U, V, W
        
        if new_res > current_res * 1.1:
            lr *= 0.5
        else:
            current_res = new_res
    
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
    
    # Approach 2: Riemannian gradient descent
    if best_result is not None and time.time() < deadline - 30:
        try:
            reduced = _riemannian_descent(best_result[0], best_result[1], best_result[2], n, m, p, max_time=45)
            if reduced is not None:
                U_r, V_r, W_r = reduced
                rank_r = U_r.shape[1]
                if rank_r < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = rank_r
        except Exception:
            pass
    
    # Approach 4: Try to reduce rank by removing terms if Riemannian descent succeeded
    if best_result is not None and time.time() < deadline - 5:
        try:
            U_r, V_r, W_r = best_result
            current_rank = U_r.shape[1]
            
            # Try to remove one term at a time
            for r in range(current_rank):
                if time.time() > deadline - 5:
                    break
                
                remaining = [x for x in range(current_rank) if x != r]
                if not remaining:
                    break
                
                M_remaining = _compute_term_vectors(U_r, V_r, W_r, n, m, p)[:, remaining]
                target_flat = np.ones((n, m, p)).flatten()
                
                try:
                    c, residuals, _, _ = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
                    
                    if residuals and len(residuals) > 0 and residuals[0] < 1e-9:
                        U_new = np.zeros((n*m, len(remaining)))
                        V_new = np.zeros((m*p, len(remaining)))
                        W_new = np.zeros((p*n, len(remaining)))
                        
                        for idx, r_idx in enumerate(remaining):
                            U_new[:, idx] = U_r[:, r_idx] * c[idx]
                            V_new[:, idx] = V_r[:, r_idx]
                            W_new[:, idx] = W_r[:, r_idx]
                        
                        if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                            U_r, V_r, W_r = U_new, V_new, W_new
                            current_rank = len(remaining)
                            best_rank = current_rank
                            best_result = (U_r, V_r, W_r)
                            break
                except Exception:
                    pass
        except Exception:
            pass
    
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
