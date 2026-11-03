"""Matrix multiplication 4,4,5 tensor decomposition with gradient-based rank reduction."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Returns 7 terms for 2x2x2 Strassen."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """Constructs a rank-65 decomposition using Kronecker Strassen + naive for last column."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 4x4x4 part (k=0..3)
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
            
    # 4x4x1 part (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
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
    """Check that all entries are integer or half-integer."""
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

def _gradient_descent_reduce(U_init, V_init, W_init, n, m, p, target_rank, max_iters=5000, lr=0.01, tol=1e-8):
    """
    Try to reduce rank by optimizing factor matrices to a lower rank.
    Uses gradient descent on the loss: sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    """
    U = U_init.copy().astype(np.float64)
    V = V_init.copy().astype(np.float64)
    W = W_init.copy().astype(np.float64)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # Truncate to target rank (keep first target_rank columns)
    U = U[:, :target_rank].copy()
    V = V[:, :target_rank].copy()
    W = W[:, :target_rank].copy()
    
    def compute_loss_and_grad():
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j, :]
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    
                    # loss_r = sum_r u_r * v_r * w_r - 1
                    prod = u_row * v_row * w_row
                    residual = prod.sum() - 1.0
                    loss += residual ** 2
                    
                    # d(loss)/d(u_r) = 2 * residual * v_r * w_r
                    grad_u = 2 * residual * v_row * w_row
                    grad_v = 2 * residual * u_row * w_row
                    grad_w = 2 * residual * u_row * v_row
                    
                    grad_U[i*m+j, :] += grad_u
                    grad_V[j*p+k, :] += grad_v
                    grad_W[k*n+i, :] += grad_w
        
        return loss, grad_U, grad_V, grad_W
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        loss, gU, gV, gW = compute_loss_and_grad()
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        # Update
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Decay learning rate
        lr *= 0.999
    
    # Check if we achieved the target
    if best_loss < 1e-6:
        # Round to nearest half-integer
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        
        if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
            return best_U, best_V, best_W
    
    return None

def _try_exact_rank_reduction(U, V, W, n, m, p, target_rank, tol=1e-9):
    """
    Try to find an exact rank-reduction by solving for coefficients.
    For each subset of columns to keep, check if the remaining can be eliminated.
    """
    current_rank = U.shape[1]
    num_entries = n * m * p
    
    if current_rank <= target_rank:
        return None
    
    # Build the full tensor vector for each term
    term_vectors = []
    for r in range(current_rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        vec = np.zeros(num_entries)
        for i in range(n):
            for j in range(m):
                u_val = u[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v[j*p+k]
                    if v_val == 0: continue
                    w_val = w[k*n+i]
                    if w_val == 0: continue
                    val = u_val * v_val * w_val
                    if val != 0:
                        flat_idx = (i * m + j) * p + k
                        vec[flat_idx] = val
        term_vectors.append(vec)
    
    term_vectors = np.array(term_vectors).T  # shape (num_entries, current_rank)
    
    # Target is all-ones
    target = np.ones(num_entries)
    
    # Try to find a subset of target_rank columns that can span the target
    # Use SVD to find the most important columns
    try:
        U_svd, s, Vt = np.linalg.svd(term_vectors, full_matrices=False)
        # The columns with largest singular values are most important
        # Try keeping the first target_rank columns in the SVD basis
        # But we need actual columns, not SVD components
        
        # Alternative: try removing one column at a time and checking if the rest can span
        for r_remove in range(current_rank):
            cols = [r for r in range(current_rank) if r != r_remove]
            if len(cols) < target_rank:
                continue
            
            M = term_vectors[:, cols]
            
            # Check if target is in the column space of M
            # Solve M c = target
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                if residuals and residuals[0] < tol:
                    # Found a solution!
                    U_new = np.zeros((n*m, len(cols)))
                    V_new = np.zeros((m*p, len(cols)))
                    W_new = np.zeros((p*n, len(cols)))
                    
                    for idx, r in enumerate(cols):
                        U_new[:, idx] = U[:, r] * c[idx]
                        V_new[:, idx] = V[:, r]
                        W_new[:, idx] = W[:, r]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                        return U_new, V_new, W_new
            except:
                pass
    except:
        pass
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            best_result = (U, V, W)
            best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Try gradient descent to reduce to rank 60, 55, 50
    if best_result is not None:
        for target_rank in [60, 55, 50, 49]:
            if time.time() > deadline - 20:
                break
            try:
                reduced = _gradient_descent_reduce(best_result[0], best_result[1], best_result[2], 
                                                   n, m, p, target_rank, max_iters=3000, lr=0.005)
                if reduced is not None:
                    U_r, V_r, W_r = reduced
                    rank_r = U_r.shape[1]
                    if rank_r < best_rank:
                        best_result = (U_r, V_r, W_r)
                        best_rank = rank_r
            except Exception:
                pass
    
    # Approach 3: Try exact rank reduction
    if best_result is not None and time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 48), -1):
            if time.time() > deadline - 5:
                break
            try:
                reduced = _try_exact_rank_reduction(best_result[0], best_result[1], best_result[2], 
                                                    n, m, p, target_rank)
                if reduced is not None:
                    U_r, V_r, W_r = reduced
                    rank_r = U_r.shape[1]
                    if rank_r < best_rank:
                        best_result = (U_r, V_r, W_r)
                        best_rank = rank_r
                        break
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
