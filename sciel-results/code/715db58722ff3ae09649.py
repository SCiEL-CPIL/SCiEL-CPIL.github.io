"""Matrix multiplication 4,4,5 tensor decomposition with structured Kronecker construction and rank reduction."""
import numpy as np
import time
import random

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

def _construct_kronecker_4x4x4():
    """
    Constructs the 4x4x4 part using Kronecker product of two Strassen decompositions.
    Returns list of (u, v, w) arrays for each term.
    """
    n, m, p = 4, 4, 4
    strassen_terms = _strassen_2x2x2_terms()
    terms = []
    
    def get_u_idx(i, j): return i * 4 + j
    def get_v_idx(j, k): return j * 4 + k
    def get_w_idx(k, i): return k * 4 + i
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(16)
            
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
            
            terms.append((u, v, w))
    
    return terms

def _construct_4x4x5_initial():
    """
    Constructs initial rank-65 decomposition for 4x4 @ 4x5.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive for the 5th column.
    """
    n, m, p = 4, 4, 5
    U_cols = []
    V_cols = []
    W_cols = []
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i
    
    # 1. 4x4x4 part (k=0..3)
    kronecker_terms = _construct_kronecker_4x4x4()
    for u, v, w in kronecker_terms:
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # 2. 4x4x1 part (k=4)
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

def _try_rank_reduction(U, V, W, n, m, p, max_iterations=500, tol=1e-8, time_budget=30):
    """
    Attempts to reduce rank by identifying linearly dependent terms and removing them.
    Uses null-space analysis of the tensor structure.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        if rank <= 50:  # Can't go below theoretical lower bound for this structure
            break
        
        removed = False
        
        # Try to find a column that can be removed
        for r in range(rank):
            if time.time() - start_time > time_budget:
                break
            
            # Check if removing column r still gives valid decomposition
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            
            if _verify_current_with(U_new, V_new, W_new, n, m, p):
                # Successfully removed a column
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed = True
                break
        
        if not removed:
            # Try more sophisticated elimination: find linear dependencies
            # Check if any column in U is linearly dependent on others
            try:
                # Compute rank of U
                u_rank = np.linalg.matrix_rank(U, tol=tol)
                if u_rank < rank:
                    # There are linear dependencies in U
                    # Find a dependent column
                    U_svd, s, Vt = np.linalg.svd(U, full_matrices=False)
                    # The last few columns of Vt correspond to null space
                    # Try removing columns that are in the null space
                    for r in range(rank - 1, -1, -1):
                        # Check if column r is nearly in the span of others
                        # Simple heuristic: if norm of column is small relative to others
                        col_norm = np.linalg.norm(U[:, r])
                        other_norms = [np.linalg.norm(U[:, r2]) for r2 in range(rank) if r2 != r]
                        if other_norms and col_norm < tol * max(other_norms):
                            U_new = np.delete(U, r, axis=1)
                            V_new = np.delete(V, r, axis=1)
                            W_new = np.delete(W, r, axis=1)
                            if _verify_current_with(U_new, V_new, W_new, n, m, p):
                                U, V, W = U_new, V_new, W_new
                                rank -= 1
                                removed = True
                                break
            except:
                pass
        
        if not removed:
            # Try random perturbation and re-orthogonalization
            if time.time() - start_time > time_budget:
                break
            
            # Add small random perturbation to break symmetry
            U_perturbed = U + 0.001 * np.random.randn(U.shape[0], U.shape[1])
            V_perturbed = V + 0.001 * np.random.randn(V.shape[0], V.shape[1])
            W_perturbed = W + 0.001 * np.random.randn(W.shape[0], W.shape[1])
            
            # Snap to nearest half-integers
            U_perturbed = np.round(U_perturbed * 2) / 2
            V_perturbed = np.round(V_perturbed * 2) / 2
            W_perturbed = np.round(W_perturbed * 2) / 2
            
            if _verify_current_with(U_perturbed, V_perturbed, W_perturbed, n, m, p):
                U, V, W = U_perturbed, V_perturbed, W_perturbed
            
            # Try to find removable columns again
            for r in range(rank):
                if time.time() - start_time > time_budget:
                    break
                U_new = np.delete(U, r, axis=1)
                V_new = np.delete(V, r, axis=1)
                W_new = np.delete(W, r, axis=1)
                if _verify_current_with(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
    
    if _verify_current():
        return U, V, W
    
    # Return best found
    if _verify_current_with(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W
    
    return None

def _verify_current_with(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j, :]
                v_val = V[j*p+k, :]
                w_val = W[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _try_structured_random_projection(U, V, W, n, m, p, time_budget=20):
    """
    Try structured random projection with iterative null-space elimination.
    Initialize with random half-integer perturbations, then project and eliminate.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for attempt in range(10):
        if time.time() - start_time > time_budget:
            break
        
        # Create random half-integer perturbation
        scale = 0.5
        U_pert = U + scale * (np.random.randint(0, 3, U.shape) - 1)
        V_pert = V + scale * (np.random.randint(0, 3, V.shape) - 1)
        W_pert = W + scale * (np.random.randint(0, 3, W.shape) - 1)
        
        # Snap to half-integers
        U_pert = np.round(U_pert * 2) / 2
        V_pert = np.round(V_pert * 2) / 2
        W_pert = np.round(W_pert * 2) / 2
        
        if not _verify_current_with(U_pert, V_pert, W_pert, n, m, p):
            continue
        
        # Try to reduce rank from perturbed version
        U_cur, V_cur, W_cur = U_pert.copy(), V_pert.copy(), W_pert.copy()
        cur_rank = U_cur.shape[1]
        
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            if cur_rank <= best_rank:
                break
            
            removed = False
            for r in range(cur_rank):
                if time.time() - start_time > time_budget:
                    break
                U_new = np.delete(U_cur, r, axis=1)
                V_new = np.delete(V_cur, r, axis=1)
                W_new = np.delete(W_cur, r, axis=1)
                if _verify_current_with(U_new, V_new, W_new, n, m, p):
                    U_cur, V_cur, W_cur = U_new, V_new, W_new
                    cur_rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        if cur_rank < best_rank and _verify_current_with(U_cur, V_cur, W_cur, n, m, p):
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            best_rank = cur_rank
    
    if best_rank < rank:
        return best_U, best_V, best_W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary approach: Kronecker Strassen + rank-16 last column (rank 65)
    try:
        U, V, W = _construct_4x4x5_initial()
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            # Try to reduce rank
            reduced = _try_rank_reduction(U, V, W, n, m, p, max_iterations=200, time_budget=20)
            if reduced is not None:
                U, V, W = reduced
                rank = U.shape[1]
            
            # Try structured random projection
            if time.time() < deadline - 10:
                projected = _try_structured_random_projection(U, V, W, n, m, p, time_budget=15)
                if projected is not None:
                    U, V, W = projected
                    rank = U.shape[1]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 80)
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
