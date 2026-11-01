import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the Strassen rank-28 solution
    U, V, W = _build_strassen_28()
    
    # Try to reduce rank via linear dependency elimination
    result = _try_reduce_rank(U, V, W, deadline)
    if result is not None:
        return result
    
    # Fallback to rank 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    # Compute T_hat[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (4, 16, rank)
    T_hat = np.einsum('abr,cr->abc', UV, W)  # (4, 16, 16)
    
    # Check non-zero entries: T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i)
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                if abs(T_hat[a, b, c] - 1) > tol:
                    return False
    
    # Check zero entries
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                i = a // 2
                j = a % 2
                j2 = b // 8
                k = b % 8
                k2 = c // 2
                i2 = c % 2
                if not (i == i2 and j == j2 and k == k2):
                    if abs(T_hat[a, b, c]) > tol:
                        return False
    
    return True

def _try_reduce_rank(U, V, W, deadline):
    """Try to find a lower-rank decomposition by checking linear dependencies."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    total_dim = num_a * num_b * num_c  # 1024
    
    rank = U.shape[1]
    
    # Flatten each rank-1 term to a vector
    # Term r: u_r ⊗ v_r ⊗ w_r, flattened
    terms = []
    for r in range(rank):
        u_r = U[:, r]  # (4,)
        v_r = V[:, r]  # (16,)
        w_r = W[:, r]  # (16,)
        tensor_r = np.einsum('a,b,c->abc', u_r, v_r, w_r)  # (4,16,16)
        terms.append(tensor_r.flatten())
    
    terms = np.array(terms)  # (rank, 1024)
    
    # Check rank of the term matrix
    # If rank < 28, there's a linear dependency
    term_rank = np.linalg.matrix_rank(terms, tol=1e-10)
    
    if term_rank < rank:
        # Find null vector
        _, S, Vt = np.linalg.svd(terms, full_matrices=False)
        # Null vectors correspond to smallest singular values
        # Find indices where S < tol
        null_mask = S < 1e-10
        if np.any(null_mask):
            # Get the null space basis
            null_vecs = Vt[null_mask, :]  # each is a vector in R^rank
            # Try to eliminate one term at a time
            for nv in null_vecs:
                # nv[r] gives the coefficient for term r
                # If we can express term j as a combination of others, remove it
                # Find the term with largest |nv[r]| to eliminate
                j = np.argmax(np.abs(nv))
                if abs(nv[j]) > 1e-10:
                    # New terms: for r != j, term'_r = term_r - (nv[r]/nv[j]) * term_j
                    # But we need to keep U, V, W structure
                    # This is complex; instead, try a different approach
                    pass
    
    # Alternative: try random search for lower ranks
    for target_rank in [24, 25, 26, 27]:
        if time.time() > deadline - 5:
            break
        result = _search_low_rank(target_rank, deadline - time.time() - 2)
        if result is not None:
            return result
    
    return None

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using random restarts."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    # Non-zero indices
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
    
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart)
        U = rng.integers(-2, 3, size=(num_a, target_rank)).astype(float)
        V = rng.integers(-2, 3, size=(num_b, target_rank)).astype(float)
        W = rng.integers(-2, 3, size=(num_c, target_rank)).astype(float)
        
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_hat = np.einsum('abr,cr->abc', UV, W)
        
        # Check non-zero entries
        max_err = 0
        for a, b, c in nonzero_indices:
            err = abs(T_hat[a, b, c] - 1)
            max_err = max(max_err, err)
        
        if max_err < 1e-6:
            # Check zero entries (sample)
            all_zero_ok = True
            for a in range(num_a):
                for b in range(num_b):
                    for c in range(num_c):
                        if (a, b, c) not in nonzero_indices:
                            if abs(T_hat[a, b, c]) > 1e-6:
                                all_zero_ok = False
                                break
                    if not all_zero_ok:
                        break
                if not all_zero_ok:
                    break
            
            if all_zero_ok:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
