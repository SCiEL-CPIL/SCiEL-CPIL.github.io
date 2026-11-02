import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    start_time = time.time()
    deadline = 145.0
    
    # Build target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    def verify_solution(U, V, W):
        """Verify exact decomposition with half-integer entries."""
        # Check entries are half-integers
        for M in [U, V, W]:
            if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
                return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        max_diff = np.max(np.abs(T - target))
        return max_diff < 1e-9

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_structured_rank(r, max_restarts=50):
        """Try to find a rank-r decomposition using structured patterns."""
        start = time.time()
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            if time.time() - start > 2.0:
                break
            
            np.random.seed(123 + r * 1000 + restart)
            
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            
            # Use half-integer values: -1, -0.5, 0, 0.5, 1
            half_ints = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            for col in range(r):
                # Sparse initialization
                n_nonzero_u = np.random.randint(1, 4)
                n_nonzero_v = np.random.randint(1, 4)
                n_nonzero_w = np.random.randint(1, 4)
                
                u_idx = np.random.choice(dim_u, size=n_nonzero_u, replace=False)
                for idx in u_idx:
                    U[idx, col] = np.random.choice(half_ints)
                
                v_idx = np.random.choice(dim_v, size=n_nonzero_v, replace=False)
                for idx in v_idx:
                    V[idx, col] = np.random.choice(half_ints)
                
                w_idx = np.random.choice(dim_w, size=n_nonzero_w, replace=False)
                for idx in w_idx:
                    W[idx, col] = np.random.choice(half_ints)
            
            # Local search: try flipping individual entries
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_err = np.sum((np.einsum('ar,br,cr->abc', best_U, best_V, best_W) - target)**2)
            
            for iteration in range(200):
                if time.time() > deadline:
                    return None
                
                # Pick a random matrix, row, and column to modify
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row = np.random.randint(dim_u)
                    col = np.random.randint(r)
                    old_val = U[row, col]
                    new_val = np.random.choice(half_ints)
                    if new_val == old_val:
                        continue
                    U[row, col] = new_val
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.sum((T - target)**2)
                    if err < best_err:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_err = err
                    else:
                        U[row, col] = old_val
                elif mat_choice == 1:
                    row = np.random.randint(dim_v)
                    col = np.random.randint(r)
                    old_val = V[row, col]
                    new_val = np.random.choice(half_ints)
                    if new_val == old_val:
                        continue
                    V[row, col] = new_val
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.sum((T - target)**2)
                    if err < best_err:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_err = err
                    else:
                        V[row, col] = old_val
                else:
                    row = np.random.randint(dim_w)
                    col = np.random.randint(r)
                    old_val = W[row, col]
                    new_val = np.random.choice(half_ints)
                    if new_val == old_val:
                        continue
                    W[row, col] = new_val
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.sum((T - target)**2)
                    if err < best_err:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_err = err
                    else:
                        W[row, col] = old_val
                
                if best_err < 1e-10:
                    if verify_solution(best_U, best_V, best_W):
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            if best_err < 1e-8:
                if verify_solution(best_U, best_V, best_W):
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return None

    # Try ranks from 45 to 49
    for r in [45, 46, 47, 48, 49]:
        if time.time() > deadline:
            break
        result = try_structured_rank(r, max_restarts=30)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
