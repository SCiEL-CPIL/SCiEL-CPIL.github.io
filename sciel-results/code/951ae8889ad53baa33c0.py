import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) in the appropriate indexing
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (Trace convention: C[k, i]? Or C[i, k]? 
    # Task says: W[k*n+i, r]. 
    # Standard matrix mult C[i, k] = sum_j A[i, j] B[j, k].
    # If W index is k*n + i, and we assume row-major for a p x n matrix?
    # If C is n x p, index is i*p + k.
    # If index is k*n + i, it implies the matrix is stored as p x n?
    # Or it implies the indices are (k, i) where k is the first index.
    # Let's assume the tensor definition in the current program is correct because it's the "seed" logic.
    # Current program: c = k * n + i.
    # Let's stick to this.
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        # Check if rounding to half-integers preserves the decomposition
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    def polish_and_round(U, V, W):
        """High-precision polish: iterate ALS a few more times with tight tolerance,
        then round to nearest half-integer."""
        for _ in range(10):  # Increased polish iterations
            # Update U
            M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, -1)
            T_1 = T.reshape(size_u, size_v * size_w)
            U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=1e-15)
            U = U.T
            
            # Update V
            M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, -1)
            T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
            V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=1e-15)
            V = V.T
            
            # Update W
            M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, -1)
            T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
            W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=1e-15)
            W = W.T
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def prune_small(U, V, W, threshold=1e-4):
        """Set entries with absolute value less than threshold to 0."""
        U[np.abs(U) < threshold] = 0
        V[np.abs(V) < threshold] = 0
        W[np.abs(W) < threshold] = 0
        return U, V, W

    def find_decomposition(rank, max_attempts=30, max_iter=100):
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
                
            # Strategy: Mix of initializations
            if attempt % 3 == 0:
                # Standard random
                U = 0.5 * np.random.randn(size_u, rank)
                V = 0.5 * np.random.randn(size_v, rank)
                W = 0.5 * np.random.randn(size_w, rank)
            elif attempt % 3 == 1:
                # Sparse random (encourages sparsity)
                U = np.random.choice([-0.5, 0, 0.5], size=(size_u, rank))
                V = np.random.choice([-0.5, 0, 0.5], size=(size_v, rank))
                W = np.random.choice([-0.5, 0, 0.5], size=(size_w, rank))
            else:
                # "Block" random: groups of non-zeros
                U = 0.5 * np.random.randn(size_u, rank)
                V = 0.5 * np.random.randn(size_v, rank)
                W = 0.5 * np.random.randn(size_w, rank)
                # Zero out some random entries
                mask_u = np.random.rand(size_u, rank) < 0.3
                mask_v = np.random.rand(size_v, rank) < 0.3
                mask_w = np.random.rand(size_w, rank) < 0.3
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            
            # ALS iterations
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                    
                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
                # Prune small entries to encourage sparsity/snap to lattice
                if it % 10 == 0:
                    U, V, W = prune_small(U, V, W, threshold=1e-3)
                
                # Compute error
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(R - T)
                
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
            
            # Try to polish and round
            polished = polish_and_round(U, V, W)
            if polished is not None:
                U_r, V_r, W_r = polished
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
            # Also check without polishing (in case polish drifted)
            if check_decomposition(U, V, W):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        return None

    best_result = None
    # Try ranks from 23 upward
    # Allocate more time to lower ranks
    rank_configs = [
        (23, 20, 80),
        (24, 20, 80),
        (25, 10, 60),
        (26, 5, 50)
    ]
    
    for rank, attempts, iters in rank_configs:
        if time.time() > deadline - 20:
            break
        res = find_decomposition(rank, max_attempts=attempts, max_iter=iters)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook rank 27
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
