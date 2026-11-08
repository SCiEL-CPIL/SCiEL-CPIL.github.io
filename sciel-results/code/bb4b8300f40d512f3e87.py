import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = 2, 3, 4
    dim_u, dim_v, dim_w = n * m, m * p, p * n  # 6, 12, 8

    # Build the multiplication tensor T[i, j, k] = 1 if A[i]*B[j] contributes to C[k]
    # i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # A[i]: row = i // m, col = i % m
    # B[j]: row = j // p, col = j % p
    # C[k]: row = k % n, col = k // n  (trace convention W[k*n+i, r] -> k is col, i is row)
    # T[i, j, k] = 1 if (i % m) == (j // p) and (i // m) == (k % n) and (j % p) == (k // n)

    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(dim_u):
        ai, aj = i // m, i % m  # A row, A col
        for j in range(dim_v):
            bj, bj2 = j // p, j % p  # B row, B col
            if aj == bj:
                for k in range(dim_w):
                    ck, ck2 = k % n, k // n  # C row, C col
                    if ai == ck and bj2 == ck2:
                        T[i, j, k] = 1.0

    # Try to find a low-rank decomposition using linear algebra
    # The tensor T can be viewed as a set of matrix slices.
    # For each k, T[:, :, k] is a dim_u x dim_v matrix.
    # If T has rank r, then all slices lie in a rank-r subspace.
    
    # Strategy: Use the fact that the multiplication tensor for (2,3,4) has known structure.
    # The known optimal rank for M_{2,3,4} is 20 (from literature).
    # Let's try to verify rank 20 with a direct construction.
    
    # Actually, let me try a different approach: use the known Strassen-like decomposition
    # but try to find if we can do better with a search.
    
    # For now, let's use the rank-22 solution that we know works, but try to
    # verify if rank 20 is achievable with a linear algebra approach.
    
    # Let me try to compute the tensor rank by checking if the tensor can be
    # decomposed with fewer terms.
    
    # A practical approach: try rank 20 with a randomized initialization and
    # iterative refinement (alternating least squares style).
    
    def try_decompose(rank, n_iter=200, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(dim_u, rank)
        V = rng.randn(dim_v, rank)
        W = rng.randn(dim_w, rank)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            # Update W: for each r, W[:, r] = argmin ||T[:,:,r] - (U[:,r] outer V[:,r])||
            # T[:,:,r] should approximate U[:,r] @ V[:,r].T
            # So W[:,r] is the projection of the residual onto the "correct" direction
            # Actually, in tensor decomposition, we have:
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed U, V: W[k,r] = sum_{i,j} T[i,j,k] * U[i,r] * V[j,r] / (sum_{i,j} U[i,r]^2 * V[j,r]^2)
            # But this is for the CP decomposition where each term is a rank-1 tensor.
            
            # Let's do alternating minimization:
            # 1. Fix U, V, update W
            # 2. Fix U, W, update V  
            # 3. Fix V, W, update U
            
            # Update W
            for r in range(rank):
                # T_approx[i,j,k] = U[i,r] * V[j,r] * W[k,r]
                # We want to minimize ||T - sum_r' U[:,r'] V[:,r']^T W[k,r']||
                # For fixed U, V: W[k,r] = <T[:,:,k], U[:,r] V[:,r]^T> / ||U[:,r] V[:,r]^T||^2
                # But T is 3D, so:
                # W[k,r] = sum_{i,j} T[i,j,k] * U[i,r] * V[j,r] / sum_{i,j} U[i,r]^2 * V[j,r]^2
                num = np.zeros(dim_w)
                den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
                if abs(den) < 1e-15:
                    continue
                for k in range(dim_w):
                    num[k] = np.sum(T[:, :, k] * np.outer(U[:, r], V[:, r]))
                W[:, r] = num / den
            
            # Update V
            for r in range(rank):
                num = np.zeros(dim_v)
                den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
                if abs(den) < 1e-15:
                    continue
                for j in range(dim_v):
                    num[j] = np.sum(T[:, j, :] * np.outer(U[:, r], W[:, r]))
                V[:, r] = num / den
            
            # Update U
            for r in range(rank):
                num = np.zeros(dim_u)
                den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
                if abs(den) < 1e-15:
                    continue
                for i in range(dim_u):
                    num[i] = np.sum(T[i, :, :] * np.outer(V[:, r], W[:, r]))
                U[:, r] = num / den
            
            # Compute error
            T_approx = np.zeros_like(T)
            for r in range(rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T - T_approx)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-8:
            return best_U, best_V, best_W, best_err
        return None
    
    # Try decreasing ranks from 20
    for rank in [20, 21]:
        if time.time() > deadline - 20:
            break
        result = try_decompose(rank, n_iter=100)
        if result is not None:
            U, V, W, err = result
            # Verify
            T_approx = np.zeros_like(T)
            for r in range(rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            if np.allclose(T, T_approx, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: use the known rank-22 construction
    rank = 22
    U = np.zeros((dim_u, rank), dtype=float)
    V = np.zeros((dim_v, rank), dtype=float)
    W = np.zeros((dim_w, rank), dtype=float)
    
    def add_term(col_idx, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            U[idx, col_idx] += val
        for idx, val in v_vals.items():
            V[idx, col_idx] += val
        for idx, val in w_vals.items():
            W[idx, col_idx] += val
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_term(0, {0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    add_term(1, {3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    add_term(2, {0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    add_term(3, {4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    add_term(4, {0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    add_term(5, {3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_term(6, {1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_term(7, {0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    add_term(8, {3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    add_term(9, {0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    add_term(10, {4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    add_term(11, {0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    add_term(12, {3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add_term(13, {1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})
    
    # Block 3: A[:, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    add_term(14, {2: 1}, {8: 1}, {0: 1})
    add_term(15, {2: 1}, {9: 1}, {2: 1})
    add_term(16, {2: 1}, {10: 1}, {4: 1})
    add_term(17, {2: 1}, {11: 1}, {6: 1})
    add_term(18, {5: 1}, {8: 1}, {1: 1})
    add_term(19, {5: 1}, {9: 1}, {3: 1})
    add_term(20, {5: 1}, {10: 1}, {5: 1})
    add_term(21, {5: 1}, {11: 1}, {7: 1})
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
