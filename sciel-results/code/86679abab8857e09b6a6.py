import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose target."""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank_sparse_als(r, num_restarts=100):
        """
        Try to find a rank r decomposition using Sparse ALS with half-integer snapping.
        """
        if time.time() > deadline:
            return None

        best_U, best_V, best_W = None, None, None
        best_err = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            # Sparse random initialization with half-integers
            # We want many zeros to help the sparsity of the rank decomposition
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Randomly populate a small number of entries with half-integers
            # Heuristic: ~10-20% non-zero
            for _ in range(r):
                # Pick a few random indices
                idx_u = np.random.choice(nm, size=nm//5, replace=False)
                idx_v = np.random.choice(mp, size=mp//5, replace=False)
                idx_w = np.random.choice(pn, size=pn//5, replace=False)
                
                U[idx_u, _] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=len(idx_u))
                V[idx_v, _] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=len(idx_v))
                W[idx_w, _] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=len(idx_w))

            # ALS Optimization
            for it in range(50):
                if time.time() > deadline - 5:
                    break

                # Update U
                # T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
                # For fixed a, sum_k U[a,k] (V[b,k] W[c,k]) = T[a,b,c]
                # Let M_U[b,c,k] = V[b,k] W[c,k]
                # Vectorize: For each a, solve M_U_flat @ U[a,:] = T[a,:,:].flat
                # M_U_flat shape (mp*pn, r)
                # We can compute this efficiently
                # V[:, k] * W[:, k] -> outer product? No, element-wise for each k?
                # M_U[:, k] = V[:, k] * W[:, k] is not quite right for the flattening order.
                # T is indexed [a, b, c].
                # Flatten b, c to index x = b*pn + c.
                # M_U[x, k] = V[b, k] * W[c, k] where b = x // pn, c = x % pn.
                
                # Efficient calculation of M_U, M_V, M_W
                # M_U: (mp*pn, r)
                # For each k, M_U[:, k] = (V[:, k] * W[:, k]).reshape(-1) ?? No.
                # V is (mp, r), W is (pn, r).
                # We need the outer product of V[:, k] and W[:, k] flattened?
                # Yes, if we flatten b and c in row-major order.
                # V[:, k] has length mp. W[:, k] has length pn.
                # np.kron(V[:, k], W[:, k])? No, that's not right if we just want element-wise product in the flattened tensor space?
                # Wait. T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k].
                # If we fix a, we want to find U[a, :] such that:
                # sum_k U[a, k] * (V[b, k] * W[c, k]) = T[a, b, c]
                # Let X_k[b, c] = V[b, k] * W[c, k].
                # Then sum_k U[a, k] * X_k[b, c] = T[a, b, c].
                # This is a linear system for U[a, :].
                # The matrix A_U has columns A_U[:, k] = X_k.flatten().
                
                A_U = np.zeros((mp * pn, r))
                for k in range(r):
                    # X_k is outer product of V[:, k] and W[:, k]
                    # V[:, k] is (mp, 1), W[:, k] is (pn, 1)
                    # Outer product is (mp, pn). Flatten to (mp*pn,)
                    A_U[:, k] = np.outer(V[:, k], W[:, k]).flatten()
                
                T_flat = target.reshape(nm, mp * pn)
                for a in range(nm):
                    U[a, :] = np.linalg.lstsq(A_U, T_flat[a], rcond=None)[0]
                    
                    # Snap to half-integers? Maybe too aggressive in the loop.
                    # Let's snap at the end of the full cycle or every few iterations.

                # Update V
                A_V = np.zeros((nm * pn, r))
                for k in range(r):
                    A_V[:, k] = np.outer(U[:, k], W[:, k]).flatten()
                
                # T[b, a, c] = sum_k V[b, k] U[a, k] W[c, k]
                # Flatten a, c. Index y = a*pn + c.
                # A_V[:, k] should be U[a, k] * W[c, k] flattened over a, c.
                # This matches np.outer(U[:, k], W[:, k]).flatten()
                
                T_reshaped_V = np.zeros((mp, nm * pn))
                # T is (nm, mp, pn). We need (mp, nm*pn).
                # T_reshaped_V[b, a*pn+c] = T[a, b, c]
                T_reshaped_V = target.transpose(1, 0, 2).reshape(mp, nm * pn)
                
                for b in range(mp):
                    V[b, :] = np.linalg.lstsq(A_V, T_reshaped_V[b], rcond=None)[0]

                # Update W
                A_W = np.zeros((nm * mp, r))
                for k in range(r):
                    A_W[:, k] = np.outer(U[:, k], V[:, k]).flatten()
                
                T_reshaped_W = np.zeros((pn, nm * mp))
                # T[c, a, b] = sum_k W[c, k] U[a, k] V[b, k]
                # Flatten a, b. Index z = a*mp + b.
                T_reshaped_W = target.transpose(2, 0, 1).reshape(pn, nm * mp)
                
                for c in range(pn):
                    W[c, :] = np.linalg.lstsq(A_W, T_reshaped_W[c], rcond=None)[0]

                # Snap to half-integers and check
                if it % 5 == 4: # Every 5th iteration
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    
                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r

            # Final check
            T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.sum((T_approx - target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    elif check_decomposition(U, V, W):
                        return U, V, W

        return None

    # Try ranks from 19 to 23
    for r in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 15:
            break
        
        res = try_rank_sparse_als(r, num_restarts=150)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
