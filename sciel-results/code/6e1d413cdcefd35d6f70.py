import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U @ V @ W reconstruction matches target tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_rec = np.zeros(T.shape)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_rec, T, atol=tol)

def _round_to_halfint(arr, tol=1e-3):
    """Round array to nearest half-integers, return (rounded, max_error)."""
    rounded = np.round(arr * 2.0) / 2.0
    max_err = np.max(np.abs(arr - rounded))
    return rounded, max_err

def _als_decompose(T, n, m, p, rank, max_iters=100, tol=1e-8):
    """
    Alternating Least Squares for tensor decomposition.
    T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    """
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    prev_loss = np.inf
    for _ in range(max_iters):
        # Solve for U: T_rec[u,v,w] = U[u,r] * (V[v,r] W[w,r])
        # For each u, we have a vector of T[u,:,:] which is sum_r U[u,r] * (V[:,r] * W[:,r])
        # Let M_vw[r, v, w] = V[v,r] * W[w,r]. This is a matrix of size rank x (m*p * p*n)?
        # Actually, for fixed u, T[u, :, :] is a matrix of size (m*p, p*n).
        # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # Let B_u be the matrix where B_u[r, (v,w)] = V[v, r] * W[w, r].
        # Then T[u, (v,w)] = U[u, :] @ B_u.
        # We can solve U[u, :] = T[u, (v,w)] @ pinv(B_u).
        
        # Efficient way:
        # Reshape T to (n*m, m*p*p*n).
        # For each r, let X_r = outer(V[:,r], W[:,r]) flattened to (m*p*p*n).
        # Then T_flat = U @ X.T ?
        # T[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # Let X be a matrix of shape (rank, m*p*p*n) where X[r, idx(v,w)] = V[v,r] W[w,r].
        # Then T_flat (shape n*m x m*p*p*n) = U (n*m x rank) @ X.T (rank x m*p*p*n).
        # So U = T_flat @ pinv(X.T) = T_flat @ pinv(X).T ?
        # U = T_flat @ (X @ X.T)^-1 @ X ? No, that's if X is full rank.
        # U = T_flat @ pinv(X.T).
        
        # Construct X
        # X[r, v*p*n + w] = V[v, r] * W[w, r]
        X = np.zeros((rank, m * p * p * n))
        for r in range(rank):
            X[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
        T_flat = T.reshape(n * m, m * p * p * n)
        
        # Solve U
        U_new, residuals, rank_u, s = np.linalg.lstsq(X.T, T_flat.T, rcond=None)
        U = U_new.T
        
        # Solve V
        # T[u, v, w] = V[v, r] * (U[u, r] W[w, r])
        # Let Y[r, u, w] = U[u, r] W[w, r].
        # T[v, u, w] ? No, T is indexed [u, v, w].
        # For fixed v, T[:, v, :] is a matrix (n*m, p*n).
        # T[u, v, w] = sum_r V[v, r] * U[u, r] * W[w, r]
        # Let Y[r, idx(u,w)] = U[u, r] * W[w, r].
        # T[:, v, :].flatten() = V[v, :] @ Y.T
        # So V[v, :] = T[:, v, :].flatten() @ pinv(Y.T)
        
        Y = np.zeros((rank, n * m * p * n))
        for r in range(rank):
            Y[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
        # T is (n*m, m*p, p*n). We need to access T[:, v, :].
        # T_v = T[:, v, :] shape (n*m, p*n). Flatten to (n*m*p*n).
        # We want to solve for V (m*p, rank).
        # For each v (0..m*p-1), T[:, v, :].flatten() = V[v, :] @ Y.T
        # Stack all v: T_stacked (m*p, n*m*p*n) = V (m*p, rank) @ Y.T (rank, n*m*p*n)
        T_stacked = np.zeros((m * p, n * m * p * n))
        for v in range(m * p):
            T_stacked[v, :] = T[:, v, :].flatten()
            
        V_new, _, _, _ = np.linalg.lstsq(Y.T, T_stacked.T, rcond=None)
        V = V_new.T
        
        # Solve W
        # T[u, v, w] = W[w, r] * (U[u, r] V[v, r])
        # Let Z[r, u, v] = U[u, r] V[v, r].
        # For fixed w, T[:, :, w] is (n*m, m*p).
        # T[u, v, w] = sum_r W[w, r] U[u, r] V[v, r]
        # T[:, :, w].flatten() = W[w, :] @ Z.T
        # Stack all w: T_stacked_w (p*n, n*m*m*p) = W (p*n, rank) @ Z.T
        
        Z = np.zeros((rank, n * m * m * p))
        for r in range(rank):
            Z[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
        T_stacked_w = np.zeros((p * n, n * m * m * p))
        for w in range(p * n):
            T_stacked_w[w, :] = T[:, :, w].flatten()
            
        W_new, _, _, _ = np.linalg.lstsq(Z.T, T_stacked_w.T, rcond=None)
        W = W_new.T
        
        # Calculate loss
        T_rec = np.zeros(T.shape)
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        loss = np.sum((T_rec - T) ** 2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
    return U, V, W, loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    T = _build_target_tensor(n, m, p)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    # Try ranks from 20 down to 15
    # Start with higher ranks to ensure we find *some* improvement, then try lower
    for target_rank in range(40, 14, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Fewer restarts for ALS as it's faster, but still use some
        n_restarts = 2
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            np.random.seed(42 + restart * 100 + target_rank)
            
            try:
                U, V, W, loss = _als_decompose(T, n, m, p, target_rank, max_iters=50)
                
                if loss < 1e-4:
                    # Round to half-integers
                    U_rounded, err_u = _round_to_halfint(U)
                    V_rounded, err_v = _round_to_halfint(V)
                    W_rounded, err_w = _round_to_halfint(W)
                    
                    max_err = max(err_u, err_v, err_w)
                    
                    if max_err < 1e-3:
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                            best_result = {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            best_rank = target_rank
                            found = True
                            break
                    else:
                        # Try integer rounding if half-int fails? Or just accept if very close?
                        # The problem requires integer/half-integer.
                        # If ALS converges to non-integer, it might not be a valid solution.
                        pass
            except Exception:
                pass
        
        if found:
            continue
        else:
            # If we fail at this rank, try lower?
            # ALS might get stuck in local minima.
            pass

    return best_result
