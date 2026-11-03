import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor T[i,j,k]
    # T_target[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-8):
        # Check if U, V, W are close to half-integers
        # The problem states "integer/half-integer entries"
        # We will check exactness of the tensor decomposition first
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)

    def snap_to_half_integer(M, tol=1e-5):
        """Snap matrix entries to nearest half-integer."""
        M_snapped = np.round(M * 2.0) / 2.0
        # Check if snapping was close
        if np.max(np.abs(M - M_snapped)) < tol:
            return M_snapped
        return None

    def try_rank_hooi(T, rank, max_iter=50, n_restarts=5, seed=42):
        """
        Use Higher-Order Orthogonal Iteration (HOOI) to find rank-r decomposition.
        Then snap to half-integers and verify.
        """
        np.random.seed(seed)
        best_result = None
        best_err = np.inf

        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize
            U = U / np.linalg.norm(U, axis=0, keepdims=True)
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            W = W / np.linalg.norm(W, axis=0, keepdims=True)

            prev_err = np.inf
            for it in range(max_iter):
                # Update U
                # T_{:,:,k} * (V_k * W_k^T) -> nm x rank
                # Reshape T to (nm, mp, pn)
                # S_u = einsum('uvw, vr, wr -> ur', T, V, W)
                S_u = np.einsum('uvw,vr,wr->ur', T, V, W)
                U_new, _, _ = np.linalg.svd(S_u, full_matrices=False)
                # Take first rank columns
                U_new = U_new[:, :rank]
                
                # Update V
                S_v = np.einsum('uvw,ur,wr->vr', T, U_new, W)
                V_new, _, _ = np.linalg.svd(S_v, full_matrices=False)
                V_new = V_new[:, :rank]
                
                # Update W
                S_w = np.einsum('uvw,ur,vr->wr', T, U_new, V_new)
                W_new, _, _ = np.linalg.svd(S_w, full_matrices=False)
                W_new = W_new[:, :rank]

                # Check convergence
                err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if err < 1e-10:
                    break

            # Compute the actual decomposition error
            T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_approx - T)
            
            if err < best_err:
                best_err = err
                best_result = (U, V, W)
                
            # If error is very small, try snapping
            if err < 1e-6:
                U_s = snap_to_half_integer(U)
                V_s = snap_to_half_integer(V)
                W_s = snap_to_half_integer(W)
                if U_s is not None and V_s is not None and W_s is not None:
                    if check_decomposition(U_s, V_s, W_s, tol=1e-6):
                        return (U_s, V_s, W_s)

        if best_result is not None and best_err < 1e-6:
            return best_result
        return None

    def try_rank_als(T, rank, max_iter=100, n_restarts=5, seed=42):
        """
        Alternating Least Squares for tensor decomposition.
        """
        np.random.seed(seed)
        best_result = None
        best_err = np.inf

        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)

            prev_err = np.inf
            for it in range(max_iter):
                # Update U
                # For each u, solve min ||T_u - U_u (V^T W^T)||
                # T_u is (mp, pn). V^T W^T is (rank, rank)? No.
                # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # Fix V, W. For each u:
                # T[u, :, :] = sum_r U[u, r] (V[:, r] outer W[:, r])
                # Let M_u = V^T W^T (rank x rank)? No.
                # Let B_u = V^T W^T is not correct.
                # We want to solve for U[:, r] given V, W.
                # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
                # Let X_u = V^T W^T? No.
                # Let's flatten.
                # For fixed u, T[u, v, w] is a matrix.
                # sum_r U[u, r] * (V[:, r] * W[:, r]^T)
                # This is a linear system in U[u, :].
                # Let A_u be a matrix where A_u[:, r] = V[:, r] * W[:, r] ? No, that's element-wise.
                # We need to solve: vec(T[u]) = sum_r U[u, r] * vec(V[:, r] * W[:, r]^T)
                # Let G_r = V[:, r] * W[:, r]^T (mp x pn matrix)
                # vec(G_r) is (mp*pn) vector.
                # We have a system: T_flat_u = G * U_u
                # Where G is (mp*pn) x rank matrix with columns vec(G_r).
                
                G = np.zeros((mp * pn, rank))
                for r in range(rank):
                    G[:, r] = (V[:, r] * W[:, r]).reshape(mp * pn) # Element-wise product V_r * W_r?
                    # Wait, T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
                    # The term for rank r is U[u,r] * (V[v,r] * W[w,r])
                    # So the "basis" tensor for rank r is V[:,r] outer W[:,r]
                    # vec(V[:,r] outer W[:,r]) is indeed V[:,r] * W[:,r] flattened?
                    # Yes, if we flatten row-major: (V_r[i] * W_r[j])
                    
                # Solve for U
                U_new, _, _, _ = np.linalg.lstsq(G, T.reshape(nm, mp * pn).T, rcond=None)
                U_new = U_new.T # shape (nm, rank)
                
                # Update V
                # T[v, u, w] = sum_r V[v, r] * (U[u, r] * W[w, r])
                # G_v = (U[:, r] * W[:, r]) flattened
                G_v = np.zeros((nm * pn, rank))
                for r in range(rank):
                    G_v[:, r] = (U[:, r] * W[:, r]).reshape(nm * pn)
                V_new, _, _, _ = np.linalg.lstsq(G_v, T.reshape(mp, nm * pn).T, rcond=None)
                V_new = V_new.T
                
                # Update W
                # T[w, u, v] = sum_r W[w, r] * (U[u, r] * V[v, r])
                G_w = np.zeros((nm * mp, rank))
                for r in range(rank):
                    G_w[:, r] = (U[:, r] * V[:, r]).reshape(nm * mp)
                W_new, _, _, _ = np.linalg.lstsq(G_w, T.reshape(pn, nm * mp).T, rcond=None)
                W_new = W_new.T

                err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if err < 1e-10:
                    break

            T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_approx - T)
            
            if err < best_err:
                best_err = err
                best_result = (U, V, W)
                
            if err < 1e-6:
                U_s = snap_to_half_integer(U)
                V_s = snap_to_half_integer(V)
                W_s = snap_to_half_integer(W)
                if U_s is not None and V_s is not None and W_s is not None:
                    if check_decomposition(U_s, V_s, W_s, tol=1e-6):
                        return (U_s, V_s, W_s)

        if best_result is not None and best_err < 1e-6:
            return best_result
        return None

    # Strategy: Try ranks from lower bound up to schoolbook
    # Lower bound is max(nm, mp, pn) = 21
    # We will try a few ranks.
    
    best_result = None
    best_rank = n * m * p + 1

    # Try ranks 21 to 42
    # We prioritize lower ranks
    for r in range(21, n * m * p + 1):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try ALS first as it is more robust for exact recovery if possible
        result = try_rank_als(T_target, r, max_iter=50, n_restarts=3, seed=42 + r)
        if result is not None:
            U, V, W = result
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
            if r == 21: # Theoretical min
                break

        # If ALS failed, try HOOI
        if time.time() - start_time < time_limit - 10:
            result = try_rank_hooi(T_target, r, max_iter=30, n_restarts=3, seed=100 + r)
            if result is not None:
                U, V, W = result
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
                if r == 21:
                    break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return get_schoolbook()
