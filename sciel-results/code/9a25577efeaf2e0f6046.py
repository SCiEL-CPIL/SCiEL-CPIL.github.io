import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill

    # Build the target tensor
    nm, mp, pn = n * m, m * p, p * n
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

    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T_target."""
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)

    def try_rank_1_peeling(max_rank=41, max_iter=50):
        """
        Try to find a decomposition of rank <= max_rank using rank-1 peeling.
        """
        residual = T_target.copy()
        U_list = []
        V_list = []
        W_list = []

        for rank_idx in range(max_rank):
            if time.time() - start_time > time_limit - 5:
                return None

            # Find best rank-1 approximation to residual
            # Use alternating maximization
            # Initialize with random or from residual
            u = np.random.randn(nm)
            v = np.random.randn(mp)
            w = np.random.randn(pn)

            best_norm = np.inf
            best_uvw = None

            for _ in range(max_iter):
                # Update u: maximize <residual, u ⊗ v ⊗ w> = u^T (residual * v ⊗ w)
                # residual * v ⊗ w means for each (i,j), the slice residual[i,j,:] * (v*w)
                # Actually: <T, u⊗v⊗w> = sum_{i,j,k} T[i,j,k] u[i] v[j] w[k]
                # = sum_i u[i] * (sum_{j,k} T[i,j,k] v[j] w[k])
                # So u = T_contracted along j,k with v,w
                # Let's compute: for each i, S[i] = sum_{j,k} residual[i,j,k] * v[j] * w[k]
                # Then u = S
                S_u = np.einsum('ijk,j,k->i', residual, v, w)
                u = S_u / (np.linalg.norm(S_u) + 1e-12)

                S_v = np.einsum('ijk,i,k->j', residual, u, w)
                v = S_v / (np.linalg.norm(S_v) + 1e-12)

                S_w = np.einsum('ijk,i,j->k', residual, u, v)
                w = S_w / (np.linalg.norm(S_w) + 1e-12)

                # Check convergence
                norm_uvw = np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(w)
                if norm_uvw < 1e-10:
                    break

            # Compute the rank-1 tensor u⊗v⊗w and subtract from residual
            # The optimal scaling: we want to subtract alpha * u⊗v⊗w
            # alpha = <residual, u⊗v⊗w> / <u⊗v⊗w, u⊗v⊗w>
            # But u, v, w are normalized, so <u⊗v⊗w, u⊗v⊗w> = 1
            # alpha = <residual, u⊗v⊗w>
            alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
            if abs(alpha) < 1e-10:
                return None  # No progress

            residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)

            U_list.append(alpha * u)
            V_list.append(v)
            W_list.append(w)

            # Check if residual is zero
            if np.linalg.norm(residual) < 1e-8:
                break

        # Form the matrices
        R = len(U_list)
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)

        # Verify
        if check_decomposition(U, V, W, tol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # Try rank-1 peeling for ranks 41 down to 40
    for target_rank in [41, 40]:
        result = try_rank_1_peeling(max_rank=target_rank, max_iter=30)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
