"""Matrix multiplication 2,3,10 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_rank55():
    """Build the known rank-55 decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part B: j=1,2 - 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        u_11 = 1
        u_12 = 2
        u_21 = 4
        u_22 = 5

        v_11 = 10 + k0
        v_12 = 10 + k1
        v_21 = 20 + k0
        v_22 = 20 + k1

        U[u_11][col] += 1; U[u_22][col] += 1; V[v_11][col] += 1; V[v_22][col] += 1; W[w_00][col] += 1; W[w_11][col] += 1; col += 1
        U[u_21][col] += 1; U[u_22][col] += 1; V[v_11][col] += 1; W[w_10][col] += 1; W[w_11][col] += -1; col += 1
        U[u_11][col] += 1; V[v_12][col] += 1; V[v_22][col] += -1; W[w_01][col] += 1; W[w_11][col] += 1; col += 1
        U[u_22][col] += 1; V[v_21][col] += 1; V[v_11][col] += -1; W[w_00][col] += 1; W[w_10][col] += 1; col += 1
        U[u_11][col] += 1; U[u_12][col] += 1; V[v_22][col] += 1; W[w_00][col] += -1; W[w_01][col] += 1; col += 1
        U[u_21][col] += 1; U[u_11][col] += -1; V[v_11][col] += 1; V[v_12][col] += 1; W[w_11][col] += 1; col += 1
        U[u_12][col] += 1; U[u_22][col] += -1; V[v_21][col] += 1; V[v_22][col] += 1; W[w_00][col] += 1; col += 1

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify decomposition."""
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]

    T_recon = np.zeros((dim_u, dim_v, dim_w))
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])

    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1

    return np.allclose(T_recon, T_true, atol=tol)

def _try_als(rank, deadline, n=2, m=3, p=10, max_attempts=3, max_iters=15, seed=42):
    """Try to find a rank-r decomposition via ALS."""
    dim_u, dim_v, dim_w = n * m, m * p, p * n

    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    np.random.seed(seed)

    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            return None

        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Compute full reconstruction
            T_recon = np.zeros((dim_u, dim_v, dim_w))
            for c in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])

            # Update U
            for c in range(rank):
                term_c = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                residual = T - (T_recon - term_c)
                num = np.einsum('ijk,j,k->i', residual, V[:, c], W[:, c])
                den = np.sum(V[:, c] ** 2) * np.sum(W[:, c] ** 2)
                if abs(den) > 1e-12:
                    U_new = num / den
                    T_recon += np.einsum('i,j,k->ijk', U_new - U[:, c], V[:, c], W[:, c])
                    U[:, c] = U_new

            # Update V
            for c in range(rank):
                term_c = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                residual = T - (T_recon - term_c)
                num = np.einsum('ijk,i,k->j', residual, U[:, c], W[:, c])
                den = np.sum(U[:, c] ** 2) * np.sum(W[:, c] ** 2)
                if abs(den) > 1e-12:
                    V_new = num / den
                    T_recon += np.einsum('i,j,k->ijk', U[:, c], V_new - V[:, c], W[:, c])
                    V[:, c] = V_new

            # Update W
            for c in range(rank):
                term_c = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                residual = T - (T_recon - term_c)
                num = np.einsum('ijk,i,j->k', residual, U[:, c], V[:, c])
                den = np.sum(U[:, c] ** 2) * np.sum(V[:, c] ** 2)
                if abs(den) > 1e-12:
                    W_new = num / den
                    T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W_new - W[:, c])
                    W[:, c] = W_new

            err = np.linalg.norm(T_recon - T)
            if err < 1e-8:
                # Convert to Fractions
                U_frac = [[Fraction(U[i, c]).limit_denominator(1000000) for c in range(rank)] for i in range(dim_u)]
                V_frac = [[Fraction(V[j, c]).limit_denominator(1000000) for c in range(rank)] for j in range(dim_v)]
                W_frac = [[Fraction(W[k, c]).limit_denominator(1000000) for c in range(rank)] for k in range(dim_w)]

                # Verify with float
                if _verify(U_frac, V_frac, W_frac, n, m, p, tol=1e-6):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}

    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # First, build and verify the rank-55 decomposition
    U55, V55, W55 = _build_rank55()
    if _verify(U55, V55, W55, n, m, p):
        best = {"U": U55, "V": V55, "W": W55}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col_s] = 1
                    V_school[j * p + k][col_s] = 1
                    W_school[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    # Try to find a lower-rank decomposition
    for target_rank in [54, 53, 52]:
        if time.time() > deadline - 30:
            break
        result = _try_als(target_rank, deadline, n, m, p)
        if result is not None:
            best = result
            break

    return best
# EVOLVE-BLOCK-END
