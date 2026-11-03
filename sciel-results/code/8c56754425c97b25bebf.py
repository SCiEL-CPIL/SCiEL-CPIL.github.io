"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    n, m, p = DIMENSIONS

    def build_strassen_decomp():
        """Build the known rank-55 Strassen-based decomposition."""
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part A: j=0 (first column of A, first row of B) - naive
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        # Part B: j=1,2 - Strassen on 2x2 blocks of B
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11][col] += 1; U[u_22][col] += 1
            V[v_11][col] += 1; V[v_22][col] += 1
            W[w_00][col] += 1; W[w_11][col] += 1
            col += 1
            # M2 = (a21 + a22)b11
            U[u_21][col] += 1; U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1; W[w_11][col] += -1
            col += 1
            # M3 = a11(b12 - b22)
            U[u_11][col] += 1
            V[v_12][col] += 1; V[v_22][col] += -1
            W[w_01][col] += 1; W[w_11][col] += 1
            col += 1
            # M4 = a22(b21 - b11)
            U[u_22][col] += 1
            V[v_21][col] += 1; V[v_11][col] += -1
            W[w_00][col] += 1; W[w_10][col] += 1
            col += 1
            # M5 = (a11 + a12)b22
            U[u_11][col] += 1; U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1; W[w_01][col] += 1
            col += 1
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21][col] += 1; U[u_11][col] += -1
            V[v_11][col] += 1; V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12][col] += 1; U[u_22][col] += -1
            V[v_21][col] += 1; V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1

        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W

    def build_schoolbook():
        """Build the schoolbook decomposition (rank n*m*p = 60)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W):
        """Verify that the decomposition exactly reproduces the matmul tensor."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Build true tensor
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        # Reconstruct from decomposition
        T_recon = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])

        return np.allclose(T_recon, T_true, atol=1e-10)

    def try_remove_factors(U, V, W, max_removals=5):
        """Try to remove individual factors from the decomposition."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Build true tensor
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        # Compute current tensor
        T_cur = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            T_cur += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])

        removed = []
        for c in range(rank - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            # Check if removing factor c still gives correct result
            T_test = T_cur - np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])
            if np.allclose(T_test, T_true, atol=1e-10):
                removed.append(c)
                T_cur = T_test

        if not removed:
            return U, V, W, rank

        # Remove the factors
        keep = [c for c in range(rank) if c not in removed]
        new_rank = len(keep)
        U_new = [[U[i][c] for c in keep] for i in range(n * m)]
        V_new = [[V[i][c] for c in keep] for i in range(m * p)]
        W_new = [[W[i][c] for c in keep] for i in range(p * n)]
        return U_new, V_new, W_new, new_rank

    def try_merge_factors(U, V, W):
        """Try to merge pairs of factors that are linearly dependent."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Build true tensor
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        # Check for linearly dependent factor triples
        # A factor (u, v, w) can be expressed as linear combination of others
        # if the tensor it contributes is in the span of the others' tensors
        T_cur = np.zeros((n * m, m * p, p * n))
        factor_tensors = []
        for c in range(rank):
            ft = np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])
            factor_tensors.append(ft)
            T_cur += ft

        # Flatten tensors for linear algebra
        dim_flat = n * m * m * p * p * n
        T_true_flat = T_true.flatten()

        # Try removing each factor and check if the remaining factors span the true tensor
        # This is expensive, so only try a subset
        removed = []
        for c in range(rank - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            T_test = T_cur - factor_tensors[c]
            if np.allclose(T_test, T_true, atol=1e-10):
                removed.append(c)
                T_cur = T_test

        if not removed:
            return U, V, W, rank

        keep = [c for c in range(rank) if c not in removed]
        new_rank = len(keep)
        U_new = [[U[i][c] for c in keep] for i in range(n * m)]
        V_new = [[V[i][c] for c in keep] for i in range(m * p)]
        W_new = [[W[i][c] for c in keep] for i in range(p * n)]
        return U_new, V_new, W_new, new_rank

    def randomized_search(U, V, W, rank, iterations=1000):
        """Randomly perturb the decomposition and try to reduce rank."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        best_U, best_V, best_W, best_rank = U, V, W, rank

        for it in range(iterations):
            if time.time() - start_time > DEADLINE:
                break

            # Randomly perturb
            U_pert = U_np.copy()
            V_pert = V_np.copy()
            W_pert = W_np.copy()

            # Small random perturbations
            noise_scale = 0.1
            U_pert += np.random.randn(n * m, rank) * noise_scale
            V_pert += np.random.randn(m * p, rank) * noise_scale
            W_pert += np.random.randn(p * n, rank) * noise_scale

            # Check if still valid (should be close but not exact)
            # Try to project back to integer/half-integer
            U_round = np.round(U_pert * 2) / 2
            V_round = np.round(V_pert * 2) / 2
            W_round = np.round(W_pert * 2) / 2

            T_test = np.zeros((n * m, m * p, p * n))
            for c in range(rank):
                T_test += np.einsum('i,j,k->ijk', U_round[:, c], V_round[:, c], W_round[:, c])

            if np.allclose(T_test, T_true, atol=1e-10):
                # Valid! Try to reduce rank
                U_new, V_new, W_new, new_rank = try_remove_factors(
                    [[U_round[i, c] for c in range(rank)] for i in range(n * m)],
                    [[V_round[i, c] for c in range(rank)] for i in range(m * p)],
                    [[W_round[i, c] for c in range(rank)] for i in range(p * n)],
                    new_rank
                )
                if new_rank < best_rank:
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank

        return best_U, best_V, best_W, best_rank

    # Build initial decomposition
    U, V, W = build_strassen_decomp()
    rank = len(U[0])

    # Verify
    if not verify_decomp(U, V, W):
        U, V, W = build_schoolbook()
        rank = len(U[0])

    # Try to reduce rank by removing redundant factors
    U, V, W, rank = try_remove_factors(U, V, W)

    # Try randomized search for better decomposition
    if time.time() - start_time < DEADLINE - 5:
        U, V, W, rank = randomized_search(U, V, W, rank, iterations=500)

    # Final verification
    if not verify_decomp(U, V, W):
        U, V, W = build_schoolbook()
        rank = len(U[0])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
