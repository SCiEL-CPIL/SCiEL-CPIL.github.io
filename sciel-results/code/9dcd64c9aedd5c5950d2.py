"""Combined approach: structured search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s limit

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def verify_decomposition_numpy(U_mat, V_mat, W_mat, rank):
        """Vectorized verification using numpy."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        result = np.zeros((n, m, p))
        for c in range(rank):
            result += np.einsum('ij,jk,ki->ijk', U_3d[:,:,c], V_3d[:,:,c], W_3d[:,:,c])

        return np.allclose(result, np.ones((n,m,p)), atol=1e-9)

    def try_rank(rank, num_attempts=100):
        """Try to find a valid decomposition at the given rank."""
        if time.time() > deadline - 10:
            return None

        # Strategy 1: Random search with structured initialization
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                return None

            # Use different random seeds for diversity
            seed = hash((rank, attempt)) % (2**32)
            np_rng = np.random.RandomState(seed)

            # Random initialization
            U = np_rng.randn(n*m, rank) * 0.5
            V = np_rng.randn(m*p, rank) * 0.5
            W = np_rng.randn(p*n, rank) * 0.5

            if verify_decomposition_numpy(U, V, W, rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Strategy 2: Try sparse structured decompositions
        # For small ranks, try to find solutions with mostly zeros
        if rank <= 30:
            for attempt in range(min(50, num_attempts)):
                if time.time() > deadline - 10:
                    return None

                seed = hash((rank, attempt, 'sparse')) % (2**32)
                np_rng = np.random.RandomState(seed)

                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))

                # Randomly place non-zero entries
                for c in range(rank):
                    # Choose which indices get non-zero values
                    u_indices = np_rng.choice(n*m, size=min(4, n*m), replace=False)
                    v_indices = np_rng.choice(m*p, size=min(4, m*p), replace=False)
                    w_indices = np_rng.choice(p*n, size=min(4, p*n), replace=False)

                    U[u_indices, c] = np_rng.choice([-1, 1], size=len(u_indices))
                    V[v_indices, c] = np_rng.choice([-1, 1], size=len(v_indices))
                    W[w_indices, c] = np_rng.choice([-1, 1], size=len(w_indices))

                if verify_decomposition_numpy(U, V, W, rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound considerations:
    # - Trivial bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # - But we can potentially do better with clever constructions
    # Try from 16 up to schoolbook
    for rank in range(16, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break

        if rank >= best_rank:
            continue

        result = try_rank(rank, num_attempts=200)

        if result is not None:
            best_rank = rank
            best_result = result
            # If we found a very low rank, we can stop early
            if rank <= 20:
                break

    return best_result
# EVOLVE-BLOCK-END
