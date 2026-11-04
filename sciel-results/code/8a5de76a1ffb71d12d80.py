"""
Rank-minimal tensor decomposition for 4x5 @ 5x6 matrix multiplication.

The multiplication tensor T_{i,j,k} = 1 for all i in [0,4), j in [0,5), k in [0,6).
We seek U (20 x R), V (30 x R), W (24 x R) with integer/half-integer entries such that
sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all (i,j,k).

Strategy:
1. Try to find a decomposition with rank < 120 using structured random search
   with local optimization (coordinate descent on the error tensor).
2. If no improvement is found within the time budget, return the schoolbook
   decomposition with rank 120.
"""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    target_rank = N * M * P  # 120

    def schoolbook():
        U = [[0] * target_rank for _ in range(N * M)]
        V = [[0] * target_rank for _ in range(M * P)]
        W = [[0] * target_rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def compute_error(U, V, W, R):
        """Compute the number of (i,j,k) where the decomposition fails."""
        err_count = 0
        max_err = 0.0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    val = 0.0
                    for r in range(R):
                        val += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                    if abs(val - 1.0) > 1e-9:
                        err_count += 1
                        max_err = max(max_err, abs(val - 1.0))
        return err_count, max_err

    def local_search(U, V, W, R, max_steps=20000):
        """
        Local search: try to flip individual entries to reduce the error count.
        Returns (U, V, W, err_count) after local search.
        """
        err_count, _ = compute_error(U, V, W, R)
        if err_count == 0:
            return U, V, W, 0

        # Precompute which (i,j,k) are in error
        def get_error_mask():
            mask = set()
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        val = 0.0
                        for r in range(R):
                            val += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                        if abs(val - 1.0) > 1e-9:
                            mask.add((i, j, k))
            return mask

        error_mask = get_error_mask()
        step = 0
        while step < max_steps and time.time() < deadline - 2:
            improved = False
            # Try flipping a random entry in U
            if random.random() < 0.33:
                idx = random.randrange(N * M)
                r = random.randrange(R)
                old = U[idx][r]
                if old == 0:
                    new_val = random.choice([-1, 1])
                else:
                    new_val = -old if random.random() < 0.5 else 0
                if new_val != old:
                    U[idx][r] = new_val
                    new_err, _ = compute_error(U, V, W, R)
                    if new_err < err_count:
                        err_count = new_err
                        improved = True
                        step += 1
                    else:
                        U[idx][r] = old
            # Try flipping a random entry in V
            elif random.random() < 0.66:
                idx = random.randrange(M * P)
                r = random.randrange(R)
                old = V[idx][r]
                if old == 0:
                    new_val = random.choice([-1, 1])
                else:
                    new_val = -old if random.random() < 0.5 else 0
                if new_val != old:
                    V[idx][r] = new_val
                    new_err, _ = compute_error(U, V, W, R)
                    if new_err < err_count:
                        err_count = new_err
                        improved = True
                        step += 1
                    else:
                        V[idx][r] = old
            # Try flipping a random entry in W
            else:
                idx = random.randrange(P * N)
                r = random.randrange(R)
                old = W[idx][r]
                if old == 0:
                    new_val = random.choice([-1, 1])
                else:
                    new_val = -old if random.random() < 0.5 else 0
                if new_val != old:
                    W[idx][r] = new_val
                    new_err, _ = compute_error(U, V, W, R)
                    if new_err < err_count:
                        err_count = new_err
                        improved = True
                        step += 1
                    else:
                        W[idx][r] = old

            if err_count == 0:
                return U, V, W, 0

            # If no improvement in a while, break
            if step % 5000 == 0 and not improved and step > 5000:
                break

        return U, V, W, err_count

    # Try to find a decomposition with rank < 120
    # Start with structured initializations that are close to schoolbook

    best_result = None

    for R in range(target_rank - 1, max(target_rank - 15, 90), -1):
        if time.time() > deadline - 15:
            break

        for restart in range(3):
            if time.time() > deadline - 5:
                break

            # Initialize with a perturbed schoolbook-like structure
            # Use R columns; try to make them as "independent" as possible
            U = [[0] * R for _ in range(N * M)]
            V = [[0] * R for _ in range(M * P)]
            W = [[0] * R for _ in range(P * N)]

            # Strategy: start with schoolbook pattern for first min(R,120) columns,
            # then add random structure for remaining columns
            num_base = min(R, target_rank)
            col = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        if col >= R:
                            break
                        U[i * M + j][col] = 1
                        V[j * P + k][col] = 1
                        W[k * N + i][col] = 1
                        col += 1
                    if col >= R:
                        break
                if col >= R:
                    break

            # Add some random entries to break the schoolbook structure
            for _ in range(R * 5):
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randrange(N * M)
                    r = random.randrange(R)
                    U[idx][r] = random.choice([-1, 0, 1])
                elif mat == 'V':
                    idx = random.randrange(M * P)
                    r = random.randrange(R)
                    V[idx][r] = random.choice([-1, 0, 1])
                else:
                    idx = random.randrange(P * N)
                    r = random.randrange(R)
                    W[idx][r] = random.choice([-1, 0, 1])

            # Run local search
            U, V, W, err_count = local_search(U, V, W, R, max_steps=5000)

            if err_count == 0:
                return {"U": U, "V": V, "W": W}

            if best_result is None or R < best_result[0]:
                best_result = (R, U, V, W, err_count)

    # If no improvement found, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
