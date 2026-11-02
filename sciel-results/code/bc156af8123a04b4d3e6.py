"""Find a rank-minimal exact tensor decomposition for multiplying a 2x4 by 4x8 matrix."""
import time

DIMENSIONS = [2, 4, 8]
WALL_LIMIT = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 8

    def schoolbook():
        rank = n * m * p  # 64
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    best = schoolbook()

    deadline = start_time + WALL_LIMIT - 10.0  # 140s guard

    # --- Quick search for lower rank ---
    import numpy as np
    np.random.seed(42)

    # For each candidate rank r < 64, try to find integer/half-integer coefficients
    # that exactly satisfy the decomposition equations.
    # The tensor equation: for all (i,j,k), C_{ik} = sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
    # But the decomposition is: T_{(i,j),(j,k),(k,i)} = sum_a U[i*m+j, a] V[j*p+k, a] W[k*n+i, a]
    # The target tensor has T_{(i,j),(j,k),(k,i)} = 1 for all (i,j,k), and the output
    # index is (k,i) not (i,k). Wait, let me re-read.

    # The task says: output factor W[k*n+i, r]
    # The multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor: T_{(i,j),(j,k),(i,k)} = 1  (since C[i,k] gets contribution A[i,j]*B[j,k])
    # But the W index is W[k*n + i], which is (k,i) order. So the third index is (k,i).
    # Let me re-check: W[k*n+i, r]. So W is indexed by (k,i) where k is row of B / col of C,
    # and i is row of A / row of C. So the third factor index is (k, i).
    # The tensor equation: sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1 for all (i,j,k).

    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) with entries in (1/2)Z
    # such that for all (i,j,k): sum_a U[i*m+j,a] V[j*p+k,a] W[k*n+i,a] = 1.

    # This is a bilinear/trilinear system. For small r, we can try random integer/half-integer
    # vectors and check.

    # Given the structure, for 2x4x8 the rank is very likely 64. But let's do a brief
    # randomized check for r = 63, 62 to see if we can find a solution.

    def try_random_rank(r, num_trials=20000):
        """Try to find a valid decomposition of rank r using random half-integer entries."""
        for trial in range(num_trials):
            # Generate random half-integer entries: {-1, -0.5, 0, 0.5, 1}
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))

            for a in range(r):
                for idx in range(n * m):
                    U[idx, a] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                for idx in range(m * p):
                    V[idx, a] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                for idx in range(p * n):
                    W[idx, a] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])

            # Check: for all (i,j,k), sum_a U[i*m+j,a] V[j*p+k,a] W[k*n+i,a] == 1
            # Do a quick check on a subset
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for a in range(r):
                            s += U[i * m + j, a] * V[j * p + k, a] * W[k * n + i, a]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        return None

    # Try a few ranks below 64 with limited trials
    for r in [63, 62, 61]:
        if time.time() > deadline:
            break
        # Use a smaller number of trials given time constraints
        trials = 5000 if r >= 62 else 2000
        result = try_random_rank(r, trials)
        if result is not None:
            U_list, V_list, W_list = result
            best = {"U": U_list, "V": V_list, "W": W_list, "rank": r}
            break

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
