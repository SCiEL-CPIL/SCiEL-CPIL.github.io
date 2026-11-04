"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import copy

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18

    # Schoolbook baseline: rank = n*m*p = 54
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    best_rank = rank_school

    def verify(U, V, W, rank):
        """Verify that the tensor decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Fast verification using float arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to improve via local search
    # Strategy: try reducing rank by 1 at a time
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() - start_time > DEADLINE - 10:
            break

        # Try several random attempts at this rank
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - 5:
                break

            # Initialize with schoolbook and zero out some columns, then perturb
            U_c = [[0] * target_rank for _ in range(nm)]
            V_c = [[0] * target_rank for _ in range(mp)]
            W_c = [[0] * target_rank for _ in range(pn)]

            # Copy first target_rank columns from schoolbook
            for r in range(target_rank):
                for i in range(nm):
                    U_c[i][r] = U[i][r]
                for j in range(mp):
                    V_c[j][r] = V[j][r]
                for k in range(pn):
                    W_c[k][r] = W[k][r]

            # Perturb randomly
            for r in range(target_rank):
                if random.random() < 0.3:
                    row = random.randint(0, nm - 1)
                    U_c[row][r] += random.choice([-1, 1])
                if random.random() < 0.3:
                    row = random.randint(0, mp - 1)
                    V_c[row][r] += random.choice([-1, 1])
                if random.random() < 0.3:
                    row = random.randint(0, pn - 1)
                    W_c[row][r] += random.choice([-1, 1])

            # Fast verification first
            if verify_fast(U_c, V_c, W_c, target_rank):
                # Confirm with exact verification
                if verify(U_c, V_c, W_c, target_rank):
                    best = {"U": U_c, "V": V_c, "W": W_c}
                    best_rank = target_rank
                    break

    return best
# EVOLVE-BLOCK-END
