"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 138.0  # Safety margin before 150s kill

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    fallback = schoolbook()

    # Search for lower-rank decompositions.
    # Strategy: random U, V with entries in {-1,0,1}; compute integer M[k*n+i,r]
    # = sum_j U[i*m+j,r]*V[j*p+k,r]; for each (k,i) pick a single r with M=±1
    # (W=±1) or M=±2 (W=±0.5). This keeps entries integer/half-integer.
    def try_rank(rank, max_attempts):
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            # Random sparse U, V with entries in {-1, 0, 1}
            U = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(m * p)]

            # M[k*n+i][r] = sum_j U[i*m+j][r] * V[j*p+k][r]  (integer)
            M = [[0] * rank for _ in range(p * n)]
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    for r in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j][r] * V[j * p + k][r]
                        M[idx][r] = s

            # Build W: for each (k,i), find a single r with M=±1 or M=±2
            W = [[0] * rank for _ in range(p * n)]
            valid = True
            for idx in range(p * n):
                row = M[idx]
                found_r = False
                for r in range(rank):
                    mv = row[r]
                    if mv == 1:
                        W[idx][r] = 1
                        found_r = True
                        break
                    elif mv == -1:
                        W[idx][r] = -1
                        found_r = True
                        break
                    elif mv == 2:
                        W[idx][r] = 0.5
                        found_r = True
                        break
                    elif mv == -2:
                        W[idx][r] = -0.5
                        found_r = True
                        break
                if not found_r:
                    valid = False
                    break
            if not valid:
                continue

            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        return None

    # Lower bound: tensor rank >= max(n, m, p) is weak; use a practical start.
    # A safe lower bound is max(n*p, m*p, n*m) is too high. Use max(n,m,p).
    # We'll scan ranks from a modest lower bound upward.
    lower = max(n, m, p)  # = 7
    upper = n * m * p  # = 84

    for rank in range(lower, upper):
        if time.time() > deadline:
            break
        # More attempts for smaller ranks (harder to satisfy), fewer for larger.
        attempts = 400 if rank < 40 else 150
        result = try_rank(rank, attempts)
        if result is not None:
            return result

    return fallback
