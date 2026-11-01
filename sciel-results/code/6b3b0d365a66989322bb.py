"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, R):
        for a in range(nm):
            for b in range(mp):
                if a % m != b // p:
                    for c in range(pn):
                        s = 0.0
                        for r in range(R):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if abs(s) > 1e-9:
                            return False
                else:
                    c_t = (b % p) * n + (a // m)
                    for c in range(pn):
                        s = 0.0
                        for r in range(R):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if c == c_t:
                            if abs(s - 1) > 1e-9:
                                return False
                        else:
                            if abs(s) > 1e-9:
                                return False
        return True

    def trivial_solution():
        R = n * m * p
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, R

    U0, V0, W0, R0 = trivial_solution()

    random.seed(42)

    for R in range(18, 24):
        if time.time() > deadline:
            break
        for attempt in range(500):
            if time.time() > deadline:
                break
            U = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(R)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(R)] for _ in range(pn)]
            if verify(U, V, W, R):
                return {"U": U, "V": V, "W": W}

    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
