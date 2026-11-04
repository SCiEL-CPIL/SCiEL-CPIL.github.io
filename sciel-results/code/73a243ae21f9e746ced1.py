"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook decomposition (rank 84) as guaranteed fallback
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

    # Verify a decomposition: check sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k
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

    # Fallback result
    fallback = schoolbook()

    # Search for lower-rank decompositions from rank 28 upward
    for rank in range(28, n * m * p):
        if time.time() > deadline:
            break
        # Try random attempts at this rank
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Generate random U, V with entries in {0, 1, -1}
            U = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(m * p)]

            # Compute M[k][i][r] = sum_j U[i*m+j][r] * V[j*p+k][r]
            M = [[[0] * rank for _ in range(n)] for _ in range(p)]
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j][r] * V[j * p + k][r]
                        M[k][i][r] = s

            # Build W: for each (k,i), find r with M[k][i][r] in {1, -1, 2, -2}
            W = [[0] * rank for _ in range(p * n)]
            valid = True
            for k in range(p):
                for i in range(n):
                    found_r = False
                    for r in range(rank):
                        mv = M[k][i][r]
                        if mv == 1:
                            W[k * n + i][r] = 1
                            found_r = True
                            break
                        elif mv == -1:
                            W[k * n + i][r] = -1
                            found_r = True
                            break
                        elif mv == 2:
                            W[k * n + i][r] = 0.5
                            found_r = True
                            break
                        elif mv == -2:
                            W[k * n + i][r] = -0.5
                            found_r = True
                            break
                    if not found_r:
                        valid = False
                        break
                if not valid:
                    break

            if valid and verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
                found = True
        if found:
            break

    return fallback
# EVOLVE-BLOCK-END
