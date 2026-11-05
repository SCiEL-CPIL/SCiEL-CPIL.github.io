"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    N = 1008

    # Precompute modular inverses for 1..1008
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    # Build 2D prefix sum matrix for conflict counting
    # M[i][j] = 1 if inv[i] == j else 0
    # We use 1-based indexing for i and j in the matrix logic
    # Size (N+1) x (N+1)
    M = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        M[x][inv[x]] = 1

    # Compute prefix sums P
    # P[i][j] = sum of M[a][b] for 1<=a<=i, 1<=b<=j
    P = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            P[i][j] = P[i-1][j] + P[i][j-1] - P[i-1][j-1] + M[i][j]

    def count_conflicts(L, R):
        # Count x in [L, R] such that inv[x] in [L, R]
        # This is the number of points (x, inv[x]) in the square [L, R] x [L, R]
        if L > R:
            return 0
        return P[R][R] - P[L-1][R] - P[R][L-1] + P[L-1][L-1]

    best = [1, 0]
    L = 1
    for R in range(1, N + 1):
        # Shrink L until [L, R] is valid
        while L <= R and count_conflicts(L, R) > 0:
            L += 1
        
        if L <= R:
            length = R - L + 1
            if length > best[1]:
                best = [L, length]
        
        # Safety check for deadline, though this is very fast
        if time.time() > deadline:
            break

    return best
# EVOLVE-BLOCK-END
