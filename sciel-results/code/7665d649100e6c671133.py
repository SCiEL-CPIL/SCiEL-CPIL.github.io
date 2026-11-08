"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import time
import copy
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    n, m, p = n, m, p
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0
                for c in range(rank):
                    lhs += U[i * m + j][c] * V[j * p + k][c]
                for c in range(rank):
                    if lhs != W[k * n + i][c]:
                        return False
    return True


def _bclr_step(U, V, W, n, m, p, max_iter=400):
    """One BCLR elimination step. Returns (U,V,W, success)."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, False
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]

    for _ in range(max_iter):
        # Find a column c with W[i,c]*W[j,c] != 0 for some i != j
        found = False
        for c in range(rank):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    wi = W[i][c]
                    wj = W[j][c]
                    if wi != 0 and wj != 0:
                        # Check feasibility: need U[i,m-1,c] != 0 and V[m-1,p-1,c] != 0
                        if U[i * m + m - 1][c] != 0 and V[(m - 1) * p + (p - 1)][c] != 0:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            return U, V, W, False

        # Perform elimination
        for i2 in range(n):
            for j2 in range(m):
                U[i2 * m + j2][c] -= U[i * m + m - 1][c] * W[i2][c] / W[i][c]
        for j2 in range(m):
            for k2 in range(p):
                V[j2 * p + k2][c] -= V[(m - 1) * p + (p - 1)][c] * W[j2][c] / W[j][c]
        # Zero out W[i,c]
        W[i][c] = 0

        # Clean up near-zero entries (exact arithmetic so just check == 0)
        for row in U:
            for idx in range(rank):
                if row[idx] == 0:
                    row[idx] = 0
        for row in V:
            for idx in range(rank):
                if row[idx] == 0:
                    row[idx] = 0
        for row in W:
            for idx in range(rank):
                if row[idx] == 0:
                    row[idx] = 0

    return U, V, W, False


def _try_random_perturbation(U, V, W, n, m, p, max_iter=300):
    """Apply BCLR with random column permutations to escape local minima."""
    rank = len(U[0])
    U2 = [row[:] for row in U]
    V2 = [row[:] for row in V]
    W2 = [row[:] for row in W]

    # Try multiple random permutations of columns
    for attempt in range(20):
        perm = list(range(rank))
        random.shuffle(perm)
        U3 = [row[:] for row in U2]
        V3 = [row[:] for row in V2]
        W3 = [row[:] for row in W2]
        # Apply permutation
        for idx in range(rank):
            src = perm[idx]
            for i in range(n * m):
                U3[i][idx] = U3[i][src]
            for i in range(m * p):
                V3[i][idx] = V3[i][src]
            for i in range(p * n):
                W3[i][idx] = W3[i][src]

        U3, V3, W3, success = _bclr_step(U3, V3, W3, n, m, p, max_iter=max_iter)
        if success:
            return U3, V3, W3
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build schoolbook decomposition
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Iteratively apply BCLR
    for outer in range(50):
        if time.time() > deadline:
            break
        U2, V2, W2, success = _bclr_step(U, V, W, n, m, p, max_iter=200)
        if success:
            # Verify
            if _verify(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
                new_rank = len(U[0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
            else:
                break
        else:
            # Try random perturbation
            if time.time() > deadline:
                break
            U3, V3, W3 = _try_random_perturbation(U, V, W, n, m, p, max_iter=200)
            if _verify(U3, V3, W3, n, m, p):
                new_rank = len(U3[0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = [row[:] for row in U3], [row[:] for row in V3], [row[:] for row in W3]
                    U, V, W = U3, V3, W3
                else:
                    break
            else:
                break

    # Convert to integers if possible, else keep as floats
    def to_output(mat):
        out = []
        for row in mat:
            out_row = []
            for val in row:
                if val.denominator == 1:
                    out_row.append(int(val))
                else:
                    # Half-integer: multiply by 2 and check
                    doubled = val * 2
                    if doubled.denominator == 1:
                        out_row.append(int(doubled) / 2.0)
                    else:
                        out_row.append(float(val))
            out.append(out_row)
        return out

    return {"U": to_output(best_U), "V": to_output(best_V), "W": to_output(best_W)}
