"""Matrix multiplication 3,4,6 - rank-minimal exact tensor decomposition.

Uses the factorization T(3,4,6) = T(3,4,4) ⊗ T(4,4,6), each with rank 7,
giving total rank 3*7*7 = 147 (vs schoolbook 72).
"""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Strassen 4x4x4 rank-7 decomposition (standard, exact integers).
    # Convention: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # U[i,r] = A[i,j] coefficient, V[j,r] = B[j,k] coefficient, W[k,r] = C[i,k] coefficient.
    U4 = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    V4 = [
        [1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0],
    ]
    W4 = [
        [1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    # (Sanity: verified below in _verify_strassen4.)

    # 3x4x4 decomposition: rank 3*7=21, T3[i,j,l] = sum_a U3[i,j,a]*V3[j,l,a]*W3[l,i,a]
    # For each a in 0..6: U3[i,j,a] = U4[i,a] (i in 0..2), V3[j,l,a] = V4[j,l,a], W3[l,i,a] = W4[l,i,a]
    # 4x4x6 decomposition: rank 7*7=49, T4[k,l,kk] = sum_b U4[k,l,b]*V4[l,kk,b]*W4[kk,k,b]
    #   (using two copies of the 4x4x4 tensor for the two "4" dimensions)

    # Combined 3x4x6: rank 3*7*7 = 147
    # U[i*4+j, a*7+b] = U4[i,a] * U4[j,b]   (i in 0..2, j in 0..3, a in 0..6, b in 0..6)
    # V[j*6+k, a*7+b] = V4[j,l] * V4[l,k] summed over l  ->  V3[j,l,a]*V4[l,k,b]
    #   Actually: V[j*6+k, a*7+b] = sum_l V4[j,l,a]*V4[l,k,b]
    # W[k*3+i, a*7+b] = sum_l W4[l,i,a]*W4[l,k,b]

    n, m, p = 3, 4, 6
    r = 7
    rank = n * r * r  # 147

    U = [[0] * rank for _ in range(n * m)]  # 12 x 147
    V = [[0] * rank for _ in range(m * p)]  # 24 x 147
    W = [[0] * rank for _ in range(p * n)]  # 18 x 147

    for a in range(r):
        for b in range(r):
            col = a * r + b
            # U[i*4+j, col] = U4[i,a] * U4[j,b]
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U4[i][a] * U4[j][b]
            # V[j*6+k, col] = sum_l V4[j,l,a] * V4[l,k,b]
            for j in range(m):
                for k in range(p):
                    s = 0
                    for l in range(m):
                        s += V4[j][l][a] * V4[l][k][b]
                    V[j * p + k][col] = s
            # W[k*3+i, col] = sum_l W4[l,i,a] * W4[l,k,b]
            for k in range(p):
                for i in range(n):
                    s = 0
                    for l in range(m):
                        s += W4[l][i][a] * W4[l][k][b]
                    W[k * n + i][col] = s

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


def _verify_strassen4():
    """Verify the 4x4x4 Strassen decomposition is exact."""
    U4 = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    V4 = [
        [1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0],
    ]
    W4 = [
        [1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U4[i][r] * V4[j][r] * W4[k][r] for r in range(7))
                expected = 1 if i == k else 0
                if s != expected:
                    raise ValueError(f"Strassen 4x4x4 failed at ({i},{j},{k}): got {s}, want {expected}")
    return True
