"""Refined Strassen + compact 2x1x8 construction for 2x3x8 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Attempt rank 42: 4*7 (Strassen blocks) + 14 (compact 2x1x8)
    # The 2x1x8 tensor T[i,k] = A[i,2]*B[2,k] needs rank >= max(2,8) = 8
    # But we can potentially share structure. Let's try rank 14 for the 2x1x8 part.
    # Actually, 2x1x8 is just outer product of a 2-vector and 8-vector.
    # Minimum rank is max(2,1,8) = 8? No, tensor rank of a* b^T where a is 2x1, b is 1x8.
    # This is a rank-1 tensor in the (i,k) space? No, it's T[i,k] = a[i]*b[k].
    # This is already rank 1 as a matrix, but as a 3-tensor with the middle index fixed...
    # Wait, the tensor is T[i, a, k] where a is fixed at 2. So it's T[i, 2, k] = A[i,2]*B[2,k].
    # For the decomposition, we need U[i*3+2, c] * V[2*8+k, c] * W[k*2+i, c] = A[i,2]*B[2,k]
    # This is a 2x8 matrix factorization problem essentially. Rank 8 is the schoolbook.
    # Can we do better? The tensor is separable: T[i,k] = u[i]*v[k]. 
    # In tensor decomposition terms, this is a rank-1 tensor if we consider it as a 3-tensor
    # with the middle dimension collapsed. But we need to embed it in the full 3-tensor space.
    # 
    # Let me just use rank 8 for this part and see if total 44 validates.
    # Then try to optimize.

    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0

    # Part 1: A_left (2x2) * B_top (2x8) - 4 Strassen blocks
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A_right (2x1) * B_bottom (1x8) - schoolbook rank 8
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
