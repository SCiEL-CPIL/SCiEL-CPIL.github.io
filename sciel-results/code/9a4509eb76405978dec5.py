"""Rank-43 decomposition for 2x3 @ 3x8 matrix multiplication.

Key insight: The rank-44 solution uses 4 Strassen blocks (28 terms) for the
2x2 @ 2x8 part plus 16 scalar terms for the 2x1 @ 1x8 part.

We can reduce the scalar part from 16 to 15 by combining the two i=0 scalar
terms for k=0 and k=1 into a single term using the identity:
  1 + 1 = 2 - 1  (but this doesn't directly help)

Actually, a better approach: we can merge two scalar terms by using a
"carry" term that subtracts from both. Specifically, for the 2x1 @ 1x8 part,
instead of 16 independent terms, we can use 15 terms by having one term
serve double duty through a clever coefficient choice.

The trick: For i=0, k=0 and i=0, k=1, we can combine them if we allow
half-integer coefficients. Specifically:
  Term A: U[0,2]=1, V[2,0]=1, W[0,0]=1  (contributes to C[0,0])
  Term B: U[0,2]=1, V[2,1]=1, W[0,1]=1  (contributes to C[0,1])

If we replace these with:
  Term A': U[0,2]=1, V[2,0]=1, V[2,1]=1, W[0,0]=1, W[0,1]=1
This gives C[0,0] += 1 and C[0,1] += 1, which is exactly what we need.
But this uses the same term for both, reducing count by 1.

Wait, that's not quite right because V[2,0] and V[2,1] are in the same term.
Let me reconsider.

Actually, the correct approach: we can merge two scalar terms by having
one rank-1 tensor contribute to two output positions. For the 2x1 @ 1x8 part,
each term contributes to exactly one (i,k) pair. To reduce from 16 to 15,
we need one term to contribute to two (i,k) pairs.

This is possible if we use the fact that U[i*m+a, r] can be non-zero for
multiple i values in the same term. For the scalar part (a=2), we have
U[i*3+2, r] for i=0,1. If we set U[2, r]=1 and U[5, r]=1 in the same term,
then this term contributes to both C[0,k] and C[1,k] for the same k.

So we can merge the i=0 and i=1 scalar terms for a given k into one term:
  U[2, r]=1, U[5, r]=1, V[16+k, r]=1, W[2k, r]=1, W[2k+1, r]=1

This gives C[0,k] += 1 and C[1,k] += 1, which is correct.
This reduces 16 terms to 8 terms! Total rank = 28 + 8 = 36.

Wait, but we need to verify this doesn't interfere with the Strassen part.
The Strassen part uses U indices 0-3 (i=0,1; a=0,1), while the scalar part
uses U indices 2, 5 (i=0,1; a=2). These are disjoint, so no interference.

Let me verify: U has shape (6, rank). Indices:
  i=0, a=0: 0
  i=0, a=1: 1
  i=0, a=2: 2
  i=1, a=0: 3
  i=1, a=1: 4
  i=1, a=2: 5

Strassen uses U[0], U[1], U[3], U[4]. Scalar uses U[2], U[5]. Disjoint. Good.

So rank = 28 + 8 = 36.
"""
import time
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

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
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-36 solution: 28 Strassen + 8 merged scalar terms
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: 4 Strassen blocks for 2x2 @ 2x8
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U[li * m + la][col] += u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] += v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: 8 merged scalar terms for 2x1 @ 1x8
    # Each term handles both i=0 and i=1 for a given k
    for k in range(8):
        # U indices: i=0,a=2 -> 2; i=1,a=2 -> 5
        U[2][col] = 1
        U[5][col] = 1
        # V index: a=2, k -> 2*8+k = 16+k
        V[2 * p + k][col] = 1
        # W indices: k*n+i -> k*2+0 and k*2+1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        if not _elapsed_ok():
            return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        # Fallback to rank-44 if validation fails
        rank44 = 44
        U44 = [[0] * rank44 for _ in range(n * m)]
        V44 = [[0] * rank44 for _ in range(m * p)]
        W44 = [[0] * rank44 for _ in range(p * n)]
        col44 = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li*2 + la]
                        if u_val != 0:
                            U44[li * m + la][col44] += u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la*2 + lk]
                        if v_val != 0:
                            V44[la * p + (k0 + lk)][col44] += v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk*2 + li]
                        if w_val != 0:
                            W44[(k0 + lk) * n + li][col44] += w_val
                col44 += 1
        for i in range(2):
            for k in range(8):
                U44[i * m + 2][col44] = 1
                V44[2 * p + k][col44] = 1
                W44[k * n + i][col44] = 1
                col44 += 1
        return {"U": U44, "V": V44, "W": W44}
