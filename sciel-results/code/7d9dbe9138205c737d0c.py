"""Structural redesign: exploit n=2 to lift a 4x4 product to 2x4x4,
then perform an exact rank-reduction search (drop / merge columns)."""
import time
import random

DIMENSIONS = [2, 4, 4]
SEED = 20240517
_DEADLINE = time.time() + 150.0
_MARGIN = 15.0


def _mul_tensor(n, m, p, U, V, W):
    """Reconstruct C[i*n+k, j*n+l] = sum_{a,b} A[i*n+a, j*n+b]*B[a*n+k, b*n+l]."""
    r = len(U[0])
    T = {}
    for c in range(r):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        nzU = [a for a in range(n * m) if u[a] != 0]
        nzV = [b for b in range(m * p) if v[b] != 0]
        nzW = [d for d in range(p * n) if w[d] != 0]
        for a in nzU:
            i, j = divmod(a, m)
            ua = u[a]
            for b in nzV:
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                uva = ua * v[b]
                for d in nzW:
                    l, i2 = divmod(d, n)
                    if i2 != i:
                        continue
                    T[(i, l, j, k)] = T.get((i, l, j, k), 0) + uva * w[d]
    return T


def _build_4x4_rank16():
    """Rank-16 (schoolbook) decomposition of a 4x4 product, tensor T[a*4+c, b*4+d]."""
    n = m = p = 4
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                U[a * 4 + b][col] = 1
                V[b * 4 + c][col] = 1
                W[c * 4 + a][col] = 1
                col += 1
    return U, V, W


def _lift(U4, V4, W4):
    """Lift a 4x4 rank-r decomposition to a 2x4x4 rank-2r decomposition.
    C0 = A0*B0 (block i=0), C1 = A1*B1 (block i=1), independent."""
    n = m = p = 4
    r = len(U4[0])
    rank = 2 * r
    U = [[0] * rank for _ in range(2 * 4)]
    V = [[0] * rank for _ in range(4 * 4)]
    W = [[0] * rank for _ in range(4 * 2)]
    for c in range(r):
        # block i=0
        for a in range(4):
            for b in range(4):
                U[0 * 4 + b][c] += U4[a * 4 + b][c]
        for b in range(4):
            for cidx in range(4):
                V[b * 4 + cidx][c] += V4[b * 4 + cidx][c]
        for cidx in range(4):
            for i in range(2):
                W[cidx * 2 + i][c] += W4[cidx * 4 + 0][c]
        # block i=1
        for a in range(4):
            for b in range(4):
                U[1 * 4 + b][c + r] += U4[a * 4 + b][c]
        for b in range(4):
            for cidx in range(4):
                V[b * 4 + cidx][c + r] += V4[b * 4 + cidx][c]
        for cidx in range(4):
            for i in range(2):
                W[cidx * 2 + i][c + r] += W4[cidx * 4 + 1][c]
    return U, V, W


def _drop_column(U, V, W, c):
    r = len(U[0])
    U2 = [row[:c] + row[c + 1:] for row in U]
    V2 = [row[:c] + row[c + 1:] for row in V]
    W2 = [row[:c] + row[c + 1:] for row in W]
    return U2, V2, W2


def _merge_columns(U, V, W, c1, c2):
    """Absorb column c2 into column c1 (c1 += c2) then drop c2."""
    r = len(U[0])
    U2 = []
    for row in U:
        U2.append([x + (row[c2] if i == c1 else 0) if i != c2 else 0 for i, x in enumerate(row)])
    V2 = []
    for row in V:
        V2.append([x + (row[c2] if i == c1 else 0) if i != c2 else 0 for i, x in enumerate(row)])
    W2 = []
    for row in W:
        W2.append([x + (row[c2] if i == c1 else 0) if i != c2 else 0 for i, x in enumerate(row)])
    # remove column c2 (which is now zero)
    U3 = [row[:c2] + row[c2 + 1:] for row in U2]
    V3 = [row[:c2] + row[c2 + 1:] for row in V2]
    W3 = [row[:c2] + row[c2 + 1:] for row in W2]
    return U3, V3, W3


def _validate(U, V, W):
    n, m, p = DIMENSIONS
    # Build target tensor T[(i,l,j,k)] = delta(i==l)*delta(j==k)
    target = {}
    for i in range(n):
        for l in range(p):
            for j in range(m):
                for k in range(p):
                    if i == l and j == k:
                        target[(i, l, j, k)] = 1
    got = _mul_tensor(n, m, p, U, V, W)
    if set(got.keys()) != set(target.keys()):
        return False
    for key, val in target.items():
        if got.get(key, 0) != val:
            return False
    return True


def _rank_reduction_search(U, V, W, rng, deadline):
    """Try to reduce rank by dropping / merging columns, validating exactly."""
    n, m, p = DIMENSIONS
    best_rank = len(U[0])
    best = (U, V, W)
    current = (U, V, W)
    U, V, W = current
    r = len(U[0])
    # Iterate: try to drop one column at a time until stuck.
    while True:
        if time.time() > deadline:
            break
        r = len(U[0])
        # Try dropping each column
        dropped = False
        order = list(range(r))
        rng.shuffle(order)
        for c in order:
            U2, V2, W2 = _drop_column(U, V, W, c)
            if _validate(U2, V2, W2):
                U, V, W = U2, V2, W2
                best_rank = len(U[0])
                best = (U, V, W)
                dropped = True
                break
        if not dropped:
            # Try merging pairs (absorb c2 into c1)
            merged = False
            pairs = [(a, b) for a in range(r) for b in range(r) if a != b]
            rng.shuffle(pairs)
            for c1, c2 in pairs:
                U2, V2, W2 = _merge_columns(U, V, W, c1, c2)
                if _validate(U2, V2, W2):
                    U, V, W = U2, V2, W2
                    best_rank = len(U[0])
                    best = (U, V, W)
                    merged = True
                    break
            if not merged:
                break
    return best_rank, best


def construct():
    rng = random.Random(SEED)
    deadline = _DEADLINE - _MARGIN

    U4, V4, W4 = _build_4x4_rank16()
    U, V, W = _lift(U4, V4, W4)
    assert _validate(U, V, W), "lifted rank-16 must be valid"

    best_rank, (bU, bV, bW) = _rank_reduction_search(U, V, W, rng, deadline)

    # Fallback: return the best valid found (at least rank-16).
    return {"U": bU, "V": bV, "W": bW}
