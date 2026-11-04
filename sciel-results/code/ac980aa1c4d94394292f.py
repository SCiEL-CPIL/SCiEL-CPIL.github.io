"""Tensor decomposition for 4x4 x 4x5 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _winograd_222_core():
    """Return list of 7 triples (uvec,vvec,wvec) for 2x2x2 multiply, rank 7.
    uvec indexed by i*2+j, vvec by j*2+k, wvec by k*2+i, i,j,k in {0,1}.
    """
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    U[0][0] += 1; U[3][0] += 1; V[0][0] += 1; V[3][0] += 1; W[0][0] += 1; W[2][0] += 1
    U[2][1] += 1; U[3][1] += 1; V[0][1] += 1; W[0][1] += 1; W[3][1] += 1
    U[0][2] += 1; V[2][2] += 1; V[0][2] -= 1; W[1][0] += 1
    U[3][3] += 1; V[1][3] += 1; V[3][3] -= 1; W[1][1] += 1
    U[0][4] += 1; U[1][4] += 1; V[3][4] += 1; W[1][1] += 1; W[2][1] += 1
    U[2][5] += 1; U[0][5] -= 1; V[0][5] += 1; V[1][5] += 1; W[0][0] += 1; W[1][0] += 1
    U[1][6] += 1; U[3][6] -= 1; V[2][6] += 1; V[3][6] += 1; W[0][1] += 1; W[2][1] += 1
    core = []
    for c in range(7):
        uvec = [U[r][c] for r in range(4)]
        vvec = [V[r][c] for r in range(4)]
        wvec = [W[r][c] for r in range(4)]
        core.append((uvec, vvec, wvec))
    return core


def _build_block(i0, j0):
    """2x2x5 block for i in {i0,i0+1}, j in {j0,j0+1}, k in {0..4}.
    Global: U row = i*4+j, V row = j*5+k, W row = k*4+i.
    Returns list of (gu,gv,gw).
    """
    core = _winograd_222_core()
    terms = []
    for (uvec, vvec, wvec) in core:
        gu = [0.0]*16
        gv = [0.0]*20
        gw = [0.0]*20
        for r in range(4):
            i = r // 2; j = r % 2
            gu[(i0+i)*4 + (j0+j)] = uvec[r]
        for r in range(4):
            j = r // 2; k = r % 2
            gv[(j0+j)*5 + k] = vvec[r]
        for r in range(4):
            k = r // 2; i = r % 2
            gw[k*4 + (i0+i)] = wvec[r]
        terms.append((gu, gv, gw))
    return terms


def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS
    all_terms = []
    for i0 in (0, 2):
        for j0 in (0, 2):
            all_terms.extend(_build_block(i0, j0))
    rank = len(all_terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    for c, (gu, gv, gw) in enumerate(all_terms):
        for r in range(n*m):
            if gu[r]: U[r][c] = gu[r]
        for r in range(m*p):
            if gv[r]: V[r][c] = gv[r]
        for r in range(p*n):
            if gw[r]: W[r][c] = gw[r]
    # Verify.
    ok = True
    for i in range(n):
        for k in range(p):
            expected = 1.0 if (k < m and i == k) else 0.0
            s = 0.0
            for c in range(rank):
                inner = 0.0
                for j in range(m):
                    inner += U[i*m+j][c] * V[j*p+k][c]
                s += W[k*n+i][c] * inner
            if abs(s - expected) > 1e-6:
                ok = False
                break
        if not ok:
            break
    if not ok:
        rank = n*m*p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
