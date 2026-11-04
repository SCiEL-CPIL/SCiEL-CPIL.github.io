"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 3, 4]
_START = time.time()
DEADLINE = 150.0 - 20.0  # safety margin

def _remaining():
    return DEADLINE - (time.time() - _START)

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    # Check sum_r U[(i',j'),r] V[(j'',k''),r] W[(k,i),r] == delta[i',i] delta[j',j''] delta[j'',k]
    rank = len(U[0])
    for ip in range(n):
        for jp in range(m):
            for jpp in range(m):
                for kpp in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[ip*m+jp][r] * V[jpp*p+kpp][r] * W[k*n+i][r]
                            expected = (1.0 if (ip==i and jp==jpp and jpp==k) else 0.0)
                            if abs(s - expected) > 1e-6:
                                return False
    return True

def _is_halfint(x):
    return abs(x - round(2*x)/2) < 1e-6

def _to_halfint_list(mat):
    out = []
    for row in mat:
        nr = []
        for x in row:
            h = round(2*x)
            val = h/2.0
            if abs(x - val) > 1e-6:
                return None
            nr.append(val)
        out.append(nr)
    return out

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_schoolbook(n, m, p)
    best = {"U": U, "V": V, "W": W}
    # verify schoolbook (sanity)
    if _verify(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
    else:
        # fallback: keep as-is
        pass
    return best
# EVOLVE-BLOCK-END
