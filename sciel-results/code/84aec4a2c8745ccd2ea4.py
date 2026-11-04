import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Rank-56 decomposition using 8 disjoint 2x2x2 Strassen blocks.
    # A is 2x4, B is 4x8.
    # Split j (0..3) into pairs: (0,1) and (2,3).
    # Split k (0..7) into pairs: (0,1), (2,3), (4,5), (6,7).
    # This creates 2 * 4 = 8 blocks of size 2x2x2.
    # Each 2x2x2 block has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.

    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x rank
    V = [[0] * rank for _ in range(m * p)]   # 32 x rank
    W = [[0] * rank for _ in range(p * n)]   # 16 x rank

    def add_u(i, j, col, val):
        U[i * m + j][col] += val
    def add_v(j, k, col, val):
        V[j * p + k][col] += val
    def add_w(k, i, col, val):
        W[k * n + i][col] += val

    def strassen_terms(j0, k0):
        # Returns list of (u_dict, v_dict, w_dict) for a 2x2x2 block
        # starting at j=j0, k=k0.
        # Indices in dicts are (i, j) for U, (j, k) for V, (k, i) for W.
        terms = []
        # T1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        terms.append((
            {(0, j0): 1, (1, j0+1): 1},
            {(j0, k0): 1, (j0+1, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1}
        ))
        # T2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        terms.append((
            {(1, j0): 1, (1, j0+1): 1},
            {(j0, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1}
        ))
        # T3: A00(B01-B11) -> C01 += 1, C11 += 1
        terms.append((
            {(0, j0): 1},
            {(j0, k0+1): 1, (j0+1, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1}
        ))
        # T4: A11(B10-B00) -> C00 += 1, C10 += 1
        terms.append((
            {(1, j0+1): 1},
            {(j0+1, k0): 1, (j0, k0): -1},
            {(k0, 0): 1, (k0, 1): 1}
        ))
        # T5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        terms.append((
            {(0, j0): 1, (0, j0+1): 1},
            {(j0+1, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1}
        ))
        # T6: (A10-A00)(B00+B01) -> C11 += 1
        terms.append((
            {(1, j0): 1, (0, j0): -1},
            {(j0, k0): 1, (j0, k0+1): 1},
            {(k0+1, 1): 1}
        ))
        # T7: (A01-A11)(B10+B11) -> C00 += 1
        terms.append((
            {(0, j0+1): 1, (1, j0+1): -1},
            {(j0+1, k0): 1, (j0+1, k0+1): 1},
            {(k0, 0): 1}
        ))
        return terms

    col = 0
    for jp in range(2):
        for kp in range(4):
            if _time_left() < 0.5:
                break
            j0 = jp * 2
            k0 = kp * 2
            for (ud, vd, wd) in strassen_terms(j0, k0):
                for (ii, jj), cv in ud.items():
                    if cv:
                        U[ii * m + jj][col] += cv
                for (jj, kk), cv in vd.items():
                    if cv:
                        V[jj * p + kk][col] += cv
                for (kk, ii), cv in wd.items():
                    if cv:
                        W[kk * n + ii][col] += cv
                col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
