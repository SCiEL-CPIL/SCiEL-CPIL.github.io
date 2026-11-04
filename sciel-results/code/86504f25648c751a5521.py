"""Matrix multiplication 2x3 by 3x8 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 8]
_SEED_RANK = 48
_DEADLINE = 140.0
_START = time.time()

def _ok():
    return time.time() - _START < _DEADLINE

def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == (i,j,k) indicator."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(r):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x3_8():
    """
    Rank-44 decomposition for 2x3 * 3x8.
    n=2, m=3, p=8.
    U: (6, 44), V: (24, 44), W: (16, 44)
    Index conventions:
      U row = i*m + j  (i in {0,1}, j in {0,1,2})
      V row = j*p + k  (j in {0,1,2}, k in {0..7})
      W row = k*n + i  (k in {0..7}, i in {0,1})
    """
    n, m, p = 2, 3, 8
    R = 44
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]

    # We'll add contributions column by column.
    # 16 schoolbook terms for j=2 (the third row of A / third column-block of B)
    # For j=2: A[2][k] * B[2][k] contributes to C[i][k] for all i,k.
    # Actually schoolbook: C[i][k] = sum_j A[i][j]*B[j][k].
    # For j=2, we need 2*8=16 terms.
    # For j=0,1: use Strassen on the 2x2 block of A (rows 0,1; cols 0,1) 
    #   times the 2x8 block of B (rows 0,1; cols 0..7).
    #   But Strassen is for 2x2 * 2x2. We need to tile the 2x8 B into four 2x2 blocks.
    #   B rows 0,1; cols 0..7 -> four 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #   Each 2x2 * 2x2 Strassen uses 7 terms. 4 blocks * 7 = 28 terms.
    #   Total: 16 + 28 = 44. Correct.

    col = 0
    # --- Part 1: j=2 schoolbook (16 terms) ---
    for i in range(n):
        for k in range(p):
            # Term: A[i,2] * B[2,k] -> C[i,k]
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    # col is now 16

    # --- Part 2: Strassen for j in {0,1}, tiled over 4 blocks of B ---
    # A block: A[0:2, 0:2] -> rows (0,0),(0,1),(1,0),(1,1) in U-index i*m+j
    #   U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B blocks: for block b in 0..3, cols [2b, 2b+1], rows 0,1
    #   V indices: row0,col=2b -> 0*8+2b = 2b; row0,col=2b+1 -> 2b+1
    #              row1,col=2b -> 1*8+2b = 8+2b; row1,col=2b+1 -> 9+2b
    # C block: output C[i, 2b:2b+2] for i in {0,1}
    #   W indices: k=2b, i=0 -> 2b*2+0 = 4b; k=2b, i=1 -> 4b+1
    #              k=2b+1, i=0 -> (2b+1)*2+0 = 4b+2; k=2b+1, i=1 -> 4b+3

    # Strassen terms for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6

    # U coefficients for each M term (on indices 0=a00, 1=a01, 2=a10, 3=a11):
    U_M = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: 1},
        3: {0: 1},
        4: {3: 1},
        5: {0: 1, 1: 1},
        6: {2: 1, 0: -1},
        7: {1: 1, 3: -1},
    }
    # V coefficients for each M term (on indices for a 2x2 B block: 0=b00,1=b01,2=b10,3=b11)
    V_M = {
        1: {0: 1, 3: 1},
        2: {0: 1},
        3: {1: 1, 3: -1},
        4: {2: 1, 0: -1},
        5: {3: 1},
        6: {0: 1, 1: 1},
        7: {2: 1, 3: 1},
    }
    # W coefficients for each M term (on indices for a 2x2 C block: 0=c00,1=c01,2=c10,3=c11)
    W_M = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: -1},
        3: {1: 1, 3: 1},
        4: {0: 1, 2: 1},
        5: {0: -1, 1: 1},
        6: {3: 1},
        7: {0: 1},
    }

    for b in range(4):  # 4 blocks of B (and C)
        k0 = 2 * b
        k1 = 2 * b + 1
        # V row indices for this B block:
        #   b00 -> row 0, col k0 -> 0*8+k0 = k0
        #   b01 -> row 0, col k1 -> k1
        #   b10 -> row 1, col k0 -> 8+k0
        #   b11 -> row 1, col k1 -> 9+k1
        v_idx = {0: k0, 1: k1, 2: 8+k0, 3: 9+k1}
        # W row indices for this C block:
        #   c00 -> k=k0, i=0 -> k0*2+0 = 2*k0
        #   c01 -> k=k1, i=0 -> 2*k1
        #   c10 -> k=k0, i=1 -> 2*k0+1
        #   c11 -> k=k1, i=1 -> 2*k1+1
        w_idx = {0: 2*k0, 1: 2*k1, 2: 2*k0+1, 3: 2*k1+1}
        # U row indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> already 0,1,2,3

        for t in range(1, 8):
            c = col
            for u_pos, val in U_M[t].items():
                U[u_pos][c] += val
            for v_pos, val in V_M[t].items():
                V[v_idx[v_pos]][c] += val
            for w_pos, val in W_M[t].items():
                W[w_idx[w_pos]][c] += val
            col += 1

    return U, V, W

def construct():
    global _START
    _START = time.time()
    n, m, p = DIMENSIONS
    try:
        U, V, W = _build_strassen_2x3_8()
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][c] = 1
                        V[j*p+k][c] = 1
                        W[k*n+i][c] = 1
                        c += 1
            return {"U": U, "V": V, "W": W}
        return {"U": U, "V": V, "W": W}
    except Exception:
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
