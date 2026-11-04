import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W):
    """Verify exact tensor decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    one = Fraction(1, 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast float-based verification."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_rank88():
    """Build the known rank-88 decomposition using Strassen on 2x2x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: i=0, schoolbook 1x4x8
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 via Strassen 2x2x2 blocks
    sU = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (j0 + lj)][col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(j0 + lj) * p + (k0 + lk)][col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(k0 + lk) * n + (1 + li)][col] += sW[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}

def _build_laderman_2x3x3():
    """
    Build Laderman rank-14 decomposition for 2x3x3 matrix multiplication.
    Convention: U[i*3+j], V[j*3+k], W[k*2+i]
    """
    n, m, p = 2, 3, 3
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Laderman's 14 rank-1 terms for 2x3x3 matmul
    # A is 2x3, B is 3x3, C is 2x3
    # Indices: A[i*3+j], B[j*3+k], C[k*2+i]
    
    # Using the standard Laderman decomposition
    # Let me define the 14 terms explicitly
    
    terms = [
        # (u_indices, v_indices, w_indices)
        # Term 1: (a00+a11)*(b00+b11) -> c00
        ({0:1, 4:1}, {0:1, 4:1}, {0:1}),
        # Term 2: (a00+a11)*(b00+b11) -> c01  
        ({0:1, 4:1}, {0:1, 4:1}, {1:1}),
        # Term 3: (a00+a11)*(b00+b11) -> c02
        ({0:1, 4:1}, {0:1, 4:1}, {2:1}),
        # Term 4: (a00+a11)*(b10+b21) -> c10
        ({0:1, 4:1}, {3:1, 5:1}, {3:1}),
        # Term 5: (a00+a11)*(b10+b21) -> c11
        ({0:1, 4:1}, {3:1, 5:1}, {4:1}),
        # Term 6: (a00+a11)*(b10+b21) -> c12
        ({0:1, 4:1}, {3:1, 5:1}, {5:1}),
        # Term 7: (a01+a12)*(b00+b11) -> c00
        ({1:1, 5:1}, {0:1, 4:1}, {0:1}),
        # Term 8: (a01+a12)*(b00+b11) -> c01
        ({1:1, 5:1}, {0:1, 4:1}, {1:1}),
        # Term 9: (a01+a12)*(b00+b11) -> c02
        ({1:1, 5:1}, {0:1, 4:1}, {2:1}),
        # Term 10: (a01+a12)*(b10+b21) -> c10
        ({1:1, 5:1}, {3:1, 5:1}, {3:1}),
        # Term 11: (a01+a12)*(b10+b21) -> c11
        ({1:1, 5:1}, {3:1, 5:1}, {4:1}),
        # Term 12: (a01+a12)*(b10+b21) -> c12
        ({1:1, 5:1}, {3:1, 5:1}, {5:1}),
        # Term 13: a10*b22 -> c10
        ({3:1}, {8:1}, {3:1}),
        # Term 14: a11*b22 -> c11
        ({4:1}, {8:1}, {4:1}),
    ]
    
    # This isn't quite right. Let me use a known valid Laderman decomposition.
    # I'll construct it from the literature.
    
    # Actually, let me just return the rank-88 solution for now.
    # The Laderman improvement needs careful verification.
    
    raise NotImplementedError

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # Start with the known rank-88 solution
    best = _build_rank88()
    
    if not _verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best
