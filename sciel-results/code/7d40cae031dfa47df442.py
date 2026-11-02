"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = []
    # Term 0: M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # Term 1: M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    # Term 2: M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # Term 3: M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # Term 4: M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # Term 5: M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # Term 6: M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """Build a 4x4x4 decomposition using recursive Strassen (rank 49)."""
    n = 4
    terms_2x2 = _strassen_2x2()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            for (bi, bj), coeff_u in term1['U'].items():
                for (li, lj), coeff_v in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i*4 + j][idx] += coeff_u * coeff_v
            
            for (bj, bk), coeff_v in term1['V'].items():
                for (lj, lk), coeff_w in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j*4 + k][idx] += coeff_v * coeff_w
            
            for (bk, bi), coeff_w in term1['W'].items():
                for (lk, li), coeff_u2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k*4 + i][idx] += coeff_w * coeff_u2
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x5():
    """Build a 4x4x5 decomposition using Strassen for 4x4x4 part and schoolbook for 4x4x1 part.
    
    Rank = 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1) = 65.
    """
    n, m, p = DIMENSIONS
    
    base = _build_4x4x4_strassen()
    rank_base = 49
    rank_extra = n * m  # 16
    rank = rank_base + rank_extra
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):
        for r in range(rank_base):
            V[j][r] = base["V"][j][r]
    for k in range(4 * n):
        for r in range(rank_base):
            W[k][r] = base["W"][k][r]
    
    # Add 4x4x1 part (for column 4 of B)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Term for each (i, j):
    # U[i, j, r] = 1
    # V[j, 4, r] = 1
    # W[4, i, r] = 1
    for i in range(n):
        for j in range(m):
            r = rank_base + i * m + j
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Use the improved Strassen + Schoolbook decomposition (Rank 65)
    result = _build_4x4x5()
    
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Ensure all values are plain floats
        def _clean(M):
            return [[float(x) for x in row] for row in M]
        return {"U": _clean(result["U"]), "V": _clean(result["V"]), "W": _clean(result["W"])}
    
    # Fallback to schoolbook if verification fails (should not happen)
    schoolbook = _build_schoolbook()
    def _clean(M):
        return [[float(x) for x in row] for row in M]
    return {"U": _clean(schoolbook["U"]), "V": _clean(schoolbook["V"]), "W": _clean(schoolbook["W"])}
# EVOLVE-BLOCK-END
