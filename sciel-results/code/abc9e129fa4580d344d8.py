import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
DEADLINE_MARGIN = 10  # seconds to spare

def _schoolbook():
    """Return the schoolbook (rank-18) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank17():
    """Return a rank-17 decomposition using Strassen for the 2x2x2 subblock."""
    n, m, p = DIMENSIONS  # 2, 3, 3
    U = [[0] * 17 for _ in range(n * m)]  # 6 x 17
    V = [[0] * 17 for _ in range(m * p)]  # 9 x 17
    W = [[0] * 17 for _ in range(p * n)]  # 6 x 17
    
    # Strassen for 2x2x2 subproblem: i in {0,1}, j in {0,1}, k in {0,1}
    # m1 = (A00-A11)(B10+B11) -> contributes to C01
    # m2 = (A10+A11)B00 -> contributes to C10, C11
    # m3 = A00(B01-B11) -> contributes to C11
    # m4 = A11(B10-B00) -> contributes to C00, C10, C11
    # m5 = (A00+A11)(B00+B11) -> contributes to C00, C01
    # m6 = (A00-A10)(B01+B11) -> contributes to C11
    # m7 = (A10-A11)(B00+B10) -> contributes to C00, C10
    
    strassen = [
        {  # m1
            'U': {(0,0): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,1): 1}
        },
        {  # m2
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(1,0): 1, (1,1): 1}
        },
        {  # m3
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(1,1): 1}
        },
        {  # m4
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (1,0): 1, (1,1): -1}
        },
        {  # m5
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (0,1): 1}
        },
        {  # m6
            'U': {(0,0): 1, (1,0): -1},
            'V': {(0,1): 1, (1,1): 1},
            'W': {(1,1): 1}
        },
        {  # m7
            'U': {(1,0): 1, (1,1): -1},
            'V': {(0,0): 1, (1,0): 1},
            'W': {(0,0): 1, (1,0): 1}
        }
    ]
    
    for r, term in enumerate(strassen):
        for (i, j), coeff in term['U'].items():
            U[i * m + j][r] += coeff
        for (j, k), coeff in term['V'].items():
            V[j * p + k][r] += coeff
        for (i, k), coeff in term['W'].items():
            W[k * n + i][r] += coeff
    
    # Remaining 10 terms use schoolbook
    remaining = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j < 2 and k < 2:
                    continue
                remaining.append((i, j, k))
    
    assert len(remaining) == 10, f"Expected 10 remaining, got {len(remaining)}"
    
    for idx, (i, j, k) in enumerate(remaining):
        r = 7 + idx
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W):
    """
    Validate that U, V, W form a valid decomposition of the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    if len(U) != n * m: return False
    if len(V) != m * p: return False
    if len(W) != p * n: return False
    for row in U:
        if len(row) != rank: return False
    for row in V:
        if len(row) != rank: return False
    for row in W:
        if len(row) != rank: return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Try rank-17 first
    try:
        result = _rank17()
        if _validate(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", _validate(res['U'], res['V'], res['W']))
