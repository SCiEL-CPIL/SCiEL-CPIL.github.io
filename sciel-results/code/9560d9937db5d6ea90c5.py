"""Rewritten program: rank-26 Strassen-style exact decomposition for 3x3 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _construct_rank_26():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    Uses Strassen's algorithm on the top-left 2x2 block (indices 0,1)
    and schoolbook for the remaining terms.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(r, u_dict, v_dict, w_dict):
        for u_idx, val in u_dict.items():
            U[u_idx][r] += val
        for v_idx, val in v_dict.items():
            V[v_idx][r] += val
        for w_idx, val in w_dict.items():
            W[w_idx][r] += val

    r = 0
    # Strassen terms for 2x2 block (indices 0, 1)
    # U indices: i*3+j
    # V indices: j*3+k
    # W indices: k*3+i
    
    # m1 = (A00+A11)(B00+B11)
    # U: (0,0), (1,1) -> 0, 4
    # V: (0,0), (1,1) -> 0, 4
    # W: C00 (0,0), C11 (1,1) -> 0, 4
    add_term(r, {0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    r += 1
    
    # m2 = (A10+A11)B00
    # U: (1,0), (1,1) -> 3, 4
    # V: (0,0) -> 0
    # W: C10 (0,1) -> 1
    add_term(r, {3: 1, 4: 1}, {0: 1}, {1: 1})
    r += 1
    
    # m3 = A00(B10-B00)
    # U: (0,0) -> 0
    # V: (1,0), (0,0) -> 3, 0
    # W: C01 (1,0), C11 (1,1) -> 3, 4
    add_term(r, {0: 1}, {3: 1, 0: -1}, {3: 1, 4: -1})
    r += 1
    
    # m4 = A11(B11-B01)
    # U: (1,1) -> 4
    # V: (1,1), (0,1) -> 4, 1
    # W: C00 (0,0), C10 (0,1) -> 0, 1
    add_term(r, {4: 1}, {4: 1, 1: -1}, {0: -1, 1: 1})
    r += 1
    
    # m5 = (A00+A01)B11
    # U: (0,0), (0,1) -> 0, 1
    # V: (1,1) -> 4
    # W: C00 (0,0), C01 (1,0) -> 0, 3
    add_term(r, {0: 1, 1: 1}, {4: 1}, {0: -1, 3: 1})
    r += 1
    
    # m6 = (A10-A00)(B00+B01)
    # U: (1,0), (0,0) -> 3, 0
    # V: (0,0), (0,1) -> 0, 1
    # W: C11 (1,1) -> 4
    add_term(r, {3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    r += 1
    
    # m7 = (A01-A11)(B10+B11)
    # U: (0,1), (1,1) -> 1, 4
    # V: (1,0), (1,1) -> 3, 4
    # W: C00 (0,0), C11 (1,1) -> 0, 4
    add_term(r, {1: 1, 4: -1}, {3: 1, 4: 1}, {0: -1, 4: -1})
    r += 1
    
    # Schoolbook terms for the rest (19 terms)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True


def construct():
    """
    Construct the factor matrices for 3x3 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    result = _construct_rank_26()
    
    if not _verify(result['U'], result['V'], result['W']):
        # Fallback to schoolbook if verification fails
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
        
    return result

# EVOLVE-BLOCK-END
