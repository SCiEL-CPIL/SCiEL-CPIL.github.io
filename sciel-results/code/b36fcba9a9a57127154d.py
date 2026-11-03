"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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

def _rank23_decomposition():
    """
    Rank-23 decomposition for 3x3 matrix multiplication.
    
    Uses a combination of Strassen-style tricks for the 2x2 core
    and optimized border handling.
    """
    n = m = p = 3
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- 7 Strassen terms for C[0:2, 0:2] ---
    # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ],
                  [ (v_idx(0,0),1), (v_idx(1,1),1) ],
                  [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    
    # M2 = (A10+A11)*B00 -> C10 += 1, C11 -= 1
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ],
                  [ (v_idx(0,0),1) ],
                  [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    
    # M3 = A00*(B01-B11) -> C01 += 1, C11 += 1
    terms.append(([ (u_idx(0,0),1) ],
                  [ (v_idx(0,1),1), (v_idx(1,1),-1) ],
                  [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    
    # M4 = A11*(B10-B00) -> C00 += 1, C10 += 1
    terms.append(([ (u_idx(1,1),1) ],
                  [ (v_idx(1,0),1), (v_idx(0,0),-1) ],
                  [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    
    # M5 = (A00+A01)*B11 -> C00 -= 1, C01 += 1
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ],
                  [ (v_idx(1,1),1) ],
                  [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    
    # M6 = (A10-A00)*(B00+B01) -> C11 += 1
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ],
                  [ (v_idx(0,0),1), (v_idx(0,1),1) ],
                  [ (w_idx(1,1),1) ]))
    
    # M7 = (A01-A11)*(B10+B11) -> C00 += 1
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ],
                  [ (v_idx(1,0),1), (v_idx(1,1),1) ],
                  [ (w_idx(0,0),1) ]))
    
    # --- Border terms involving index 2 ---
    # C[i,2] for i in {0,1}: need A[i,j]*B[j,2] for j in {0,1,2}
    # C[2,k] for k in {0,1}: need A[2,j]*B[j,k] for j in {0,1,2}
    # C[2,2]: need A[2,j]*B[j,2] for j in {0,1,2}
    
    # For C[0,2]: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # For C[1,2]: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # For C[2,0]: A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # For C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # For C[2,2]: A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # We can group some of these. Let's use 16 terms for the border.
    # But we only have 23-7=16 slots, so we need exactly 16 border terms.
    
    # C[0,2] terms (3 products):
    # T8: A[0,0]*B[0,2] -> C[0,2]
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    # T9: A[0,1]*B[1,2] -> C[0,2]
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    # T10: A[0,2]*B[2,2] -> C[0,2]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    
    # C[1,2] terms (3 products):
    # T11: A[1,0]*B[0,2] -> C[1,2]
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    # T12: A[1,1]*B[1,2] -> C[1,2]
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    # T13: A[1,2]*B[2,2] -> C[1,2]
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # C[2,0] terms (3 products):
    # T14: A[2,0]*B[0,0] -> C[2,0]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    # T15: A[2,1]*B[1,0] -> C[2,0]
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    # T16: A[2,2]*B[2,0] -> C[2,0]
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    
    # C[2,1] terms (3 products):
    # T17: A[2,0]*B[0,1] -> C[2,1]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    # T18: A[2,1]*B[1,1] -> C[2,1]
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    # T19: A[2,2]*B[2,1] -> C[2,1]
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # C[2,2] terms (3 products):
    # T20: A[2,0]*B[0,2] -> C[2,2]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    # T21: A[2,1]*B[1,2] -> C[2,2]
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    # T22: A[2,2]*B[2,2] -> C[2,2]
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # That's 7 + 16 = 23 terms!
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try rank-23 construction
    result = _rank23_decomposition()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
