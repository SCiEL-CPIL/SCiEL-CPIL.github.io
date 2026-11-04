import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Block 1: A1 (2x2) * B1a (2x2) -> C1a (2x2)
    # A1 indices: i in {0,1}, j in {0,1}
    # B1a indices: j in {0,1}, k in {0,1}
    # C1a indices: i in {0,1}, k in {0,1}
    
    # Strassen M1: (A00+A11)(B00+B11)
    c = 0
    U[u_idx(0,0)][c] += 1; U[u_idx(1,1)][c] += 1
    V[v_idx(0,0)][c] += 1; V[v_idx(1,1)][c] += 1
    W[w_idx(0,0)][c] += 1; W[w_idx(1,1)][c] += 1
    W[w_idx(1,0)][c] -= 1; W[w_idx(0,1)][c] -= 1 # Wait, C00 and C11 get +1, C10 and C01 get -1?
    # Let's re-verify Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # C00 += 1
    # C11 += 1
    # C01 += 0
    # C10 += 0
    # My previous line was wrong.
    W[w_idx(0,0)][c] = 1
    W[w_idx(1,1)][c] = 1
    # W[w_idx(1,0)][c] and W[w_idx(0,1)][c] are 0 for M1.
    
    # M2: (A10+A11)B00
    c = 1
    U[u_idx(1,0)][c] += 1; U[u_idx(1,1)][c] += 1
    V[v_idx(0,0)][c] += 1
    W[w_idx(1,0)][c] += 1
    W[w_idx(1,1)][c] -= 1
    
    # M3: A00(B01-B11)
    c = 2
    U[u_idx(0,0)][c] += 1
    V[v_idx(0,1)][c] += 1; V[v_idx(1,1)][c] -= 1
    W[w_idx(0,1)][c] += 1
    W[w_idx(1,1)][c] += 1
    
    # M4: A11(B10-B00)
    c = 3
    U[u_idx(1,1)][c] += 1
    V[v_idx(1,0)][c] += 1; V[v_idx(0,0)][c] -= 1
    W[w_idx(0,0)][c] += 1
    W[w_idx(1,0)][c] += 1
    
    # M5: (A00+A01)B11
    c = 4
    U[u_idx(0,0)][c] += 1; U[u_idx(0,1)][c] += 1
    V[v_idx(1,1)][c] += 1
    W[w_idx(0,0)][c] -= 1
    W[w_idx(0,1)][c] += 1
    
    # M6: (A10-A00)(B00+B01)
    c = 5
    U[u_idx(1,0)][c] += 1; U[u_idx(0,0)][c] -= 1
    V[v_idx(0,0)][c] += 1; V[v_idx(0,1)][c] += 1
    W[w_idx(1,1)][c] += 1
    
    # M7: (A01-A11)(B10+B11)
    c = 6
    U[u_idx(0,1)][c] += 1; U[u_idx(1,1)][c] -= 1
    V[v_idx(1,0)][c] += 1; V[v_idx(1,1)][c] += 1
    W[w_idx(0,0)][c] += 1
    
    # Block 2: A1 (2x2) * B1b (2x1) -> C1b (2x1)
    # A1 indices: i in {0,1}, j in {0,1}
    # B1b indices: j in {0,1}, k=2
    # C1b indices: i in {0,1}, k=2
    
    # M8: A00 B02 -> C02
    c = 7
    U[u_idx(0,0)][c] += 1
    V[v_idx(0,2)][c] += 1
    W[w_idx(2,0)][c] += 1
    
    # M9: A01 B12 -> C02
    c = 8
    U[u_idx(0,1)][c] += 1
    V[v_idx(1,2)][c] += 1
    W[w_idx(2,0)][c] += 1
    
    # M10: A10 B02 -> C12
    c = 9
    U[u_idx(1,0)][c] += 1
    V[v_idx(0,2)][c] += 1
    W[w_idx(2,1)][c] += 1
    
    # M11: A11 B12 -> C12
    c = 10
    U[u_idx(1,1)][c] += 1
    V[v_idx(1,2)][c] += 1
    W[w_idx(2,1)][c] += 1
    
    # Block 3: A2 (2x1) * B2 (1x3) -> C2 (2x3)
    # A2 indices: i in {0,1}, j=2
    # B2 indices: j=2, k in {0,1,2}
    # C2 indices: i in {0,1}, k in {0,1,2}
    
    # M12: A02 B20 -> C00
    c = 11
    U[u_idx(0,2)][c] += 1
    V[v_idx(2,0)][c] += 1
    W[w_idx(0,0)][c] += 1
    
    # M13: A02 B21 -> C01
    c = 12
    U[u_idx(0,2)][c] += 1
    V[v_idx(2,1)][c] += 1
    W[w_idx(0,1)][c] += 1
    
    # M14: A02 B22 -> C02
    c = 13
    U[u_idx(0,2)][c] += 1
    V[v_idx(2,2)][c] += 1
    W[w_idx(2,0)][c] += 1
    
    # M15: A12 B20 -> C10
    c = 14
    U[u_idx(1,2)][c] += 1
    V[v_idx(2,0)][c] += 1
    W[w_idx(0,1)][c] += 1 # Wait, C10 is w_idx(0,1)? No.
    # w_idx(k, i). C10 means i=1, k=0. So w_idx(0, 1).
    # C11 means i=1, k=1. So w_idx(1, 1).
    # C12 means i=1, k=2. So w_idx(2, 1).
    
    # Let's re-check W indices for Block 3.
    # C00: i=0, k=0 -> w_idx(0,0)
    # C01: i=0, k=1 -> w_idx(1,0)
    # C02: i=0, k=2 -> w_idx(2,0)
    # C10: i=1, k=0 -> w_idx(0,1)
    # C11: i=1, k=1 -> w_idx(1,1)
    # C12: i=1, k=2 -> w_idx(2,1)
    
    # So M15 (A12 B20) contributes to C10.
    W[w_idx(0,1)][c] += 1
    
    # M16: A12 B21 -> C11
    c = 15
    U[u_idx(1,2)][c] += 1
    V[v_idx(2,1)][c] += 1
    W[w_idx(1,1)][c] += 1
    
    # M17: A12 B22 -> C12
    c = 16
    U[u_idx(1,2)][c] += 1
    V[v_idx(2,2)][c] += 1
    W[w_idx(2,1)][c] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
