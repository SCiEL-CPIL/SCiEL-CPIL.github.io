import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # 1. Add Strassen block for A_L * B_T1
    # A: i in {0,1}, j in {0,1}
    # B: j in {0,1}, k in {0,1}
    # C: k in {0,1}, i in {0,1}
    col = 0
    # M1 = (A00 + A11)(B00 + B11)
    U[u_idx(0,0), col] += 1; U[u_idx(1,1), col] += 1
    V[v_idx(0,0), col] += 1; V[v_idx(1,1), col] += 1
    W[w_idx(0,0), col] += 1; W[w_idx(1,1), col] += 1
    col += 1
    
    # M2 = (A10 + A11)B00
    U[u_idx(1,0), col] += 1; U[u_idx(1,1), col] += 1
    V[v_idx(0,0), col] += 1
    W[w_idx(0,1), col] += 1
    col += 1
    
    # M3 = A00(B01 - B11)
    U[u_idx(0,0), col] += 1
    V[v_idx(0,1), col] += 1; V[v_idx(1,1), col] -= 1
    W[w_idx(1,0), col] += 1; W[w_idx(1,1), col] += 1
    col += 1
    
    # M4 = A11(B10 - B00)
    U[u_idx(1,1), col] += 1
    V[v_idx(1,0), col] += 1; V[v_idx(0,0), col] -= 1
    W[w_idx(0,0), col] += 1; W[w_idx(0,1), col] += 1
    col += 1
    
    # M5 = (A00 + A01)B11
    U[u_idx(0,0), col] += 1; U[u_idx(0,1), col] += 1
    V[v_idx(1,1), col] += 1
    W[w_idx(0,0), col] -= 1; W[w_idx(1,0), col] += 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[u_idx(1,0), col] += 1; U[u_idx(0,0), col] -= 1
    V[v_idx(0,0), col] += 1; V[v_idx(0,1), col] += 1
    W[w_idx(1,1), col] += 1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[u_idx(0,1), col] += 1; U[u_idx(1,1), col] -= 1
    V[v_idx(1,0), col] += 1; V[v_idx(1,1), col] += 1
    W[w_idx(0,0), col] += 1
    col += 1

    # 2. Add Strassen block for A_L * B_T2
    # A: i in {0,1}, j in {0,1}
    # B: j in {0,1}, k in {2,3}
    # C: k in {2,3}, i in {0,1}
    # Same structure, shift k by 2
    # M1
    U[u_idx(0,0), col] += 1; U[u_idx(1,1), col] += 1
    V[v_idx(0,2), col] += 1; V[v_idx(1,3), col] += 1
    W[w_idx(2,0), col] += 1; W[w_idx(3,1), col] += 1
    col += 1
    
    # M2
    U[u_idx(1,0), col] += 1; U[u_idx(1,1), col] += 1
    V[v_idx(0,2), col] += 1
    W[w_idx(2,1), col] += 1
    col += 1
    
    # M3
    U[u_idx(0,0), col] += 1
    V[v_idx(0,3), col] += 1; V[v_idx(1,3), col] -= 1
    W[w_idx(3,0), col] += 1; W[w_idx(3,1), col] += 1
    col += 1
    
    # M4
    U[u_idx(1,1), col] += 1
    V[v_idx(1,2), col] += 1; V[v_idx(0,2), col] -= 1
    W[w_idx(2,0), col] += 1; W[w_idx(2,1), col] += 1
    col += 1
    
    # M5
    U[u_idx(0,0), col] += 1; U[u_idx(0,1), col] += 1
    V[v_idx(1,3), col] += 1
    W[w_idx(2,0), col] -= 1; W[w_idx(3,0), col] += 1
    col += 1
    
    # M6
    U[u_idx(1,0), col] += 1; U[u_idx(0,0), col] -= 1
    V[v_idx(0,2), col] += 1; V[v_idx(0,3), col] += 1
    W[w_idx(3,1), col] += 1
    col += 1
    
    # M7
    U[u_idx(0,1), col] += 1; U[u_idx(1,1), col] -= 1
    V[v_idx(1,2), col] += 1; V[v_idx(1,3), col] += 1
    W[w_idx(2,0), col] += 1
    col += 1

    # 3. Add Schoolbook for A_L * B_T3
    # A: i in {0,1}, j in {0,1}
    # B: j in {0,1}, k in {4}
    # C: k in {4}, i in {0,1}
    # C40 = A00 B04 + A01 B14
    # C41 = A10 B04 + A11 B14
    
    # A00 B04 -> C40
    U[u_idx(0,0), col] += 1
    V[v_idx(0,4), col] += 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # A01 B14 -> C40
    U[u_idx(0,1), col] += 1
    V[v_idx(1,4), col] += 1
    W[w_idx(4,0), col] += 1
    col += 1
    
    # A10 B04 -> C41
    U[u_idx(1,0), col] += 1
    V[v_idx(0,4), col] += 1
    W[w_idx(4,1), col] += 1
    col += 1
    
    # A11 B14 -> C41
    U[u_idx(1,1), col] += 1
    V[v_idx(1,4), col] += 1
    W[w_idx(4,1), col] += 1
    col += 1

    # 4. Add Schoolbook for A_R * B_B
    # A: i in {0,1}, j in {2}
    # B: j in {2}, k in {0..4}
    # C: k in {0..4}, i in {0,1}
    # Ck0 = A02 B2k
    # Ck1 = A12 B2k
    
    for k in range(5):
        # A02 B2k -> Ck0
        U[u_idx(0,2), col] += 1
        V[v_idx(2,k), col] += 1
        W[w_idx(k,0), col] += 1
        col += 1
        
        # A12 B2k -> Ck1
        U[u_idx(1,2), col] += 1
        V[v_idx(2,k), col] += 1
        W[w_idx(k,1), col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
