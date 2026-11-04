import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 10 decomposition
    rank = 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_u(i, j, c, val):
        U[i * m + j][c] = val
        
    def set_v(j, k, c, val):
        V[j * p + k][c] = val
        
    def set_w(k, i, c, val):
        W[k * n + i][c] = val

    # --- Block 1: A1 * B11 (2x2 * 2x2) using Strassen ---
    # Indices i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1 = (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)
    
    # M2 = (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)
    
    # M3 = A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)
    
    # M4 = A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)
    
    # M5 = (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)
    
    # M6 = (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)
    
    # M7 = (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # --- Block 2: A2 * B21 (2x1 * 1x2) ---
    # j=2, k in {0,1}, i in {0,1}
    # Rank 1
    c = 7
    set_u(0, 2, c, 1); set_u(1, 2, c, 1)
    set_v(2, 0, c, 1); set_v(2, 1, c, 1)
    set_w(0, 0, c, 1); set_w(1, 0, c, 1)
    set_w(0, 1, c, 1); set_w(1, 1, c, 1)
    # Wait, W index is k*n + i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # My set_w(k, i, c, val) uses W[k*n+i].
    # So set_w(0, 0, c, 1) is W[0].
    # set_w(1, 0, c, 1) is W[2].
    # Let's re-verify the W indices for Block 2.
    # C_{i,k}.
    # i=0, k=0 -> W[0*2+0]=W[0].
    # i=1, k=0 -> W[0*2+1]=W[1].
    # i=0, k=1 -> W[1*2+0]=W[2].
    # i=1, k=1 -> W[1*2+1]=W[3].
    
    # Correcting Block 2 W assignments:
    # set_w(0, 0, c, 1) -> W[0]
    # set_w(0, 1, c, 1) -> W[1]
    # set_w(1, 0, c, 1) -> W[2]
    # set_w(1, 1, c, 1) -> W[3]
    
    # I already called set_w(0, 0, c, 1) etc in the code above?
    # No, I wrote:
    # set_w(0, 0, c, 1); set_w(1, 0, c, 1)
    # set_w(0, 1, c, 1); set_w(1, 1, c, 1)
    # This covers all 4. Correct.

    # --- Block 3: A1 * B12 (2x2 * 2x1) ---
    # j in {0,1}, k=2, i in {0,1}
    # Rank 1
    c = 8
    set_u(0, 0, c, 1); set_u(0, 1, c, 1)
    set_u(1, 0, c, 1); set_u(1, 1, c, 1)
    set_v(0, 2, c, 1); set_v(1, 2, c, 1)
    # C_{i,2}.
    # i=0, k=2 -> W[2*2+0]=W[4].
    # i=1, k=2 -> W[2*2+1]=W[5].
    set_w(2, 0, c, 1); set_w(2, 1, c, 1)

    # --- Block 4: A2 * B22 (2x1 * 1x1) ---
    # j=2, k=2, i in {0,1}
    # Rank 1
    c = 9
    set_u(0, 2, c, 1); set_u(1, 2, c, 1)
    set_v(2, 2, c, 1)
    # C_{i,2}.
    set_w(2, 0, c, 1); set_w(2, 1, c, 1)

    return {"U": U, "V": V, "W": W}
