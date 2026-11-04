import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    rank = 39
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def set_u(i, j, col, val):
        U[i * m + j, col] = val
        
    def set_v(j, k, col, val):
        V[j * p + k, col] = val
        
    def set_w(k, i, col, val):
        W[k * n + i, col] = val

    def add_strassen_222(j_start, k_start, base_rank):
        # Adds 7 rank-1 terms for A[i, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]
        # Local indices:
        # j0 = j_start, j1 = j_start+1
        # k0 = k_start, k1 = k_start+1
        # i is 0 and 1 (global, since n=2)
        
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        col = base_rank
        set_u(0, j_start, col, 1)
        set_u(1, j_start + 1, col, 1)
        set_v(j_start, k_start, col, 1)
        set_v(j_start + 1, k_start + 1, col, 1)
        set_w(k_start, 0, col, 1)      # C00
        set_w(k_start + 1, 1, col, 1)  # C11
        
        # M2 = (A10 + A11)B00 -> C10
        col = base_rank + 1
        set_u(1, j_start, col, 1)
        set_u(1, j_start + 1, col, 1)
        set_v(j_start, k_start, col, 1)
        set_w(k_start, 1, col, 1)      # C10
        
        # M3 = A00(B01 - B11) -> C01
        col = base_rank + 2
        set_u(0, j_start, col, 1)
        set_v(j_start, k_start + 1, col, 1)
        set_v(j_start + 1, k_start + 1, col, -1)
        set_w(k_start + 1, 0, col, 1)  # C01
        
        # M4 = A11(B10 - B00) -> C00, C10
        col = base_rank + 3
        set_u(1, j_start + 1, col, 1)
        set_v(j_start + 1, k_start, col, 1)
        set_v(j_start, k_start, col, -1)
        set_w(k_start, 0, col, 1)      # C00
        set_w(k_start, 1, col, 1)      # C10
        
        # M5 = (A00 + A01)B11 -> C00(-), C01(+)
        col = base_rank + 4
        set_u(0, j_start, col, 1)
        set_u(0, j_start + 1, col, 1)
        set_v(j_start + 1, k_start + 1, col, 1)
        set_w(k_start, 0, col, -1)     # C00
        set_w(k_start + 1, 0, col, 1)  # C01
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        col = base_rank + 5
        set_u(1, j_start, col, 1)
        set_u(0, j_start, col, -1)
        set_v(j_start, k_start, col, 1)
        set_v(j_start, k_start + 1, col, 1)
        set_w(k_start + 1, 1, col, 1)  # C11
        
        # M7 = (A01 - A11)(B10 + B11) -> C00
        col = base_rank + 6
        set_u(0, j_start + 1, col, 1)
        set_u(1, j_start + 1, col, -1)
        set_v(j_start + 1, k_start, col, 1)
        set_v(j_start + 1, k_start + 1, col, 1)
        set_w(k_start, 0, col, 1)      # C00

    def add_rank4_221(j_start, k_start, base_rank):
        # Adds 4 rank-1 terms for A[i, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+1]
        # k is single, so k0 = k_start
        # C00 (global i=0, k=k_start) = A00*B00 + A01*B10
        # C10 (global i=1, k=k_start) = A10*B00 + A11*B10
        
        # Term 1: A[0, j0] * B[j0, k0] -> C[0, k0]
        col = base_rank
        set_u(0, j_start, col, 1)
        set_v(j_start, k_start, col, 1)
        set_w(k_start, 0, col, 1)
        
        # Term 2: A[0, j1] * B[j1, k0] -> C[0, k0]
        col = base_rank + 1
        set_u(0, j_start + 1, col, 1)
        set_v(j_start + 1, k_start, col, 1)
        set_w(k_start, 0, col, 1)
        
        # Term 3: A[1, j0] * B[j0, k0] -> C[1, k0]
        col = base_rank + 2
        set_u(1, j_start, col, 1)
        set_v(j_start, k_start, col, 1)
        set_w(k_start, 1, col, 1)
        
        # Term 4: A[1, j1] * B[j1, k0] -> C[1, k0]
        col = base_rank + 3
        set_u(1, j_start + 1, col, 1)
        set_v(j_start + 1, k_start, col, 1)
        set_w(k_start, 1, col, 1)

    def add_rank14_217(j_fixed, base_rank):
        # Adds 14 rank-1 terms for A[i, j_fixed] * B[j_fixed, k]
        # i in {0, 1}, k in {0, ..., 6}
        col = base_rank
        for i in range(2):
            for k in range(7):
                set_u(i, j_fixed, col, 1)
                set_v(j_fixed, k, col, 1)
                set_w(k, i, col, 1)
                col += 1

    # Block 1: j in {0, 1}
    # k in {0, 1} -> Strassen
    add_strassen_222(0, 0, 0)
    # k in {2, 3} -> Strassen
    add_strassen_222(0, 2, 7)
    # k in {4, 5} -> Strassen
    add_strassen_222(0, 4, 14)
    # k in {6} -> Rank 4
    add_rank4_221(0, 6, 21)
    
    # Block 2: j = 2
    # Rank 14
    add_rank14_217(2, 25)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
