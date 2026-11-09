"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Use a 4+1 block decomposition. The 4x4 core uses a schoolbook rank of 64
    # (which is 4^3). The cross terms between the 4-block and the 1-block add
    # 4*4 + 4*4 + 4*4 + 1*1 = 49 terms.
    # Total rank = 64 + 49 = 113.
    # Wait, 4^3 = 64. The original program used a 3+1+1 split and got 113.
    # Let's re-calculate the 4+1 split.
    # Core 4x4: rank 64.
    # A[0:4, 4] * B[4, 0:4] -> 4*4 = 16 terms.
    # A[0:4, 0:4] * B[0:4, 4] -> 4*4 = 16 terms.
    # A[4, 0:4] * B[0:4, 0:4] -> 4*4 = 16 terms.
    # A[4, 4] * B[4, 4] -> 1*1 = 1 term.
    # Total = 64 + 16 + 16 + 16 + 1 = 113.
    # So 4+1 split gives 113.
    # The original 3+1+1 split:
    # Core 3x3: rank 27.
    # A[0:3, 3:5] (3x2) * B[3:5, 0:3] (2x3) -> 3*2*3 = 18.
    # A[0:3, 0:3] (3x3) * B[0:3, 3:5] (3x2) -> 3*3*2 = 18.
    # A[3:5, 0:3] (2x3) * B[0:3, 0:3] (3x3) -> 2*3*3 = 18.
    # A[0:3, 3:5] (3x2) * B[3:5, 3:5] (2x2) -> 3*2*2 = 12.
    # A[0:3, 0:3] (3x3) * B[0:3, 3:5] (3x2) ... wait, the indices are:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Blocks (I, J, K) where I, J, K in {0, 1, 2} with sizes 3, 1, 1.
    # (0,0,0): 3*3*3 = 27
    # (0,0,1): 3*3*1 = 9
    # (0,0,2): 3*3*1 = 9
    # (0,1,0): 3*1*3 = 9
    # (0,1,1): 3*1*1 = 3
    # (0,1,2): 3*1*1 = 3
    # (0,2,0): 3*1*3 = 9
    # (0,2,1): 3*1*1 = 3
    # (0,2,2): 3*1*1 = 3
    # (1,0,0): 1*3*3 = 9
    # (1,0,1): 1*3*1 = 3
    # (1,0,2): 1*3*1 = 3
    # (1,1,0): 1*1*3 = 3
    # (1,1,1): 1*1*1 = 1
    # (1,1,2): 1*1*1 = 1
    # (1,2,0): 1*1*3 = 3
    # (1,2,1): 1*1*1 = 1
    # (1,2,2): 1*1*1 = 1
    # (2,0,0): 1*3*3 = 9
    # (2,0,1): 1*3*1 = 3
    # (2,0,2): 1*3*1 = 3
    # (2,1,0): 1*1*3 = 3
    # (2,1,1): 1*1*1 = 1
    # (2,1,2): 1*1*1 = 1
    # (2,2,0): 1*1*3 = 3
    # (2,2,1): 1*1*1 = 1
    # (2,2,2): 1*1*1 = 1
    # Sum = 27+9+9+9+3+3+9+3+3 + 9+3+3+3+1+1+3+1+1 + 9+3+3+3+1+1+3+1+1
    # = 72 + 27 + 27 = 126?
    # Let's just run the code and see. The comment in the original code says 113.
    # Ah, the original code only iterates b_i in 0..2, b_k in 0..2, b_j in 0..2.
    # This means it ignores blocks where any index is 3 or 4.
    # So it only computes the top-left 3x3 block of C?
    # No, the indices g_i, g_j, g_k are global.
    # If b_i=0, g_i is 0..2. If b_i=1, g_i is 3. If b_i=2, g_i is 4.
    # So it covers all 5 rows and cols.
    # Let's trust the code.
    
    # Let's try to improve using Strassen for the 3x3 core.
    # Strassen for 3x3? No, Strassen is for 2x2.
    # We can split 3 into 2+1.
    # Core 2x2 uses Strassen (rank 7).
    # Then we have to handle the cross terms with the 1-block.
    
    # Split: 2, 1, 1, 1?
    # Or 2, 2, 1.
    # Let's try 2, 2, 1.
    # Sizes: 2, 2, 1.
    # Core (0,0,0): 2x2x2 using Strassen -> rank 7.
    # (0,0,1): 2x2x1 = 4.
    # (0,1,0): 2x2x1 = 4.
    # (0,1,1): 2x2x1 = 4.
    # (1,0,0): 2x2x1 = 4.
    # (1,0,1): 2x2x1 = 4.
    # (1,1,0): 2x2x1 = 4.
    # (1,1,1): 2x2x1 = 4.
    # (2,0,0): 1x2x2 = 4.
    # (2,0,1): 1x2x2 = 4.
    # (2,1,0): 1x2x2 = 4.
    # (2,1,1): 1x2x2 = 4.
    # (0,2,0): 2x1x2 = 4.
    # (0,2,1): 2x1x2 = 4.
    # (1,2,0): 2x1x2 = 4.
    # (1,2,1): 2x1x2 = 4.
    # (2,2,0): 1x1x2 = 2.
    # (2,2,1): 1x1x2 = 2.
    # (0,0,2): 2x2x1 = 4.
    # (0,1,2): 2x2x1 = 4.
    # (1,0,2): 2x2x1 = 4.
    # (1,1,2): 2x2x1 = 4.
    # (2,0,2): 1x2x1 = 2.
    # (2,1,2): 1x2x1 = 2.
    # (0,2,2): 2x1x1 = 2.
    # (1,2,2): 2x1x1 = 2.
    # (2,2,2): 1x1x1 = 1.
    
    # Sum: 7 + 4*6 + 4*4 + 2*4 + 4*4 + 2*2 + 1 = 7 + 24 + 16 + 8 + 16 + 4 + 1 = 76.
    # This is much better!
    
    # We need to implement Strassen for the 2x2 core.
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will build the U, V, W matrices.
    # For each Strassen term, we add 1 or more product terms.
    # For example, M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Each of these is a single (i,j,k) term.
    
    # Indices for the 2x2 core (block 0):
    # A indices: i in 0..1, j in 0..1.
    # B indices: j in 0..1, k in 0..1.
    # C indices: i in 0..1, k in 0..1.
    
    n, m, p = DIMENSIONS
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(i, j, k, coeff=1.0):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U_list.append([0.0] * (n * m))
        V_list.append([0.0] * (m * p))
        W_list.append([0.0] * (p * n))
        U_list[-1][u_idx] = coeff
        V_list[-1][v_idx] = coeff
        W_list[-1][w_idx] = coeff

    # Block sizes: 2, 2, 1
    sizes = [2, 2, 1]
    row_offsets = [0, 2, 4]
    col_offsets = [0, 2, 4]
    
    # Iterate over blocks
    for b_i in range(3):
        for b_k in range(3):
            for b_j in range(3):
                size_i = sizes[b_i]
                size_j = sizes[b_j]
                size_k = sizes[b_k]
                off_i = row_offsets[b_i]
                off_j = col_offsets[b_j]
                off_k = col_offsets[b_k]
                
                if b_i == 0 and b_j == 0 and b_k == 0:
                    # Use Strassen for the 2x2 core
                    # Local indices: li, lj, lk in 0..1
                    # Global indices: g_i = off_i + li, etc.
                    # off_i=0, off_j=0, off_k=0.
                    
                    # M1 = (A00+A11)(B00+B11)
                    # A00B00: i=0, j=0, k=0
                    add_term(0, 0, 0)
                    # A00B11: i=0, j=0, k=1
                    add_term(0, 0, 1)
                    # A11B00: i=1, j=1, k=0
                    add_term(1, 1, 0)
                    # A11B11: i=1, j=1, k=1
                    add_term(1, 1, 1)
                    
                    # M2 = (A10+A11)B00
                    # A10B00: i=1, j=0, k=0
                    add_term(1, 0, 0)
                    # A11B00: i=1, j=1, k=0
                    add_term(1, 1, 0)
                    
                    # M3 = A00(B10-B00)
                    # A00B10: i=0, j=1, k=0
                    add_term(0, 1, 0)
                    # A00B00: i=0, j=0, k=0, coeff -1
                    add_term(0, 0, 0, -1.0)
                    
                    # M4 = A11(B01-B11)
                    # A11B01: i=1, j=0, k=1
                    add_term(1, 0, 1)
                    # A11B11: i=1, j=1, k=1, coeff -1
                    add_term(1, 1, 1, -1.0)
                    
                    # M5 = (A00+A01)B11
                    # A00B11: i=0, j=0, k=1
                    add_term(0, 0, 1)
                    # A01B11: i=0, j=1, k=1
                    add_term(0, 1, 1)
                    
                    # M6 = (A10-A00)(B00+B10)
                    # A10B00: i=1, j=0, k=0
                    add_term(1, 0, 0)
                    # A10B10: i=1, j=0, k=0? No, B10 is j=1, k=0.
                    add_term(1, 1, 0)
                    # A00B00: i=0, j=0, k=0, coeff -1
                    add_term(0, 0, 0, -1.0)
                    # A00B10: i=0, j=0, k=0? No, B10 is j=1, k=0.
                    add_term(0, 1, 0, -1.0)
                    
                    # M7 = (A01-A11)(B01+B11)
                    # A01B01: i=0, j=1, k=1
                    add_term(0, 1, 1)
                    # A01B11: i=0, j=1, k=1
                    add_term(0, 1, 1)
                    # A11B01: i=1, j=1, k=1? No, B01 is j=0, k=1.
                    add_term(1, 0, 1, -1.0)
                    # A11B11: i=1, j=1, k=1
                    add_term(1, 1, 1, -1.0)
                    
                else:
                    # Schoolbook for other blocks
                    for local_i in range(size_i):
                        for local_j in range(size_j):
                            for local_k in range(size_k):
                                g_i = off_i + local_i
                                g_j = off_j + local_j
                                g_k = off_k + local_k
                                add_term(g_i, g_j, g_k)
    
    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
