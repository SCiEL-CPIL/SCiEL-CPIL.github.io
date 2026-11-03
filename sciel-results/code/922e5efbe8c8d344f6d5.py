"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy:
    1. Attempt to find a rank-32 decomposition using Strassen's algorithm structure.
       Strassen reduces 2x2x2 multiplication from rank 8 to rank 7.
       By applying this structure appropriately to the 2x4x8 tensor, we aim for
       a total rank of 32 (which is 2 * 2 * 8 / 2 * 2? No, simply 2 * 4 * 8 = 64 schoolbook.
       Strassen for 2x2 is rank 7. 
       2x4x8 can be seen as 2 blocks of 2x2x8? No.
       Let's view it as two independent 2x2x8 multiplications? No, the middle dimension is 4.
       
       Let's use the block structure:
       A is 2x4, B is 4x8.
       Split A into A1 (2x2), A2 (2x2).
       Split B into B1 (2x8), B2 (2x8).
       C = A1*B1 + A2*B2.
       
       Each term A_i (2x2) * B_i (2x8) can be computed.
       A 2x2 * 2x8 multiplication.
       We can split B_i into four 2x2 blocks: B_i1, B_i2, B_i3, B_i4.
       So C = A1*(B11+B12+B13+B14) + A2*(B21+B22+B23+B24).
       This is 8 independent 2x2*2x2 multiplications.
       If we use Strassen (rank 7) for each 2x2*2x2, the total rank would be 8 * 7 = 56.
       
       Can we do better?
       The tensor is T_{2,4,8}.
       Known bounds: R(2,4,8) <= 56 (by using Strassen on 2x2x2 blocks).
       Is there a rank 32 decomposition?
       The asymptotic rank is related to alpha. 
       For 2x2x2, rank is 7.
       For 2x4x8, if we view it as 2x2x2 tensor product structure?
       2x4x8 = (2x2x2) x (1x2x4)? No.
       
       Let's try to construct the rank 56 solution explicitly using Strassen.
       If we can get 56, that's a score of 64/56 = 1.14.
       
       Structure:
       A = [[A1, A2], [A3, A4]] (2x4)
       B = [[B1, B2, B3, B4], [B5, B6, B7, B8]] (4x8) where each Bj is 2x2.
       Actually, let's index A as 2x4 and B as 4x8.
       
       Let's use the decomposition:
       C = A * B
       Split A into two 2x2 blocks: A_L, A_R.
       Split B into two 2x8 blocks: B_T, B_B.
       C = A_L B_T + A_R B_B.
       
       Now, compute A_L B_T where A_L is 2x2 and B_T is 2x8.
       Split B_T into four 2x2 blocks: B_T1, B_T2, B_T3, B_T4.
       A_L B_T = A_L B_T1 + A_L B_T2 + A_L B_T3 + A_L B_T4.
       Each A_L B_Tj is a 2x2 * 2x2 multiplication.
       Using Strassen (rank 7) for each, we use 4 * 7 = 28 ranks for A_L B_T.
       Similarly, 28 ranks for A_R B_B.
       Total rank = 56.
       
       Let's implement this rank 56 decomposition.
       """
    
    n, m, p = 2, 4, 8
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = d(f-h)
    # M5 = (a+c)(h)
    # M6 = (b-c)(e+f)
    # M7 = (a+d)(f-g)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M5
    # C11 = M4 + M3 - M2 + M6
    
    def strassen_2x2x2(U, V, W, a_idx, b_idx, c_idx, d_idx, 
                       e_idx, f_idx, g_idx, h_idx, 
                       c00_idx, c01_idx, c10_idx, c11_idx, offset):
        # M1 = (a+d)(e+h)
        U[a_idx][offset] = 1; U[d_idx][offset] = 1
        V[e_idx][offset] = 1; V[h_idx][offset] = 1
        W[c00_idx][offset] = 1; W[c01_idx][offset] = 1
        offset += 1
        
        # M2 = (c+d)(e)
        U[c_idx][offset] = 1; U[d_idx][offset] = 1
        V[e_idx][offset] = 1
        W[c10_idx][offset] = 1; W[c11_idx][offset] = -1
        offset += 1
        
        # M3 = (a-b)(g+h)
        U[a_idx][offset] = 1; U[b_idx][offset] = -1
        V[g_idx][offset] = 1; V[h_idx][offset] = 1
        W[c01_idx][offset] = 1; W[c11_idx][offset] = 1
        offset += 1
        
        # M4 = d(f-h)
        U[d_idx][offset] = 1
        V[f_idx][offset] = 1; V[h_idx][offset] = -1
        W[c00_idx][offset] = 1; W[c11_idx][offset] = 1
        offset += 1
        
        # M5 = (a+c)(h)
        U[a_idx][offset] = 1; U[c_idx][offset] = 1
        V[h_idx][offset] = 1
        W[c00_idx][offset] = -1; W[c10_idx][offset] = 1
        offset += 1
        
        # M6 = (b-c)(e+f)
        U[b_idx][offset] = 1; U[c_idx][offset] = -1
        V[e_idx][offset] = 1; V[f_idx][offset] = 1
        W[c11_idx][offset] = 1
        offset += 1
        
        # M7 = (a+d)(f-g)
        U[a_idx][offset] = 1; U[d_idx][offset] = 1
        V[f_idx][offset] = 1; V[g_idx][offset] = -1
        W[c00_idx][offset] = 1
        offset += 1
        
        return offset

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    
    # Indices for A (2x4):
    # A = [[A00, A01, A02, A03],
    #      [A10, A11, A12, A13]]
    # A_L = [[A00, A01], [A10, A11]] -> indices 0,1,2,3
    # A_R = [[A02, A03], [A12, A13]] -> indices 4,5,6,7
    
    # Indices for B (4x8):
    # B = [[B00, B01, B02, B03, B04, B05, B06, B07],
    #      [B10, B11, B12, B13, B14, B15, B16, B17],
    #      [B20, B21, B22, B23, B24, B25, B26, B27],
    #      [B30, B31, B32, B33, B34, B35, B36, B37]]
    # B_T = first 2 rows of B (2x8)
    # B_B = last 2 rows of B (2x8)
    
    # B_T1 = B[0:2, 0:2] -> indices 0,1,8,9
    # B_T2 = B[0:2, 2:4] -> indices 2,3,10,11
    # B_T3 = B[0:2, 4:6] -> indices 4,5,12,13
    # B_T4 = B[0:2, 6:8] -> indices 6,7,14,15
    
    # B_B1 = B[2:4, 0:2] -> indices 16,17,24,25
    # B_B2 = B[2:4, 2:4] -> indices 18,19,26,27
    # B_B3 = B[2:4, 4:6] -> indices 20,21,28,29
    # B_B4 = B[2:4, 6:8] -> indices 22,23,30,31
    
    # Output C (2x8):
    # C = [[C00, C01, C02, C03, C04, C05, C06, C07],
    #      [C10, C11, C12, C13, C14, C15, C16, C17]]
    # W indices are k*n + i.
    # C00 -> k=0, i=0 -> 0
    # C01 -> k=1, i=0 -> 2
    # C02 -> k=2, i=0 -> 4
    # C03 -> k=3, i=0 -> 6
    # C04 -> k=4, i=0 -> 8
    # C05 -> k=5, i=0 -> 10
    # C06 -> k=6, i=0 -> 12
    # C07 -> k=7, i=0 -> 14
    # C10 -> k=0, i=1 -> 1
    # C11 -> k=1, i=1 -> 3
    # ...
    
    def get_w_idx(k, i):
        return k * n + i
    
    # Process A_L * B_T
    # A_L indices: 0,1,2,3
    # B_T1 indices: 0,1,8,9
    # Output C columns 0,1 -> W indices:
    # C00 (k=0,i=0)=0, C01 (k=1,i=0)=2, C10 (k=0,i=1)=1, C11 (k=1,i=1)=3
    
    offset = strassen_2x2x2(U, V, W, 
                             0, 1, 2, 3, 
                             0, 1, 8, 9, 
                             get_w_idx(0,0), get_w_idx(1,0), get_w_idx(0,1), get_w_idx(1,1), 
                             offset)
                             
    # B_T2 indices: 2,3,10,11
    # Output C columns 2,3 -> W indices:
    # C02 (k=2,i=0)=4, C03 (k=3,i=0)=6, C12 (k=2,i=1)=5, C13 (k=3,i=1)=7
    
    offset = strassen_2x2x2(U, V, W, 
                             0, 1, 2, 3, 
                             2, 3, 10, 11, 
                             get_w_idx(2,0), get_w_idx(3,0), get_w_idx(2,1), get_w_idx(3,1), 
                             offset)
                             
    # B_T3 indices: 4,5,12,13
    # Output C columns 4,5 -> W indices:
    # C04 (k=4,i=0)=8, C05 (k=5,i=0)=10, C14 (k=4,i=1)=9, C15 (k=5,i=1)=11
    
    offset = strassen_2x2x2(U, V, W, 
                             0, 1, 2, 3, 
                             4, 5, 12, 13, 
                             get_w_idx(4,0), get_w_idx(5,0), get_w_idx(4,1), get_w_idx(5,1), 
                             offset)
                             
    # B_T4 indices: 6,7,14,15
    # Output C columns 6,7 -> W indices:
    # C06 (k=6,i=0)=12, C07 (k=7,i=0)=14, C16 (k=6,i=1)=13, C17 (k=7,i=1)=15
    
    offset = strassen_2x2x2(U, V, W, 
                             0, 1, 2, 3, 
                             6, 7, 14, 15, 
                             get_w_idx(6,0), get_w_idx(7,0), get_w_idx(6,1), get_w_idx(7,1), 
                             offset)
                             
    # Process A_R * B_B
    # A_R indices: 4,5,6,7
    # B_B1 indices: 16,17,24,25
    # Output C columns 0,1 -> W indices: 0,2,1,3
    
    offset = strassen_2x2x2(U, V, W, 
                             4, 5, 6, 7, 
                             16, 17, 24, 25, 
                             get_w_idx(0,0), get_w_idx(1,0), get_w_idx(0,1), get_w_idx(1,1), 
                             offset)
                             
    # B_B2 indices: 18,19,26,27
    # Output C columns 2,3 -> W indices: 4,6,5,7
    
    offset = strassen_2x2x2(U, V, W, 
                             4, 5, 6, 7, 
                             18, 19, 26, 27, 
                             get_w_idx(2,0), get_w_idx(3,0), get_w_idx(2,1), get_w_idx(3,1), 
                             offset)
                             
    # B_B3 indices: 20,21,28,29
    # Output C columns 4,5 -> W indices: 8,10,9,11
    
    offset = strassen_2x2x2(U, V, W, 
                             4, 5, 6, 7, 
                             20, 21, 28, 29, 
                             get_w_idx(4,0), get_w_idx(5,0), get_w_idx(4,1), get_w_idx(5,1), 
                             offset)
                             
    # B_B4 indices: 22,23,30,31
    # Output C columns 6,7 -> W indices: 12,14,13,15
    
    offset = strassen_2x2x2(U, V, W, 
                             4, 5, 6, 7, 
                             22, 23, 30, 31, 
                             get_w_idx(6,0), get_w_idx(7,0), get_w_idx(6,1), get_w_idx(7,1), 
                             offset)
                             
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
