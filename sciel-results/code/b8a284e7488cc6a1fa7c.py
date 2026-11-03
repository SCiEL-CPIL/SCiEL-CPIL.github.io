import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition."""
        # Standard Strassen 2x2 multiplication
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # We are mapping 2x2 * 2x2 to rank 7.
        # The indices in U, V, W correspond to the 4 elements of A, B, C.
        # A indices: 0,1,2,3 (a,b,c,d)
        # B indices: 0,1,2,3 (e,f,g,h)
        # C indices: 0,1,2,3 (p,q,r,s)
        
        # U maps A elements to the 7 rank terms.
        # V maps B elements to the 7 rank terms.
        # W maps C elements to the 7 rank terms.
        
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        # Term 1: (a+d)(e+h) -> C00 += 1, C01 += 1, C10 += 1, C11 += -1
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[1,0] = 1; W_local[2,0] = 1; W_local[3,0] = -1
        
        # Term 2: (c+d)e -> C10 += 1, C11 += 1
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = 1
        
        # Term 3: a(f-h) -> C00 += 1, C01 += 1, C10 += -1, C11 += -1
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[0,2] = 1; W_local[1,2] = 1; W_local[2,2] = -1; W_local[3,2] = -1
        
        # Term 4: d(g-e) -> C00 += -1, C01 += 1, C10 += -1, C11 += 1
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = -1; W_local[1,3] = 1; W_local[2,3] = -1; W_local[3,3] = 1
        
        # Term 5: (a+b)h -> C00 += 1, C01 += 1
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = 1; W_local[1,4] = 1
        
        # Term 6: (c-a)(e+f) -> C10 += 1, C11 += 1
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[2,5] = 1; W_local[3,5] = 1
        
        # Term 7: (b-d)(g+h) -> C01 += 1, C11 += 1
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[1,6] = 1; W_local[3,6] = 1
        
        total_rank = 56
        U = np.zeros((n*m, total_rank), dtype=int) # 8
        V = np.zeros((m*p, total_rank), dtype=int) # 32
        W = np.zeros((p*n, total_rank), dtype=int) # 16
        
        # The problem is 2x4 * 4x8.
        # We can view A as 2 blocks of 2x2? No, A is 2x4.
        # Let's stick to the partitioning that yielded 56.
        # The previous code partitioned into 8 sub-problems of 2x2 * 2x2.
        # This implies A is split into 4 blocks of 2x2? No, A is 2x4.
        # If we split A into two 2x2 matrices horizontally: A = [A1 A2]
        # B is 4x8. Split B into two 2x8 matrices vertically? No.
        
        # Let's re-verify the partitioning logic from the previous code.
        # "Partitioning the 2x4*4x8 multiplication into 8 independent 2x2*2x2 sub-problems"
        # This suggests:
        # A is 2x4. Split rows? No, 2 rows. Split cols into 2 groups of 2?
        # So A = [A_L A_R] where A_L, A_R are 2x2.
        # B is 4x8. Split rows into 2 groups of 2?
        # So B = [B_T; B_B] where B_T, B_B are 2x8.
        # Then C = A_L B_T + A_R B_B.
        # This is a sum of two products of 2x2 and 2x8.
        # A 2x2 * 2x8 product can be seen as 4 independent 2x2 * 2x2 products?
        # Let B_T = [B_T1 B_T2 B_T3 B_T4] where each is 2x2.
        # Then A_L B_T = [A_L B_T1, A_L B_T2, A_L B_T3, A_L B_T4].
        # Each term is 2x2 * 2x2.
        # So we have 4 terms for A_L B_T and 4 terms for A_R B_B.
        # Total 8 terms.
        # Each term is a 2x2 * 2x2 multiplication.
        # Using Strassen (rank 7) for each gives 8 * 7 = 56.
        
        # Let's implement this mapping.
        
        # A indices: 0..3 (row 0: 0,1; row 1: 2,3) -> Wait, A is 2x4.
        # A[i, j]. i in 0..1, j in 0..3.
        # A_L = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # A_R = A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
        
        # B indices: 0..31 (row j, col k). j in 0..3, k in 0..7.
        # B_T = B[0:2, :] = rows 0,1.
        # B_B = B[2:4, :] = rows 2,3.
        
        # For A_L B_T:
        # A_L elements: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 in A_L local.
        # B_T elements: B[0, k], B[1, k] for k=0..7.
        # We split B_T into 4 blocks of 2x2:
        # B_T1 = B[0:2, 0:2]
        # B_T2 = B[0:2, 2:4]
        # B_T3 = B[0:2, 4:6]
        # B_T4 = B[0:2, 6:8]
        
        # For each block B_Tx, we perform 2x2 * 2x2.
        # The result is added to C.
        # C = A B. C[i, k].
        # C[0, 0:2] += A_L B_T1
        # C[1, 0:2] += A_L B_T1
        # ...
        
        # Let's define the local indices for Strassen.
        # A_local: 0=a, 1=b, 2=c, 3=d
        # B_local: 0=e, 1=f, 2=g, 3=h
        # C_local: 0=p, 1=q, 2=r, 3=s
        
        # Mapping for A_L (Left half of A):
        # A_L[0,0] -> A_global[0,0] -> index 0 in A_L_local
        # A_L[0,1] -> A_global[0,1] -> index 1 in A_L_local
        # A_L[1,0] -> A_global[1,0] -> index 2 in A_L_local
        # A_L[1,1] -> A_global[1,1] -> index 3 in A_L_local
        
        # Mapping for A_R (Right half of A):
        # A_R[0,0] -> A_global[0,2] -> index 0 in A_R_local
        # A_R[0,1] -> A_global[0,3] -> index 1 in A_R_local
        # A_R[1,0] -> A_global[1,2] -> index 2 in A_R_local
        # A_R[1,1] -> A_global[1,3] -> index 3 in A_R_local
        
        # Mapping for B_T1 (Top-Left of B_T):
        # B_T1[0,0] -> B_global[0,0]
        # B_T1[0,1] -> B_global[0,1]
        # B_T1[1,0] -> B_global[1,0]
        # B_T1[1,1] -> B_global[1,1]
        
        # General mapping for B block (row_offset, col_offset):
        # B_block[0,0] -> B_global[row_offset, col_offset]
        # B_block[0,1] -> B_global[row_offset, col_offset+1]
        # B_block[1,0] -> B_global[row_offset+1, col_offset]
        # B_block[1,1] -> B_global[row_offset+1, col_offset+1]
        
        # Mapping for C block (row_offset, col_offset):
        # C_block[0,0] -> C_global[row_offset, col_offset]
        # C_block[0,1] -> C_global[row_offset, col_offset+1]
        # C_block[1,0] -> C_global[row_offset+1, col_offset]
        # C_block[1,1] -> C_global[row_offset+1, col_offset+1]
        
        # We have 8 Strassen instances.
        # 4 for A_L * B_T (B_T1, B_T2, B_T3, B_T4)
        # 4 for A_R * B_B (B_B1, B_B2, B_B3, B_B4)
        
        # Instance 0: A_L * B_T1
        # A_L local index -> A_global index
        # B_T1 local index -> B_global index
        # C_local index -> C_global index (offset 0,0)
        
        # Let's create a helper to add a Strassen instance.
        def add_strassen_instance(a_map, b_map, c_map, base_rank):
            for l in range(7):
                r = base_rank + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val

        # A_L map: [0, 1, 2, 3] -> A_global [0, 1, 2, 3]
        # A_R map: [0, 1, 2, 3] -> A_global [4, 5, 6, 7] (since A is 2x4, indices are i*4+j)
        # A_global index = i*4 + j.
        # A_L: i=0,1; j=0,1. Indices: 0,1,2,3.
        # A_R: i=0,1; j=2,3. Indices: 2,3,6,7? No.
        # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
        # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
        # So A_L indices: 0,1,4,5?
        # Wait, A is stored row-major?
        # "U[i*m+j, r]"
        # A[i,j] is at index i*m + j.
        # m=4.
        # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
        # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
        
        # A_L (cols 0,1):
        # A_L[0,0] -> A[0,0] -> 0
        # A_L[0,1] -> A[0,1] -> 1
        # A_L[1,0] -> A[1,0] -> 4
        # A_L[1,1] -> A[1,1] -> 5
        # So a_map_L = [0, 1, 4, 5]
        
        # A_R (cols 2,3):
        # A_R[0,0] -> A[0,2] -> 2
        # A_R[0,1] -> A[0,3] -> 3
        # A_R[1,0] -> A[1,2] -> 6
        # A_R[1,1] -> A[1,3] -> 7
        # So a_map_R = [2, 3, 6, 7]
        
        # B is 4x8. Index j*p + k. p=8.
        # B[j,k] = j*8 + k.
        
        # B_T (rows 0,1):
        # B_T1 (cols 0,1):
        # B_T1[0,0] -> B[0,0] -> 0
        # B_T1[0,1] -> B[0,1] -> 1
        # B_T1[1,0] -> B[1,0] -> 8
        # B_T1[1,1] -> B[1,1] -> 9
        # b_map_T1 = [0, 1, 8, 9]
        
        # B_T2 (cols 2,3):
        # B_T2[0,0] -> B[0,2] -> 2
        # B_T2[0,1] -> B[0,3] -> 3
        # B_T2[1,0] -> B[1,2] -> 10
        # B_T2[1,1] -> B[1,3] -> 11
        # b_map_T2 = [2, 3, 10, 11]
        
        # B_T3 (cols 4,5):
        # b_map_T3 = [4, 5, 12, 13]
        
        # B_T4 (cols 6,7):
        # b_map_T4 = [6, 7, 14, 15]
        
        # B_B (rows 2,3):
        # B_B1 (cols 0,1):
        # B_B1[0,0] -> B[2,0] -> 16
        # B_B1[0,1] -> B[2,1] -> 17
        # B_B1[1,0] -> B[3,0] -> 24
        # B_B1[1,1] -> B[3,1] -> 25
        # b_map_B1 = [16, 17, 24, 25]
        
        # B_B2 (cols 2,3):
        # b_map_B2 = [18, 19, 26, 27]
        
        # B_B3 (cols 4,5):
        # b_map_B3 = [20, 21, 28, 29]
        
        # B_B4 (cols 6,7):
        # b_map_B4 = [22, 23, 30, 31]
        
        # C is 8x2. Index k*n + i. n=2.
        # C[k,i] = k*2 + i.
        
        # C_T1 (rows 0,1 of C, which correspond to k=0,1? No, C is 8x2. k is row, i is col.)
        # C[k,i]. k in 0..7, i in 0..1.
        # The product A_L B_T1 contributes to C[0:2, :].
        # So k=0,1.
        # C_T1[0,0] -> C[0,0] -> 0
        # C_T1[0,1] -> C[0,1] -> 1
        # C_T1[1,0] -> C[1,0] -> 2
        # C_T1[1,1] -> C[1,1] -> 3
        # c_map_T1 = [0, 1, 2, 3]
        
        # C_T2 (k=2,3):
        # c_map_T2 = [4, 5, 6, 7]
        
        # C_T3 (k=4,5):
        # c_map_T3 = [8, 9, 10, 11]
        
        # C_T4 (k=6,7):
        # c_map_T4 = [12, 13, 14, 15]
        
        # Wait, A_L B_T1 affects C[0:2, :].
        # A_L B_T2 affects C[2:4, :].
        # A_L B_T3 affects C[4:6, :].
        # A_L B_T4 affects C[6:8, :].
        
        # A_R B_B1 affects C[0:2, :].
        # A_R B_B2 affects C[2:4, :].
        # A_R B_B3 affects C[4:6, :].
        # A_R B_B4 affects C[6:8, :].
        
        # So C maps are the same for T and B instances, just different base ranks.
        
        a_map_L = [0, 1, 4, 5]
        a_map_R = [2, 3, 6, 7]
        
        b_maps = [
            [0, 1, 8, 9],   # T1
            [2, 3, 10, 11], # T2
            [4, 5, 12, 13], # T3
            [6, 7, 14, 15], # T4
            [16, 17, 24, 25], # B1
            [18, 19, 26, 27], # B2
            [20, 21, 28, 29], # B3
            [22, 23, 30, 31]  # B4
        ]
        
        c_maps = [
            [0, 1, 2, 3],   # k=0,1
            [4, 5, 6, 7],   # k=2,3
            [8, 9, 10, 11], # k=4,5
            [12, 13, 14, 15] # k=6,7
        ]
        
        # Instance 0: A_L * B_T1 (Base rank 0)
        add_strassen_instance(a_map_L, b_maps[0], c_maps[0], 0)
        # Instance 1: A_L * B_T2 (Base rank 7)
        add_strassen_instance(a_map_L, b_maps[1], c_maps[1], 7)
        # Instance 2: A_L * B_T3 (Base rank 14)
        add_strassen_instance(a_map_L, b_maps[2], c_maps[2], 14)
        # Instance 3: A_L * B_T4 (Base rank 21)
        add_strassen_instance(a_map_L, b_maps[3], c_maps[3], 21)
        
        # Instance 4: A_R * B_B1 (Base rank 28)
        add_strassen_instance(a_map_R, b_maps[4], c_maps[0], 28)
        # Instance 5: A_R * B_B2 (Base rank 35)
        add_strassen_instance(a_map_R, b_maps[5], c_maps[1], 35)
        # Instance 6: A_R * B_B3 (Base rank 42)
        add_strassen_instance(a_map_R, b_maps[6], c_maps[2], 42)
        # Instance 7: A_R * B_B4 (Base rank 49)
        add_strassen_instance(a_map_R, b_maps[7], c_maps[3], 49)
        
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify tensor equation."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True

    U56, V56, W56 = build_strassen_56()
    
    if verify_decomposition(U56, V56, W56):
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    # Fallback to rank 64 if something went wrong (should not happen)
    U = np.zeros((8, 64), dtype=int)
    V = np.zeros((32, 64), dtype=int)
    W = np.zeros((16, 64), dtype=int)
    for i in range(2):
        for j in range(4):
            for k in range(8):
                r = i*32 + j*8 + k
                U[i*4+j, r] = 1
                V[j*8+k, r] = 1
                W[k*2+i, r] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
