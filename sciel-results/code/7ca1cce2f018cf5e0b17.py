"""Matrix multiplication 4x4x5 - rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    def build_strassen_2x2x2():
        """
        Returns U, V, W for 2x2 * 2x2 = 2x2 with rank 7.
        U shape: (4, 7)  indexed by A[i,j] -> i*2+j
        V shape: (4, 7)  indexed by B[i,j] -> i*2+j
        W shape: (4, 7)  indexed by C[i,j] -> i*2+j
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a00 + a11)(b00 + b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (a10 + a11)b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3 = a00(b01 - b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4 = a11(b10 - b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # m5 = (a00 + a01)b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (a10 - a00)(b00 + b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (a01 - a11)(b10 + b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    def build_hybrid_2x2x3():
        """
        Returns U, V, W for 2x2 * 2x3 = 2x3 with rank 11.
        Uses Strassen (rank 7) for first 2 cols, Schoolbook (rank 4) for 3rd col.
        U shape: (4, 11)
        V shape: (6, 11)
        W shape: (6, 11)
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        
        U_total = np.zeros((4, 11))
        V_total = np.zeros((6, 11))
        W_total = np.zeros((6, 11))
        
        # Copy Strassen part (cols 0, 1 of B)
        U_total[:, 0:7] = U_s
        V_total[:, 0:7] = V_s
        W_total[:, 0:7] = W_s
        
        # Schoolbook part for col 2 of B (index 2 in 0..2, so global col 2)
        # B indices: (0,2) is index 2, (1,2) is index 5 in V_total (i*3+j)
        # C indices: (0,2) is index 2, (1,2) is index 5 in W_total (i*3+j)
        
        # c02 = a00*b02 + a01*b12
        # Term 1: a00 * b02 -> c02
        U_total[0, 7] = 1
        V_total[2, 7] = 1  # B[0,2]
        W_total[2, 7] = 1  # C[0,2]
        
        # Term 2: a01 * b12 -> c02
        U_total[1, 8] = 1
        V_total[5, 8] = 1  # B[1,2]
        W_total[2, 8] = 1  # C[0,2]
        
        # c12 = a10*b02 + a11*b12
        # Term 3: a10 * b02 -> c12
        U_total[2, 9] = 1
        V_total[2, 9] = 1  # B[0,2]
        W_total[5, 9] = 1  # C[1,2]
        
        # Term 4: a11 * b12 -> c12
        U_total[3, 10] = 1
        V_total[5, 10] = 1  # B[1,2]
        W_total[5, 10] = 1  # C[1,2]
        
        return U_total, V_total, W_total

    def verify_decomposition(U, V, W, rank, tol=1e-4):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def to_nice_list(mat):
        """Convert numpy array to list of integers/half-integers."""
        result = []
        for row in mat:
            nice_row = []
            for val in row:
                rounded = round(val * 2) / 2.0
                nice_row.append(rounded)
            result.append(nice_row)
        return result

    # Target rank: 72
    # Strategy:
    # Split A into four 2x2 blocks.
    # Each 2x2 block of A multiplies the corresponding 2x5 block of B.
    # Split the 2x5 block of B into 2x3 and 2x2 sub-blocks.
    # Use Hybrid (rank 11) for the 2x3 part.
    # Use Strassen (rank 7) for the 2x2 part.
    # Rank per A block: 11 + 7 = 18.
    # Total rank: 4 * 18 = 72.
    
    target_rank = 72
    
    U_total = np.zeros((n * m, target_rank))
    V_total = np.zeros((m * p, target_rank))
    W_total = np.zeros((p * n, target_rank))
    
    col_offset = 0
    
    # Precompute sub-decompositions
    U_s22, V_s22, W_s22 = build_strassen_2x2x2()
    U_s23, V_s23, W_s23 = build_hybrid_2x2x3()
    
    # Iterate over 4 blocks of A
    for r0 in [0, 2]:
        for c0 in [0, 2]:
            # A block is at rows r0:r0+2, cols c0:c0+2
            b_r0 = c0
            
            # Sub-block 1: B rows b_r0, b_r0+1; Cols 0, 1, 2 (2x3)
            l0 = 0
            for r in range(11):
                for i in range(2):
                    for j in range(2):
                        U_total[(r0 + i) * m + (c0 + j), col_offset + r] = U_s23[i * 2 + j, r]
                for i in range(2):
                    for j in range(3):
                        V_total[(b_r0 + i) * p + (l0 + j), col_offset + r] = V_s23[i * 3 + j, r]
                for i in range(2):
                    for j in range(3):
                        W_total[(l0 + j) * n + (r0 + i), col_offset + r] = W_s23[i * 3 + j, r]
            col_offset += 11
            
            # Sub-block 2: B rows b_r0, b_r0+1; Cols 3, 4 (2x2)
            l0 = 3
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_total[(r0 + i) * m + (c0 + j), col_offset + r] = U_s22[i * 2 + j, r]
                for i in range(2):
                    for j in range(2):
                        V_total[(b_r0 + i) * p + (l0 + j), col_offset + r] = V_s22[i * 2 + j, r]
                for i in range(2):
                    for j in range(2):
                        W_total[(l0 + j) * n + (r0 + i), col_offset + r] = W_s22[i * 2 + j, r]
            col_offset += 7
    
    if verify_decomposition(U_total, V_total, W_total, target_rank):
        U_list = to_nice_list(U_total)
        V_list = to_nice_list(V_total)
        W_list = to_nice_list(W_total)
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if verification fails
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    U_list = to_nice_list(U_sb)
    V_list = to_nice_list(V_sb)
    W_list = to_nice_list(W_sb)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
