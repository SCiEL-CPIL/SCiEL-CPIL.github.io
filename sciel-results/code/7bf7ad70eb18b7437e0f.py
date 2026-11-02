"""Matrix multiplication 4x4x5 - rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    
    # Target rank: 72
    # Strategy: 
    # Split A into four 2x2 blocks.
    # Each 2x2 block of A multiplies the corresponding 2x5 block of B.
    # Split the 2x5 block of B into 2x2, 2x2, and 2x1 sub-blocks.
    # Use Strassen (rank 7) for 2x2 * 2x2.
    # Use Schoolbook (rank 4) for 2x2 * 2x1.
    # Rank per A block: 7 + 7 + 4 = 18.
    # Total rank: 4 * 18 = 72.
    
    target_rank = 72
    
    def build_strassen_2x2x2():
        """
        Returns U, V, W for 2x2 * 2x2 = 2x2 with rank 7.
        U shape: (4, 7)  indexed by A[i,j] -> i*2+j
        V shape: (4, 7)  indexed by B[i,j] -> i*2+j
        W shape: (4, 7)  indexed by C[i,j] -> i*2+j (assuming row-major for local block)
        Note: In the global context, W mapping depends on the convention.
        Here we return local factors, and map them in construct().
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
    
    def build_schoolbook_2x2x1():
        """
        Returns U, V, W for 2x2 * 2x1 = 2x1 with rank 4.
        U shape: (4, 4) indexed by A[i,j] -> i*2+j
        V shape: (2, 4) indexed by B[i,j] -> i (since j is fixed 0)
        W shape: (2, 4) indexed by C[i,j] -> i
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        
        # Term 1: a00 * b0 -> c0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: a01 * b1 -> c0
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Term 3: a10 * b0 -> c1
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Term 4: a11 * b1 -> c1
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W
    
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
    
    U_total = np.zeros((n * m, target_rank))
    V_total = np.zeros((m * p, target_rank))
    W_total = np.zeros((p * n, target_rank))
    
    col_offset = 0
    
    # Precompute sub-decompositions
    U_s, V_s, W_s = build_strassen_2x2x2()
    U_sb, V_sb, W_sb = build_schoolbook_2x2x1()
    
    # Iterate over 4 blocks of A
    for r0 in [0, 2]:
        for c0 in [0, 2]:
            # A block is at rows r0:r0+2, cols c0:c0+2
            
            # Corresponding B rows are r0:r0+2 (since A's col index corresponds to B's row index)
            # Wait, A[i, k] * B[k, j].
            # A block (r0, c0) uses A rows r0, r0+1 and cols c0, c0+1.
            # These cols c0, c0+1 correspond to B rows c0, c0+1.
            b_r0 = c0
            
            # B sub-blocks:
            # 1. B rows b_r0, b_r0+1; Cols 0, 1 (2x2)
            # 2. B rows b_r0, b_r0+1; Cols 2, 3 (2x2)
            # 3. B rows b_r0, b_r0+1; Col 4 (2x1)
            
            # Sub-block 1: 2x2
            l0, l1 = 0, 2
            for r in range(7):
                # Map U
                for i in range(2):
                    for j in range(2):
                        U_total[(r0 + i) * m + (c0 + j), col_offset + r] = U_s[i * 2 + j, r]
                # Map V
                for i in range(2):
                    for j in range(2):
                        V_total[(b_r0 + i) * p + (l0 + j), col_offset + r] = V_s[i * 2 + j, r]
                # Map W
                # W index is k * n + i, where k is col of C, i is row of C.
                # C rows are r0, r0+1. C cols are l0, l0+1.
                for i in range(2):
                    for j in range(2):
                        W_total[(l0 + j) * n + (r0 + i), col_offset + r] = W_s[i * 2 + j, r]
            col_offset += 7
            
            # Sub-block 2: 2x2
            l0, l1 = 2, 4
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_total[(r0 + i) * m + (c0 + j), col_offset + r] = U_s[i * 2 + j, r]
                for i in range(2):
                    for j in range(2):
                        V_total[(b_r0 + i) * p + (l0 + j), col_offset + r] = V_s[i * 2 + j, r]
                for i in range(2):
                    for j in range(2):
                        W_total[(l0 + j) * n + (r0 + i), col_offset + r] = W_s[i * 2 + j, r]
            col_offset += 7
            
            # Sub-block 3: 2x1
            l0 = 4
            for r in range(4):
                for i in range(2):
                    for j in range(2):
                        U_total[(r0 + i) * m + (c0 + j), col_offset + r] = U_sb[i * 2 + j, r]
                for i in range(2):
                    V_total[(b_r0 + i) * p + l0, col_offset + r] = V_sb[i, r]
                for i in range(2):
                    W_total[l0 * n + (r0 + i), col_offset + r] = W_sb[i, r]
            col_offset += 4
    
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
