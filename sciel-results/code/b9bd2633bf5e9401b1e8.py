import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_laderman_3x3():
        """
        Laderman's algorithm for 3x3 matrix multiplication, rank 15.
        Returns U, V, W arrays of shape (9, 15) each.
        U[i*3+j, r] corresponds to A[i,j]
        V[j*3+k, r] corresponds to B[j,k]
        W[k*3+i, r] corresponds to C[i,k]
        """
        U = np.zeros((9, 15))
        V = np.zeros((9, 15))
        W = np.zeros((9, 15))
        
        # Laderman's 15 rank-1 terms
        # M1 = A[0,0] * B[0,0]
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M2 = A[0,1] * B[1,0]
        U[1, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
        # M3 = A[1,0] * B[0,1]
        U[3, 2] = 1; V[1, 2] = 1; W[4, 2] = 1
        # M4 = A[1,1] * B[1,1]
        U[4, 3] = 1; V[4, 3] = 1; W[7, 3] = 1
        # M5 = A[2,2] * B[2,2]
        U[8, 4] = 1; V[8, 4] = 1; W[8, 4] = 1
        # M6 = (A[0,0]+A[0,1]) * (B[1,0]-B[1,1])
        U[0, 5] = 1; U[1, 5] = 1
        V[3, 5] = 1; V[4, 5] = -1
        W[3, 5] = 1; W[4, 5] = -1
        # M7 = (A[1,1]+A[2,1]) * (B[1,1]-B[1,2])
        U[4, 6] = 1; U[7, 6] = 1
        V[4, 6] = 1; V[7, 6] = -1
        W[7, 6] = 1; W[8, 6] = -1
        # M8 = (A[2,2]+A[2,0]) * (B[0,2]-B[0,1])
        U[8, 7] = 1; U[6, 7] = 1
        V[2, 7] = 1; V[1, 7] = -1
        W[6, 7] = 1; W[7, 7] = -1
        # M9 = (A[0,1]+A[2,1]) * (B[1,0]+B[1,2])
        U[1, 8] = 1; U[7, 8] = 1
        V[3, 8] = 1; V[7, 8] = 1
        W[3, 8] = 1; W[8, 8] = 1
        # M10 = (A[1,2]+A[1,0]) * (B[0,1]+B[0,2])
        U[5, 9] = 1; U[3, 9] = 1
        V[1, 9] = 1; V[2, 9] = 1
        W[4, 9] = 1; W[5, 9] = 1
        # M11 = (A[2,0]+A[0,0]) * (B[0,2]+B[2,2])
        U[6, 10] = 1; U[0, 10] = 1
        V[2, 10] = 1; V[8, 10] = 1
        W[6, 10] = 1; W[8, 10] = 1
        # M12 = (A[0,2]+A[0,1]) * (B[1,1]+B[1,2])
        U[2, 11] = 1; U[1, 11] = 1
        V[4, 11] = 1; V[7, 11] = 1
        W[5, 11] = 1; W[8, 11] = 1
        # M13 = (A[1,0]+A[1,1]) * (B[0,1]+B[0,0])
        U[3, 12] = 1; U[4, 12] = 1
        V[1, 12] = 1; V[0, 12] = 1
        W[3, 12] = 1; W[4, 12] = 1
        # M14 = (A[2,1]+A[2,2]) * (B[1,2]+B[1,1])
        U[7, 13] = 1; U[8, 13] = 1
        V[7, 13] = 1; V[4, 13] = 1
        W[8, 13] = 1; W[7, 13] = 1
        # M15 = (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2])
        U[0, 14] = 1; U[4, 14] = 1; U[8, 14] = 1
        V[0, 14] = 1; V[4, 14] = 1; V[8, 14] = 1
        W[0, 14] = 1; W[4, 14] = 1; W[8, 14] = 1
        
        return U, V, W

    def build_strassen_2x2():
        """
        Strassen's algorithm for 2x2 matrix multiplication, rank 7.
        Returns U, V, W arrays of shape (4, 7) each.
        U[i*2+j, r] corresponds to A[i,j]
        V[j*2+k, r] corresponds to B[j,k]
        W[k*2+i, r] corresponds to C[i,k]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = -1; W[2, 5] = 1
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    def build_block_decomposition():
        """
        Build decomposition using Laderman for 3x3, Strassen for 2x2,
        and schoolbook for cross-terms.
        """
        # Total rank estimate: 15 + 7 + cross terms
        # Cross terms (schoolbook):
        # A12(3x2) x B21(2x3) -> 3x3: 18
        # A11(3x3) x B12(3x2) -> 3x2: 18
        # A21(2x3) x B11(3x3) -> 2x3: 18
        # A12(3x2) x B22(2x2) -> 3x2: 12
        # A21(2x3) x B12(3x2) -> 2x2: 12
        # A22(2x2) x B21(2x3) -> 2x3: 12
        # Total cross: 18+18+18+12+12+12 = 90
        # Total: 15 + 7 + 90 = 112
        
        total_rank = 112
        U = np.zeros((25, total_rank))
        V = np.zeros((25, total_rank))
        W = np.zeros((25, total_rank))
        
        col = 0
        
        # 1. Laderman for A11 x B11 -> C11 (indices 0-2 for rows/cols)
        U_l, V_l, W_l = build_laderman_3x3()
        for r in range(15):
            for i in range(3):
                for j in range(3):
                    U[i*5+j, col] += U_l[i*3+j, r]
            for j in range(3):
                for k in range(3):
                    V[j*5+k, col] += V_l[j*3+k, r]
            for k in range(3):
                for i in range(3):
                    W[k*5+i, col] += W_l[k*3+i, r]
            col += 1
        
        # 2. Strassen for A22 x B22 -> C22 (indices 3-4 for rows/cols)
        U_s, V_s, W_s = build_strassen_2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[(3+i)*5+(3+j), col] += U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[(3+j)*5+(3+k), col] += V_s[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(3+k)*5+(3+i), col] += W_s[k*2+i, r]
            col += 1
        
        # 3. Schoolbook for A12(3x2) x B21(2x3) -> C11(3x3)
        # A12: rows 0-2, cols 3-4
        # B21: rows 3-4, cols 0-2
        # C11: rows 0-2, cols 0-2
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U[i*5+(3+j), col] = 1
                    V[(3+j)*5+k, col] = 1
                    W[k*5+i, col] = 1
                    col += 1
        
        # 4. Schoolbook for A11(3x3) x B12(3x2) -> C12(3x2)
        # A11: rows 0-2, cols 0-2
        # B12: rows 0-2, cols 3-4
        # C12: rows 0-2, cols 3-4
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*5+j, col] = 1
                    V[j*5+(3+k), col] = 1
                    W[(3+k)*5+i, col] = 1
                    col += 1
        
        # 5. Schoolbook for A21(2x3) x B11(3x3) -> C21(2x3)
        # A21: rows 3-4, cols 0-2
        # B11: rows 0-2, cols 0-2
        # C21: rows 3-4, cols 0-2
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[(3+i)*5+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*5+(3+i), col] = 1
                    col += 1
        
        # 6. Schoolbook for A12(3x2) x B22(2x2) -> C12(3x2)
        # A12: rows 0-2, cols 3-4
        # B22: rows 3-4, cols 3-4
        # C12: rows 0-2, cols 3-4
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    U[i*5+(3+j), col] = 1
                    V[(3+j)*5+(3+k), col] = 1
                    W[(3+k)*5+i, col] = 1
                    col += 1
        
        # 7. Schoolbook for A21(2x3) x B12(3x2) -> C22(2x2)
        # A21: rows 3-4, cols 0-2
        # B12: rows 0-2, cols 3-4
        # C22: rows 3-4, cols 3-4
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[(3+i)*5+j, col] = 1
                    V[j*5+(3+k), col] = 1
                    W[(3+k)*5+(3+i), col] = 1
                    col += 1
        
        # 8. Schoolbook for A22(2x2) x B21(2x3) -> C21(2x3)
        # A22: rows 3-4, cols 3-4
        # B21: rows 3-4, cols 0-2
        # C21: rows 3-4, cols 0-2
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(3+i)*5+(3+j), col] = 1
                    V[(3+j)*5+k, col] = 1
                    W[k*5+(3+i), col] = 1
                    col += 1
        
        # Verify col == total_rank
        assert col == total_rank, f"Column count mismatch: {col} vs {total_rank}"
        
        return U, V, W

    # Build the block decomposition
    U_block, V_block, W_block = build_block_decomposition()
    rank_block = U_block.shape[1]
    
    # Verify the block decomposition
    max_err = verify(U_block, V_block, W_block, rank_block)
    if max_err > 1e-10:
        # Fall back to schoolbook if block decomposition is incorrect
        U_school = np.zeros((n*m, rank_schoolbook))
        V_school = np.zeros((m*p, rank_schoolbook))
        W_school = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U_block.tolist(),
        "V": V_block.tolist(),
        "W": W_block.tolist()
    }
