import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
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

    # Construct a block-based decomposition
    # Split indices into I1={0,1}, I2={2,3,4}
    # J1={0,1}, J2={2,3,4}
    # K1={0,1}, K2={2,3,4}
    
    # We build U, V, W by concatenating the factors from each block multiplication.
    # The blocks are:
    # C11 = A11 B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    # C12 = A11 B12 (2x2 * 2x3) -> Rank 12 (Schoolbook)
    # C12 += A12 B22 (2x3 * 3x3) -> Rank 18 (Schoolbook)
    # C21 = A21 B11 (3x2 * 2x2) -> Rank 12 (Schoolbook)
    # C21 += A22 B21 (3x3 * 3x2) -> Rank 18 (Schoolbook)
    # C22 = A21 B12 (3x2 * 2x3) -> Rank 18 (Schoolbook)
    # C22 += A22 B22 (3x3 * 3x3) -> Rank 24 (Strassen 1+2 split)
    
    # Total Rank = 7 + 12 + 18 + 12 + 18 + 18 + 24 = 109
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(rank_block, U_block, V_block, W_block):
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)
    
    # Helper to create schoolbook block
    def schoolbook_block(n_r, m_c, p_c):
        # n_r: rows of A, m_c: cols of A (rows of B), p_c: cols of B
        rank = n_r * m_c * p_c
        U_b = np.zeros((n_r * m_c, rank))
        V_b = np.zeros((m_c * p_c, rank))
        W_b = np.zeros((p_c * n_r, rank))
        col = 0
        for i in range(n_r):
            for j in range(m_c):
                for k in range(p_c):
                    U_b[i*m_c+j, col] = 1
                    V_b[j*p_c+k, col] = 1
                    W_b[k*n_r+i, col] = 1
                    col += 1
        return U_b, V_b, W_b

    # Helper to create Strassen block (2x2)
    def strassen_block():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices for 2x2: i,j,k in {0,1}
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Strassen 2x2:
        # C11 = M1 - M5 + M4 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # W index: k*2+i. C11:(0,0)->0, C12:(1,0)->2, C21:(0,1)->1, C22:(1,1)->3

        # M1 = (A11+A22)(B11+B22): coeff +1 in C11, +1 in C22
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1

        # M2 = (A21+A22)B11: coeff +1 in C21, -1 in C22
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1

        # M3 = A11(B12-B22): coeff +1 in C12, +1 in C22
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[2, 2] = 1; W_s[3, 2] = 1

        # M4 = A22(B21-B11): coeff +1 in C11, +1 in C21
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1

        # M5 = (A11+A12)B22: coeff -1 in C11, +1 in C12
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[2, 4] = 1

        # M6 = (A21-A11)(B11+B12): coeff +1 in C22
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1

        # M7 = (A12-A22)(B21+B22): coeff +1 in C11
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s

    # Helper to create 3x3 Strassen-based block using 1+2 split
    def strassen_3x3_block():
        # Returns U, V, W of shape (9, 24), (9, 24), (9, 24)
        # Indices for 3x3: i,j,k in {0,1,2}
        # U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
        # Split: I1={0}, I2={1,2}, J1={0}, J2={1,2}, K1={0}, K2={1,2}
        
        U_s = np.zeros((9, 24))
        V_s = np.zeros((9, 24))
        W_s = np.zeros((9, 24))
        
        col = 0
        
        # C11 = A11 * B11 (1x1 * 1x1) -> rank 1
        # C[0,0] = A[0,0]*B[0,0]
        U_s[0, col] = 1  # U[i*3+j] = U[0*3+0] = U[0]
        V_s[0, col] = 1  # V[j*3+k] = V[0*3+0] = V[0]
        W_s[0, col] = 1  # W[k*3+i] = W[0*3+0] = W[0]
        col += 1
        
        # C12 part 1 = A11 * B12 (1x1 * 1x2) -> rank 2
        # C[0,1] += A[0,0]*B[0,1]
        U_s[0, col] = 1
        V_s[1, col] = 1  # V[0*3+1] = V[1]
        W_s[3, col] = 1  # W[1*3+0] = W[3]
        col += 1
        # C[0,2] += A[0,0]*B[0,2]
        U_s[0, col] = 1
        V_s[2, col] = 1  # V[0*3+2] = V[2]
        W_s[6, col] = 1  # W[2*3+0] = W[6]
        col += 1
        
        # C12 part 2 = A12 * B22 (1x2 * 2x2) -> rank 4
        # C[0,1] += A[0,1]*B[1,1] + A[0,2]*B[2,1]
        U_s[1, col] = 1   # U[0*3+1] = U[1]
        V_s[4, col] = 1   # V[1*3+1] = V[4]
        W_s[3, col] = 1   # W[1*3+0] = W[3]
        col += 1
        U_s[1, col] = 1   # U[0*3+1] = U[1]
        V_s[5, col] = 1   # V[1*3+2] = V[5]
        W_s[6, col] = 1   # W[2*3+0] = W[6]
        col += 1
        U_s[2, col] = 1   # U[0*3+2] = U[2]
        V_s[7, col] = 1   # V[2*3+1] = V[7]
        W_s[3, col] = 1   # W[1*3+0] = W[3]
        col += 1
        U_s[2, col] = 1   # U[0*3+2] = U[2]
        V_s[8, col] = 1   # V[2*3+2] = V[8]
        W_s[6, col] = 1   # W[2*3+0] = W[6]
        col += 1
        
        # C21 part 1 = A21 * B11 (2x1 * 1x1) -> rank 2
        # C[1,0] += A[1,0]*B[0,0]
        U_s[3, col] = 1   # U[1*3+0] = U[3]
        V_s[0, col] = 1   # V[0*3+0] = V[0]
        W_s[1, col] = 1   # W[0*3+1] = W[1]
        col += 1
        # C[2,0] += A[2,0]*B[0,0]
        U_s[6, col] = 1   # U[2*3+0] = U[6]
        V_s[0, col] = 1   # V[0*3+0] = V[0]
        W_s[2, col] = 1   # W[0*3+2] = W[2]
        col += 1
        
        # C21 part 2 = A22 * B21 (2x2 * 2x1) -> rank 4
        # C[1,0] += A[1,1]*B[1,0] + A[1,2]*B[2,0]
        U_s[4, col] = 1   # U[1*3+1] = U[4]
        V_s[3, col] = 1   # V[1*3+0] = V[3]
        W_s[1, col] = 1   # W[0*3+1] = W[1]
        col += 1
        U_s[4, col] = 1   # U[1*3+1] = U[4]
        V_s[6, col] = 1   # V[2*3+0] = V[6]
        W_s[1, col] = 1   # W[0*3+1] = W[1]
        col += 1
        U_s[7, col] = 1   # U[2*3+1] = U[7]
        V_s[3, col] = 1   # V[1*3+0] = V[3]
        W_s[2, col] = 1   # W[0*3+2] = W[2]
        col += 1
        U_s[7, col] = 1   # U[2*3+1] = U[7]
        V_s[6, col] = 1   # V[2*3+0] = V[6]
        W_s[2, col] = 1   # W[0*3+2] = W[2]
        col += 1
        
        # C22 part 1 = A21 * B12 (2x1 * 1x2) -> rank 4
        # C[1,1] += A[1,0]*B[0,1]
        U_s[3, col] = 1   # U[1*3+0] = U[3]
        V_s[1, col] = 1   # V[0*3+1] = V[1]
        W_s[4, col] = 1   # W[1*3+1] = W[4]
        col += 1
        # C[1,2] += A[1,0]*B[0,2]
        U_s[3, col] = 1   # U[1*3+0] = U[3]
        V_s[2, col] = 1   # V[0*3+2] = V[2]
        W_s[7, col] = 1   # W[2*3+1] = W[7]
        col += 1
        # C[2,1] += A[2,0]*B[0,1]
        U_s[6, col] = 1   # U[2*3+0] = U[6]
        V_s[1, col] = 1   # V[0*3+1] = V[1]
        W_s[5, col] = 1   # W[1*3+2] = W[5]
        col += 1
        # C[2,2] += A[2,0]*B[0,2]
        U_s[6, col] = 1   # U[2*3+0] = U[6]
        V_s[2, col] = 1   # V[0*3+2] = V[2]
        W_s[8, col] = 1   # W[2*3+2] = W[8]
        col += 1
        
        # C22 part 2 = A22 * B22 (2x2 * 2x2) -> Strassen, rank 7
        # Local indices: i_l in {0,1} -> i_g = 1+i_l, j_l in {0,1} -> j_g = 1+j_l, k_l in {0,1} -> k_g = 1+k_l
        # U row: i_g*3+j_g, V row: j_g*3+k_g, W row: k_g*3+i_g
        # W index: k_g*3+i_g. C[1,1]:(k=1,i=1)->4, C[1,2]:(k=2,i=1)->7, C[2,1]:(k=1,i=2)->5, C[2,2]:(k=2,i=2)->8
        
        # M1 = (A11+A22)(B11+B22): coeff +1 in C[1,1], +1 in C[2,2]
        U_s[4, col] = 1; U_s[8, col] = 1  # A[1,1], A[2,2]
        V_s[4, col] = 1; V_s[8, col] = 1  # B[1,1], B[2,2]
        W_s[4, col] = 1; W_s[8, col] = 1  # C[1,1], C[2,2]
        col += 1

        # M2 = (A21+A22)B11: coeff +1 in C[2,1], -1 in C[2,2]
        U_s[7, col] = 1; U_s[8, col] = 1  # A[2,1], A[2,2]
        V_s[4, col] = 1  # B[1,1]
        W_s[5, col] = 1; W_s[8, col] = -1  # C[2,1], C[2,2]
        col += 1

        # M3 = A11(B12-B22): coeff +1 in C[1,2], +1 in C[2,2]
        U_s[4, col] = 1  # A[1,1]
        V_s[5, col] = 1; V_s[8, col] = -1  # B[1,2], B[2,2]
        W_s[7, col] = 1; W_s[8, col] = 1  # C[1,2], C[2,2]
        col += 1

        # M4 = A22(B21-B11): coeff +1 in C[1,1], +1 in C[2,1]
        U_s[8, col] = 1  # A[2,2]
        V_s[7, col] = 1; V_s[4, col] = -1  # B[2,1], B[1,1]
        W_s[4, col] = 1; W_s[5, col] = 1  # C[1,1], C[2,1]
        col += 1

        # M5 = (A11+A12)B22: coeff -1 in C[1,1], +1 in C[1,2]
        U_s[4, col] = 1; U_s[5, col] = 1  # A[1,1], A[1,2]
        V_s[8, col] = 1  # B[2,2]
        W_s[4, col] = -1; W_s[7, col] = 1  # C[1,1], C[1,2]
        col += 1

        # M6 = (A21-A11)(B11+B12): coeff +1 in C[2,2]
        U_s[7, col] = 1; U_s[4, col] = -1  # A[2,1], A[1,1]
        V_s[4, col] = 1; V_s[5, col] = 1  # B[1,1], B[1,2]
        W_s[8, col] = 1  # C[2,2]
        col += 1

        # M7 = (A12-A22)(B21+B22): coeff +1 in C[1,1]
        U_s[5, col] = 1; U_s[8, col] = -1  # A[1,2], A[2,2]
        V_s[7, col] = 1; V_s[8, col] = 1  # B[2,1], B[2,2]
        W_s[4, col] = 1  # C[1,1]
        col += 1
        
        assert col == 24, f"Expected 24 columns, got {col}"
        return U_s, V_s, W_s

    # Build the full matrices
    # We need to map local indices to global indices.
    # Global U: (25, Rank)
    # Global V: (25, Rank)
    # Global W: (25, Rank)
    
    total_rank = 109
    U_full = np.zeros((25, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((25, total_rank))
    
    col_offset = 0
    
    # C11: A11 B11 (2x2)
    U_b, V_b, W_b = strassen_block()
    # Map indices
    # U local (i,j) -> global (i,j)
    # V local (j,k) -> global (j,k)
    # W local (k,i) -> global (k,i)
    # i,j,k in {0,1}
    U_full[0:4, col_offset:col_offset+7] = U_b
    V_full[0:4, col_offset:col_offset+7] = V_b
    W_full[0:4, col_offset:col_offset+7] = W_b
    col_offset += 7
    
    # C12 Part 1: A11 B12 (2x2 * 2x3)
    # A11: i in {0,1}, j in {0,1}
    # B12: j in {0,1}, k in {2,3,4}
    # U: i in {0,1}, j in {0,1} -> 0..3
    # V: j in {0,1}, k in {2,3,4} -> j*5+k
    # W: k in {2,3,4}, i in {0,1} -> k*5+i
    U_b, V_b, W_b = schoolbook_block(2, 2, 3)
    # U_b is 4x12
    # V_b is 6x12 (j*3+k local)
    # W_b is 6x12 (k*2+i local)
    
    for c in range(12):
        for lj in range(2):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[lj*5 + (2+lk), col_offset+c] = val
    # U is same indices
    U_full[0:4, col_offset:col_offset+12] = U_b
    # W: local k in {0,1,2}, local i in {0,1}
    # W local index lk*2+li -> global (2+lk)*5 + li
    for c in range(12):
        for lk in range(3):
            for li in range(2):
                val = W_b[lk*2+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + li, col_offset+c] = val
    col_offset += 12
    
    # C12 Part 2: A12 B22 (2x3 * 3x3)
    # A12: i in {0,1}, j in {2,3,4}
    # B22: j in {2,3,4}, k in {2,3,4}
    U_b, V_b, W_b = schoolbook_block(2, 3, 3)
    # U: i in {0,1}, j in {2,3,4} -> i*5+j
    # V: j in {2,3,4}, k in {2,3,4} -> j*5+k
    # W: k in {2,3,4}, i in {0,1} -> k*5+i
    for c in range(18):
        for li in range(2):
            for lj in range(3):
                val = U_b[li*3+lj, c]
                if val != 0:
                    U_full[li*5 + (2+lj), col_offset+c] = val
        for lj in range(3):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[(2+lj)*5 + (2+lk), col_offset+c] = val
        for lk in range(3):
            for li in range(2):
                val = W_b[lk*2+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + li, col_offset+c] = val
    col_offset += 18
    
    # C21 Part 1: A21 B11 (3x2 * 2x2)
    U_b, V_b, W_b = schoolbook_block(3, 2, 2)
    for c in range(12):
        for li in range(3):
            for lj in range(2):
                val = U_b[li*2+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + lj, col_offset+c] = val
        for lj in range(2):
            for lk in range(2):
                val = V_b[lj*2+lk, c]
                if val != 0:
                    V_full[lj*5 + lk, col_offset+c] = val
        for lk in range(2):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[lk*5 + (2+li), col_offset+c] = val
    col_offset += 12
    
    # C21 Part 2: A22 B21 (3x3 * 3x2)
    U_b, V_b, W_b = schoolbook_block(3, 3, 2)
    for c in range(18):
        for li in range(3):
            for lj in range(3):
                val = U_b[li*3+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + (2+lj), col_offset+c] = val
        for lj in range(3):
            for lk in range(2):
                val = V_b[lj*2+lk, c]
                if val != 0:
                    V_full[(2+lj)*5 + lk, col_offset+c] = val
        for lk in range(2):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[lk*5 + (2+li), col_offset+c] = val
    col_offset += 18
    
    # C22 Part 1: A21 B12 (3x2 * 2x3)
    U_b, V_b, W_b = schoolbook_block(3, 2, 3)
    for c in range(18):
        for li in range(3):
            for lj in range(2):
                val = U_b[li*2+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + lj, col_offset+c] = val
        for lj in range(2):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[lj*5 + (2+lk), col_offset+c] = val
        for lk in range(3):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + (2+li), col_offset+c] = val
    col_offset += 18
    
    # C22 Part 2: A22 B22 (3x3 * 3x3) using Strassen 1+2 split
    U_b, V_b, W_b = strassen_3x3_block()
    # Local indices: i,j,k in {0,1,2}
    # Global: i_g = 2+li, j_g = 2+lj, k_g = 2+lk
    for c in range(24):
        for li in range(3):
            for lj in range(3):
                val = U_b[li*3+lj, c]
                if val != 0:
                    U_full[(2+li)*5 + (2+lj), col_offset+c] = val
        for lj in range(3):
            for lk in range(3):
                val = V_b[lj*3+lk, c]
                if val != 0:
                    V_full[(2+lj)*5 + (2+lk), col_offset+c] = val
        for lk in range(3):
            for li in range(3):
                val = W_b[lk*3+li, c]
                if val != 0:
                    W_full[(2+lk)*5 + (2+li), col_offset+c] = val
    col_offset += 24
    
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    # Verify the constructed decomposition
    err = verify(U_full, V_full, W_full, total_rank)
    if err < 1e-8:
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
        U_school = np.zeros((25, 125))
        V_school = np.zeros((25, 125))
        W_school = np.zeros((25, 125))
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U_school[i*5+j, col] = 1
                    V_school[j*5+k, col] = 1
                    W_school[k*5+i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
