import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    def get_schoolbook():
        rank = n * m * p  # 42
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def get_strassen_2x2x2():
        """Rank-7 decomposition of 2x2x2 matrix multiplication."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Standard Strassen 7 products
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_2x1x2_rank4():
        """Rank-4 decomposition of 2x1x2 (vector-matrix product)."""
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

    def get_2x3x1_rank6():
        """Rank-6 decomposition of 2x3x1 (matrix-vector product)."""
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    def get_2x3x2_rank11():
        """Rank-11 decomposition of 2x3x2 matrix multiplication.
        Combines Strassen (2x2x2, rank 7) and 2x1x2 (rank 4)."""
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_212, V_212, W_212 = get_2x1x2_rank4()
        
        # Strassen part handles columns 0-1 of the 3x2 and 2x2
        # U rows: 0,1 for i=0; 2,3 for i=1 (but we only use 2x2 part for i=0,1)
        # Actually, the 2x3x2 problem: A is 2x3, B is 3x2, C is 2x2
        # We split B into B[:,0:2] (full) and use Strassen on the 2x2x2 part
        # plus handle the extra column of A
        
        # The 2x3x2 can be seen as:
        # C = A @ B where A is 2x3, B is 3x2
        # Split A into A[:,0:2] (2x2) and A[:,2:3] (2x1)
        # C = A[:,0:2] @ B + A[:,2:3] @ B[:,2:3]... no that's not right
        
        # Better: C[i,k] = sum_j A[i,j] B[j,k] for j=0,1,2
        # = sum_j=0,1 A[i,j] B[j,k] + A[i,2] B[2,k]
        # First term: 2x2x2 multiplication (rank 7 with Strassen)
        # Second term: 2x1x2 multiplication (rank 4)
        # Total: 7 + 4 = 11
        
        # Strassen part: handles j=0,1 (first 2 columns of A, first 2 rows of B)
        # U_s is 4x7 for 2x2x2, where U rows are (i,j) for i=0,1; j=0,1
        # We need to map: U row i*2+j for i=0,1; j=0,1 -> rows 0,1,2,3
        # But our U for 2x3x2 has rows i*3+j for i=0,1; j=0,1,2 -> rows 0,1,2,3,4,5
        
        # For Strassen part (j=0,1):
        U[0, 0:7] = U_s[0, :]  # i=0, j=0
        U[1, 0:7] = U_s[1, :]  # i=0, j=1
        U[3, 0:7] = U_s[2, :]  # i=1, j=0
        U[4, 0:7] = U_s[3, :]  # i=1, j=1
        
        # V for Strassen part: V rows are (j,k) for j=0,1; k=0,1
        # V_s is 4x7 for (j,k) -> rows 0,1,2,3
        # Our V for 2x3x2 has rows j*2+k for j=0,1,2; k=0,1 -> rows 0,1,2,3,4,5,6,7,8
        V[0, 0:7] = V_s[0, :]  # j=0, k=0
        V[1, 0:7] = V_s[1, :]  # j=0, k=1
        V[2, 0:7] = V_s[2, :]  # j=1, k=0
        V[3, 0:7] = V_s[3, :]  # j=1, k=1
        
        # W for Strassen part: W rows are (k,i) for k=0,1; i=0,1
        W[0, 0:7] = W_s[0, :]  # k=0, i=0
        W[1, 0:7] = W_s[1, :]  # k=0, i=1
        W[2, 0:7] = W_s[2, :]  # k=1, i=0
        W[3, 0:7] = W_s[3, :]  # k=1, i=1
        
        # 2x1x2 part: handles j=2 (third column of A, third row of B)
        # U_212 is 2x4 for (i,j) where i=0,1; j=0 (the single column)
        # Map to U rows: i*3+2 for i=0,1 -> rows 2, 5
        U[2, 7:11] = U_212[0, :]  # i=0, j=2
        U[5, 7:11] = U_212[1, :]  # i=1, j=2
        
        # V_212 is 2x4 for (j,k) where j=0 (single row); k=0,1
        # Map to V rows: 2*2+k for k=0,1 -> rows 4, 5
        V[4, 7:11] = V_212[0, :]  # j=2, k=0
        V[5, 7:11] = V_212[1, :]  # j=2, k=1
        
        # W_212 is 4x4 for (k,i) where k=0,1; i=0,1
        # Map to W rows: k*2+i for k=0,1; i=0,1 -> rows 0,1,2,3
        W[0, 7:11] = W_212[0, :]  # k=0, i=0
        W[1, 7:11] = W_212[1, :]  # k=0, i=1
        W[2, 7:11] = W_212[2, :]  # k=1, i=0
        W[3, 7:11] = W_212[3, :]  # k=1, i=1
        
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    # Construct full 2x3x7 using partition of 7 = 2+2+2+1
    # Three 2x3x2 blocks (rank 11 each) + one 2x3x1 block (rank 6) = 39
    U11_a, V11_a, W11_a = get_2x3x2_rank11()
    U11_b, V11_b, W11_b = get_2x3x2_rank11()
    U11_c, V11_c, W11_c = get_2x3x2_rank11()
    U6, V6, W6 = get_2x3x1_rank6()
    
    R_total = 11 + 11 + 11 + 6  # 39
    U_full = np.zeros((N_U, R_total))
    V_full = np.zeros((N_V, R_total))
    W_full = np.zeros((N_W, R_total))
    
    # Block 1: cols 0-1 of B, cols 0-1 of C
    U_full[:, 0:11] = U11_a
    for j in range(3):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + k
            V_full[global_idx, 0:11] = V11_a[local_idx, :]
    for k in range(2):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = k * 2 + i
            W_full[global_idx, 0:11] = W11_a[local_idx, :]
            
    # Block 2: cols 2-3 of B, cols 2-3 of C
    U_full[:, 11:22] = U11_b
    for j in range(3):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + (k + 2)
            V_full[global_idx, 11:22] = V11_b[local_idx, :]
    for k in range(2):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = (k + 2) * 2 + i
            W_full[global_idx, 11:22] = W11_b[local_idx, :]
            
    # Block 3: cols 4-5 of B, cols 4-5 of C
    U_full[:, 22:33] = U11_c
    for j in range(3):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + (k + 4)
            V_full[global_idx, 22:33] = V11_c[local_idx, :]
    for k in range(2):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = (k + 4) * 2 + i
            W_full[global_idx, 22:33] = W11_c[local_idx, :]
            
    # Block 4: col 6 of B, col 6 of C
    U_full[:, 33:39] = U6
    for j in range(3):
        local_idx = j
        global_idx = j * 7 + 6
        V_full[global_idx, 33:39] = V6[local_idx, :]
    for i in range(2):
        local_idx = i
        global_idx = 6 * 2 + i
        W_full[global_idx, 33:39] = W6[local_idx, :]
        
    # Verify the decomposition
    if verify_decomposition(U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    
    # Fallback to schoolbook if verification fails
    return get_schoolbook()

# EVOLVE-BLOCK-END
