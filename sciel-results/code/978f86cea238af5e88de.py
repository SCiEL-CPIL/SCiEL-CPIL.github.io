"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # A is 2x5. Cols partitioned 2, 2, 1.
    # B is 5x5. Rows partitioned 2, 2, 1. Cols partitioned 2, 2, 1.
    
    # We build U, V, W matrices.
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (10, rank)
    # Note: W[k*n+i, r] corresponds to C[i, k]
    
    # We will append columns to U, V, W for each block.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(u_vec, v_vec, w_vec):
        """
        u_vec: length 10 (flattened A indices i*m+j)
        v_vec: length 25 (flattened B indices j*p+k)
        w_vec: length 10 (flattened C indices k*n+i)
        """
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to create zero vectors
    def zero_U(): return [0.0] * (n * m)
    def zero_V(): return [0.0] * (m * p)
    def zero_W(): return [0.0] * (p * n)
    
    # Indices for A (2x5): i in 0..1, j in 0..4. Flat idx = i*5 + j
    # Indices for B (5x5): j in 0..4, k in 0..4. Flat idx = j*5 + k
    # Indices for C (2x5): i in 0..1, k in 0..4. Flat idx = k*2 + i (Trace convention W[k*n+i])
    
    # Partitions
    # J (cols of A, rows of B): 0,1 | 2,3 | 4
    # K (cols of B, cols of C): 0,1 | 2,3 | 4
    
    # Block (J1, K1): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen 2x2
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_2x2(j_start, k_start):
        # A indices: i in 0..1, j in j_start..j_start+1
        # B indices: j in j_start..j_start+1, k in k_start..k_start+1
        # C indices: i in 0..1, k in k_start..k_start+1
        
        # M1
        u = zero_U(); v = zero_V(); w = zero_W()
        # A11+A22
        u[j_start] = 1; u[j_start+1] = 1 # A[0, j_start] + A[0, j_start+1]? No.
        # A11 is A[0, j_start]. A22 is A[1, j_start+1].
        # Flat A idx: i*5+j.
        u[0*5 + j_start] = 1
        u[1*5 + (j_start+1)] = 1
        # B11+B22
        # B11 is B[j_start, k_start]. B22 is B[j_start+1, k_start+1].
        # Flat B idx: j*5+k.
        v[j_start*5 + k_start] = 1
        v[(j_start+1)*5 + (k_start+1)] = 1
        # C11 and C22 get +1
        w[k_start*2 + 0] = 1
        w[(k_start+1)*2 + 1] = 1
        add_block(u, v, w)
        
        # M2
        u = zero_U(); v = zero_V(); w = zero_W()
        # A21+A22
        u[1*5 + j_start] = 1
        u[1*5 + (j_start+1)] = 1
        # B11
        v[j_start*5 + k_start] = 1
        # C21 +1, C22 -1
        w[k_start*2 + 1] = 1
        w[(k_start+1)*2 + 1] = -1
        add_block(u, v, w)
        
        # M3
        u = zero_U(); v = zero_V(); w = zero_W()
        # A11
        u[0*5 + j_start] = 1
        # B12-B22
        v[j_start*5 + (k_start+1)] = 1
        v[(j_start+1)*5 + (k_start+1)] = -1
        # C12 +1, C22 +1
        w[(k_start+1)*2 + 0] = 1
        w[(k_start+1)*2 + 1] = 1
        add_block(u, v, w)
        
        # M4
        u = zero_U(); v = zero_V(); w = zero_W()
        # A22
        u[1*5 + (j_start+1)] = 1
        # B21-B11
        v[(j_start+1)*5 + k_start] = 1
        v[j_start*5 + k_start] = -1
        # C11 +1, C21 +1
        w[k_start*2 + 0] = 1
        w[k_start*2 + 1] = 1
        add_block(u, v, w)
        
        # M5
        u = zero_U(); v = zero_V(); w = zero_W()
        # A11+A12
        u[0*5 + j_start] = 1
        u[0*5 + (j_start+1)] = 1
        # B22
        v[(j_start+1)*5 + (k_start+1)] = 1
        # C11 -1, C12 +1
        w[k_start*2 + 0] = -1
        w[(k_start+1)*2 + 0] = 1
        add_block(u, v, w)
        
        # M6
        u = zero_U(); v = zero_V(); w = zero_W()
        # A21-A11
        u[1*5 + j_start] = 1
        u[0*5 + j_start] = -1
        # B11+B12
        v[j_start*5 + k_start] = 1
        v[j_start*5 + (k_start+1)] = 1
        # C22 +1
        w[(k_start+1)*2 + 1] = 1
        add_block(u, v, w)
        
        # M7
        u = zero_U(); v = zero_V(); w = zero_W()
        # A12-A22
        u[0*5 + (j_start+1)] = 1
        u[1*5 + (j_start+1)] = -1
        # B21+B22
        v[(j_start+1)*5 + k_start] = 1
        v[(j_start+1)*5 + (k_start+1)] = 1
        # C11 +1
        w[k_start*2 + 0] = 1
        add_block(u, v, w)
        
    # Add 4 Strassen blocks
    add_strassen_2x2(0, 0) # J1, K1
    add_strassen_2x2(0, 2) # J1, K2
    add_strassen_2x2(2, 0) # J2, K1
    add_strassen_2x2(2, 2) # J2, K2
    
    # Block (J1, K3): A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # A 2x2, B 2x1. Rank 2.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # M1 = (A11+A21) B1 ? No.
    # M1 = A11 B1 + A12 B2? No.
    # M1 = (A11+A21) B1? No.
    # Standard:
    # M1 = A11 B1
    # M2 = A12 B2
    # M3 = A21 B1
    # M4 = A22 B2
    # C1 = M1 + M2
    # C2 = M3 + M4
    # Rank 4?
    # Can we do rank 2?
    # C1 = (A11+A21) B1? No.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # Let M1 = (A11+A21) B1? No.
    # Let M1 = (A11+A12) B1? No.
    # Actually, rank of 2x2x1 is 2.
    # C1 = M1 + M2
    # C2 = M3 + M4
    # Is it possible with 2 terms?
    # T_{i,j,k}. i in 1,2. j in 1,2. k in 1.
    # T = [[1, 1], [1, 1]] for k=1? No.
    # T_{1,1,1}=1, T_{1,2,1}=1
    # T_{2,1,1}=1, T_{2,2,1}=1
    # Rank 1?
    # If rank 1, C1 = c1 L(A) R(B), C2 = c2 L(A) R(B).
    # L(A) must be linear combo of A11, A12 for C1 and A21, A22 for C2.
    # Impossible.
    # So rank 4?
    # Wait, 2x2x1 rank is 2?
    # Let's check literature. Rank of 2x2x1 is 2?
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # M1 = (A11+A21) B1? No.
    # M1 = A11 B1 + A21 B2? No.
    # Actually, we can do:
    # M1 = (A11+A21) B1? No.
    # M1 = (A11+A12) B1? No.
    # Let's just use rank 4 to be safe? 4*2 = 8.
    # Total rank 28 + 8 + 2 + 2 + 2 = 42. Still better than 46.
    # But let's try rank 2.
    # M1 = (A11+A21) B1? No.
    # M1 = (A11+A22) B1? No.
    # I will use rank 4 for these blocks.
    
    def add_block_2x2x1(j_start, k_start):
        # A[0:2, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+1]
        # j_start is 0 or 2. k_start is 4.
        # M1: A[0, j_start] B[j_start, k_start]
        u = zero_U(); v = zero_V(); w = zero_W()
        u[0*5 + j_start] = 1
        v[j_start*5 + k_start] = 1
        w[k_start*2 + 0] = 1
        add_block(u, v, w)
        
        # M2: A[0, j_start+1] B[j_start+1, k_start]
        u = zero_U(); v = zero_V(); w = zero_W()
        u[0*5 + (j_start+1)] = 1
        v[(j_start+1)*5 + k_start] = 1
        w[k_start*2 + 0] = 1
        add_block(u, v, w)
        
        # M3: A[1, j_start] B[j_start, k_start]
        u = zero_U(); v = zero_V(); w = zero_W()
        u[1*5 + j_start] = 1
        v[j_start*5 + k_start] = 1
        w[k_start*2 + 1] = 1
        add_block(u, v, w)
        
        # M4: A[1, j_start+1] B[j_start+1, k_start]
        u = zero_U(); v = zero_V(); w = zero_W()
        u[1*5 + (j_start+1)] = 1
        v[(j_start+1)*5 + k_start] = 1
        w[k_start*2 + 1] = 1
        add_block(u, v, w)
        
    add_block_2x2x1(0, 4) # J1, K3
    add_block_2x2x1(2, 4) # J2, K3
    
    # Block (J3, K1): A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]
    # A 2x1, B 1x2. Outer product. Rank 1.
    # C_{i,k} = A[i,4] B[4,k]
    # M1 = A[0,4] B[4,0]? No.
    # M1 = (A[0,4] + A[1,4]) (B[4,0] + B[4,1])? No.
    # M1 = A[0,4] B[4,0] + A[1,4] B[4,1]? No.
    # M1 = A[0,4] B[4,0]
    # M2 = A[0,4] B[4,1]
    # M3 = A[1,4] B[4,0]
    # M4 = A[1,4] B[4,1]
    # Rank 4?
    # No, rank 1.
    # M1 = (A[0,4]) (B[4,0])?
    # C[0,0] = A[0,4] B[4,0]
    # C[0,1] = A[0,4] B[4,1]
    # C[1,0] = A[1,4] B[4,0]
    # C[1,1] = A[1,4] B[4,1]
    # M1 = A[0,4] B[4,0] -> C[0,0]
    # M2 = A[0,4] B[4,1] -> C[0,1]
    # M3 = A[1,4] B[4,0] -> C[1,0]
    # M4 = A[1,4] B[4,1] -> C[1,1]
    # This is rank 4.
    # Can we do rank 1?
    # M1 = (A[0,4] + A[1,4]) (B[4,0] + B[4,1])?
    # No.
    # Rank of 2x1x2 is 1?
    # T_{i,j,k}. j fixed to 4.
    # T_{0,4,0}=1, T_{0,4,1}=1, T_{1,4,0}=1, T_{1,4,1}=1.
    # T = u v^T w^*?
    # u=[1,1]^T, v=[1]^T, w=[1,1]^T.
    # T_{i,j,k} = u_i v_j w_k = 1.
    # Yes, rank 1.
    # M1 = (A[0,4] + A[1,4])? No.
    # M1 = (A[0,4]) (B[4,0])?
    # Decomposition:
    # C_{i,k} = W_{i,k,1} L_1(A) R_1(B)
    # L_1(A) = A[0,4] + A[1,4]? No.
    # L_1(A) must generate A[0,4] and A[1,4] separately?
    # No, L_1(A) is a single linear form.
    # If L_1(A) = A[0,4], then C[1,k] = 0.
    # So rank 1 is impossible.
    # Rank must be 2?
    # M1 = A[0,4] (B[4,0] + B[4,1])?
    # C[0,0] = M1
    # C[0,1] = M1
    # C[1,0] = 0
    # No.
    # Rank 4 is safe.
    
    def add_block_2x1x2(j_start, k_start):
        # A[0:2, j_start:j_start+1] * B[j_start:j_start+1, k_start:k_start+2]
        # j_start is 4. k_start is 0 or 2.
        for i in range(2):
            for k in range(2):
                u = zero_U(); v = zero_V(); w = zero_W()
                u[i*5 + j_start] = 1
                v[j_start*5 + (k_start+k)] = 1
                w[(k_start+k)*2 + i] = 1
                add_block(u, v, w)
                
    add_block_2x1x2(4, 0) # J3, K1
    add_block_2x1x2(4, 2) # J3, K2
    
    # Block (J3, K3): A[0:2, 4:5] * B[4:5, 4:5] -> C[0:2, 4:5]
    # A 2x1, B 1x1. Rank 2.
    # C[0,4] = A[0,4] B[4,4]
    # C[1,4] = A[1,4] B[4,4]
    # M1 = A[0,4] B[4,4]
    # M2 = A[1,4] B[4,4]
    
    def add_block_2x1x1(j_start, k_start):
        for i in range(2):
            u = zero_U(); v = zero_V(); w = zero_W()
            u[i*5 + j_start] = 1
            v[j_start*5 + k_start] = 1
            w[k_start*2 + i] = 1
            add_block(u, v, w)
            
    add_block_2x1x1(4, 4) # J3, K3
    
    # Build final matrices
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
