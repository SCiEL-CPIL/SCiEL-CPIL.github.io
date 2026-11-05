"""Matrix multiplication 5x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-117 decomposition for 5x5x5 matrix multiplication.
    
    Strategy: Partition each 5x5 matrix into blocks of sizes (2,2,1).
    - A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]] where A11,A22,A33 are 2x2 or 1x1
    - Use Strassen's rank-7 algorithm for the 2x2 sub-blocks (A11*A21, etc.)
    - Use schoolbook for the remaining multiplications
    
    The 5x5 matrix product C = A*B where A and B are 5x5:
    Partition A into blocks:
        A = [[A11(2x2), A12(2x2), A13(2x1)],
             [A21(2x2), A22(2x2), A23(2x1)],
             [A31(1x2), A32(1x2), A33(1x1)]]
    Partition B into blocks:
        B = [[B11(2x2), B12(2x2), B13(2x2)],
             [B21(2x2), B22(2x2), B23(2x2)],
             [B31(1x2), B32(1x2), B33(1x1)]]
    
    Then C = A*B has blocks:
        C11(2x2) = A11*B11 + A12*B21 + A13*B31
        C12(2x2) = A11*B12 + A12*B22 + A13*B32
        C13(2x1) = A11*B13 + A12*B23 + A13*B33
        C21(2x2) = A21*B11 + A22*B21 + A23*B31
        C22(2x2) = A21*B12 + A22*B22 + A23*B32
        C23(2x1) = A21*B13 + A22*B23 + A23*B33
        C31(1x2) = A31*B11 + A32*B21 + A33*B31
        C32(1x2) = A31*B12 + A32*B22 + A33*B32
        C33(1x1) = A31*B13 + A32*B23 + A33*B33
    
    For the 2x2 block multiplications (like A11*B11), we can use Strassen's 7-multiplication algorithm.
    For mixed-size blocks (2x2 * 2x1, 2x1 * 2x2, 2x1 * 1x1, 1x2 * 2x2, etc.), we use schoolbook.
    
    Let me count the rank:
    - 2x2 * 2x2 multiplications: C11 has A11*B11, C12 has A12*B22, C21 has A22*B21, C22 has A21*B12... 
      Wait, let me be more careful.
      
    Actually, let me think about this differently. The key insight from the knowledge is:
    "Using Strassen's rank-7 algorithm for the eight 2x2x2 sub-blocks in a 2+2+1 partition reduces the total rank from 125 to 117"
    
    So there are 8 instances of 2x2x2 sub-problems where we can apply Strassen.
    Each 2x2x2 schoolbook costs 8, Strassen costs 7, saving 1 per instance.
    8 instances * 1 saving = 8 savings. 125 - 8 = 117.
    
    Let me identify the 8 2x2x2 sub-problems:
    Looking at C = A*B with the (2,2,1) partition:
    
    The 2x2 * 2x2 -> 2x2 multiplications that appear:
    - In C11(2x2): A11(2x2) * B11(2x2) -> 2x2
    - In C12(2x2): A12(2x2) * B22(2x2) -> 2x2  ... wait no
      C12 = A11*B12 + A12*B22 + A13*B32
      A11(2x2)*B12(2x2) -> 2x2
      A12(2x2)*B22(2x2) -> 2x2
      A13(2x1)*B32(1x2) -> 2x2
      
    Hmm, let me recount. For each output block C_ij of size (s_i x s_j) where s = (2,2,1):
    C_ij = sum_k A_ik * B_kj
    
    The term A_ik * B_kj has dimensions (s_i x s_k) * (s_k x s_j) = s_i x s_j.
    
    2x2x2 sub-problems occur when s_i=2, s_k=2, s_j=2:
    - i=1,k=1,j=1: A11*B11 in C11
    - i=1,k=1,j=2: A11*B12 in C12
    - i=1,k=2,j=1: A12*B21 in C12? No, A12(2x2)*B21(2x2) in C12
    - i=1,k=2,j=2: A12*B22 in C12
    - i=2,k=1,j=1: A21*B11 in C21
    - i=2,k=1,j=2: A21*B12 in C21
    - i=2,k=2,j=1: A22*B21 in C22
    - i=2,k=2,j=2: A22*B22 in C22
    
    That's 8 instances of 2x2 * 2x2 -> 2x2. Each can use Strassen (rank 7 instead of 8).
    
    Total schoolbook rank: 5*5*5 = 125
    Savings: 8 * (8-7) = 8
    Total rank: 125 - 8 = 117
    
    Now I need to construct the actual U, V, W matrices.
    """
    n, m, p = DIMENSIONS  # n=5, m=5, p=5
    rank = 117
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # (25, 117)
    V = [[0] * rank for _ in range(m * p)]  # (25, 117)
    W = [[0] * rank for _ in range(p * n)]  # (25, 117)
    
    # Block sizes: s = [2, 2, 1]
    s = [2, 2, 1]
    
    # Define block row/column start indices
    def block_start(idx):
        """Start index of block idx (0-based) in the 5-element vector."""
        if idx == 0:
            return 0
        elif idx == 1:
            return s[0]  # 2
        else:
            return s[0] + s[1]  # 4
    
    # We'll assign rank indices as we go
    rank_idx = 0
    
    # For each output block (i,j) and intermediate (k), handle A_ik * B_kj
    for i in range(3):  # output block row
        for j in range(3):  # output block col
            for k in range(3):  # intermediate block
                si, sk, sj = s[i], s[k], s[j]
                
                if si == 2 and sk == 2 and sj == 2:
                    # Use Strassen's 7-multiplication algorithm
                    # A_ik is 2x2, B_kj is 2x2, result is 2x2
                    # Strassen:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B12-B22)
                    # M4 = A22(B21-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # A_ik block starts at row block_start(i) in A, col block_start(k) in A
                    # B_kj block starts at row block_start(k) in B, col block_start(j) in B
                    # C_ij block starts at row block_start(i) in C, col block_start(j) in C
                    
                    a_r0 = block_start(i)  # row start of A_ik in A
                    a_c0 = block_start(k)  # col start of A_ik in A
                    b_r0 = block_start(k)  # row start of B_kj in B
                    b_c0 = block_start(j)  # col start of B_kj in B
                    c_r0 = block_start(i)  # row start of C_ij in C
                    c_c0 = block_start(j)  # col start of C_ij in C
                    
                    # Strassen factors
                    # M1 = (A11+A22)(B11+B22)
                    # U: rows a_r0..a_r0+1, cols a_c0..a_c0+1 for A11 and A22
                    # V: rows b_r0..b_r0+1, cols b_c0..b_c0+1 for B11 and B22
                    
                    # M1
                    U[a_r0][rank_idx] = 1; U[a_r0+1][rank_idx] = 1  # A11 diag? No...
                    # Wait, I need to be more careful.
                    # A_ik is a 2x2 matrix with elements:
                    # A_ik[0,0] = A[a_r0, a_c0]
                    # A_ik[0,1] = A[a_r0, a_c0+1]
                    # A_ik[1,0] = A[a_r0+1, a_c0]
                    # A_ik[1,1] = A[a_r0+1, a_c0+1]
                    #
                    # B_kj is a 2x2 matrix with elements:
                    # B_kj[0,0] = B[b_r0, b_c0]
                    # B_kj[0,1] = B[b_r0, b_c0+1]
                    # B_kj[1,0] = B[b_r0+1, b_c0]
                    # B_kj[1,1] = B[b_r0+1, b_c0+1]
                    #
                    # For Strassen M1 = (A11+A22)(B11+B22):
                    # U part: A11 and A22 both contribute with +1
                    # V part: B11 and B22 both contribute with +1
                    # W part: contributes to C11, C22 with +1, C12 with 0, C21 with 0... 
                    # Actually W contributes to the output C_ij.
                    
                    # Let me define helper functions
                    def strassen_add_U(row, col, val):
                        """Add val to U[row, rank_idx]"""
                        U[row][rank_idx] += val
                    
                    def strassen_add_V(row, col, val):
                        """Add val to V[row, rank_idx]"""
                        V[row][rank_idx] += val
                    
                    def strassen_add_W(row, col, val):
                        """Add val to W[row, rank_idx]"""
                        W[row][rank_idx] += val
                    
                    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                    # U: A[0,0] and A[1,1] each get +1
                    # V: B[0,0] and B[1,1] each get +1
                    # W: C[0,0] gets +1, C[1,1] gets +1
                    strassen_add_U(a_r0, a_c0, 1)
                    strassen_add_U(a_r0+1, a_c0+1, 1)
                    strassen_add_V(b_r0, b_c0, 1)
                    strassen_add_V(b_r0+1, b_c0+1, 1)
                    strassen_add_W(c_c0 + c_r0*n, c_r0, 1)  # W[k*n+i, r] where k=0,i=0,r=0 -> W[0*n+0, 0]
                    # Wait, W indexing: W[k*n+i, r]
                    # For output C[i,j], the element C[i,j] corresponds to W[j*n+i, r]
                    # where r is the row in W.
                    # C[0,0] in the block means global C[c_r0, c_c0], so W[c_c0*n + c_r0, r]
                    # C[1,1] in the block means global C[c_r0+1, c_c0+1], so W[(c_c0+1)*n + (c_r0+1), r]
                    
                    # Let me redo: W index for C[di, dj] in block (i,j) is:
                    # W[(c_c0+dj)*n + (c_r0+di), rank_idx]
                    
                    strassen_add_W(c_c0*n + c_r0, 1)  # C[0,0] -> W[0*n+0, r] for di=0,dj=0
                    strassen_add_W((c_c0+1)*n + (c_r0+1), 1)  # C[1,1] -> W[1*n+1, r] for di=1,dj=1
                    
                    rank_idx += 1
                    
                    # M2 = (A[1,0]+A[1,1]) * B[0,0]
                    strassen_add_U(a_r0+1, a_c0, 1)
                    strassen_add_U(a_r0+1, a_c0+1, 1)
                    strassen_add_V(b_r0, b_c0, 1)
                    # W: C[1,0] gets +1
                    strassen_add_W(c_c0*n + (c_r0+1), 1)
                    
                    rank_idx += 1
                    
                    # M3 = A[0,0] * (B[0,1]-B[1,1])
                    strassen_add_U(a_r0, a_c0, 1)
                    strassen_add_V(b_r0, b_c0+1, 1)
                    strassen_add_V(b_r0+1, b_c0+1, -1)
                    # W: C[0,1] gets +1
                    strassen_add_W((c_c0+1)*n + c_r0, 1)
                    
                    rank_idx += 1
                    
                    # M4 = A[1,1] * (B[1,0]-B[0,0])
                    strassen_add_U(a_r0+1, a_c0+1, 1)
                    strassen_add_V(b_r0+1, b_c0, 1)
                    strassen_add_V(b_r0, b_c0, -1)
                    # W: C[0,0] gets +1, C[1,0] gets +1
                    strassen_add_W(c_c0*n + c_r0, 1)
                    strassen_add_W(c_c0*n + (c_r0+1), 1)
                    
                    rank_idx += 1
                    
                    # M5 = (A[0,0]+A[0,1]) * B[1,1]
                    strassen_add_U(a_r0, a_c0, 1)
                    strassen_add_U(a_r0, a_c0+1, 1)
                    strassen_add_V(b_r0+1, b_c0+1, 1)
                    # W: C[0,1] gets +1, C[1,1] gets -1
                    strassen_add_W((c_c0+1)*n + c_r0, 1)
                    strassen_add_W((c_c0+1)*n + (c_r0+1), -1)
                    
                    rank_idx += 1
                    
                    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
                    strassen_add_U(a_r0+1, a_c0, 1)
                    strassen_add_U(a_r0, a_c0, -1)
                    strassen_add_V(b_r0, b_c0, 1)
                    strassen_add_V(b_r0, b_c0+1, 1)
                    # W: C[1,1] gets +1
                    strassen_add_W((c_c0+1)*n + (c_r0+1), 1)
                    
                    rank_idx += 1
                    
                    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
                    strassen_add_U(a_r0, a_c0+1, 1)
                    strassen_add_U(a_r0+1, a_c0+1, -1)
                    strassen_add_V(b_r0+1, b_c0, 1)
                    strassen_add_V(b_r0+1, b_c0+1, 1)
                    # W: C[0,0] gets +1
                    strassen_add_W(c_c0*n + c_r0, 1)
                    
                    rank_idx += 1
                    
                else:
                    # Schoolbook for this sub-block
                    for di in range(si):
                        for dk in range(sk):
                            for dj in range(sj):
                                # A_ik[di,dk] * B_kj[dk,dj] -> C_ij[di,dj]
                                U[a_r0+di][rank_idx] = 1
                                V[b_r0+dk][rank_idx] = 1
                                W[(c_c0+dj)*n + (c_r0+di)][rank_idx] = 1
                                rank_idx += 1
    
    # Verify rank_idx equals rank
    assert rank_idx == rank, f"Expected rank {rank}, got {rank_idx}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
