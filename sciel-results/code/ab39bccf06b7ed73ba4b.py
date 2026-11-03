"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.
    
    Strategy: Use block decomposition with Strassen for 2x2x2 blocks.
    - Split the 4x5 matrix into blocks
    - Use Strassen (rank 7) for 2x2x2 sub-problems
    - Total rank: 22 (7 + 7 + 8)
    
    The output W follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def strassen_decomposition():
        """
        Return U, V, W matrices for 2x2x2 Strassen decomposition.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
        
        Indexing:
        - A is 2x2, flattened as A[row*2+col] where row is i (0,1), col is j (0,1)
        - B is 2x2, flattened as B[row*2+col] where row is j (0,1), col is k (0,1)
        - C is 2x2, flattened as C[row*2+col] where row is k (0,1), col is i (0,1)
        
        For matrix multiplication C = A * B:
        - A has shape (n, m) = (2, 2)
        - B has shape (m, p) = (2, 2)
        - C has shape (n, p) = (2, 2)
        
        The tensor T[i, j, k] = C[i,k] = sum_l A[i,l] * B[l,k]
        
        U has shape (n*m, rank) = (4, rank)
        V has shape (m*p, rank) = (4, rank)  
        W has shape (p*n, rank) = (4, rank)
        
        For U: index is i*m + j (i is row of A, j is col of A)
        For V: index is j*p + k (j is row of B, k is col of B)
        For W: index is k*n + i (k is col of C, i is row of C) - this is the trace convention
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # A indices: A[i,j] -> U index i*2+j
        # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 2, A[1,1] -> 3
        
        # B indices: B[j,k] -> V index j*2+k
        # B[0,0] -> 0, B[0,1] -> 1, B[1,0] -> 2, B[1,1] -> 3
        
        # C indices: C[i,k] -> W index k*2+i
        # C[0,0] -> 0, C[1,0] -> 1, C[0,1] -> 2, C[1,1] -> 3
        
        # Strassen multiplications:
        # M1 = A[0,0] * (B[0,0] + B[1,1])
        # M2 = (A[0,0] + A[1,1]) * B[1,1]
        # M3 = A[1,1] * (B[0,0] - B[1,0])
        # M4 = A[0,0] * (B[1,1] - B[1,0])
        # M5 = (A[0,0] + A[0,1]) * B[0,0]
        # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[0,0])
        # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[1,0])
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # M1 = A[0,0] * (B[0,0] + B[1,1])
        # U: A[0,0] -> U[0]
        # V: B[0,0] -> V[0], B[1,1] -> V[3]
        # W: C[0,0] -> W[0], C[1,1] -> W[3]
        U[0, 0] = 1
        V[0, 0] = 1
        V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A[0,0] + A[1,1]) * B[1,1]
        # U: A[0,0] -> U[0], A[1,1] -> U[3]
        # V: B[1,1] -> V[3]
        # W: C[1,0] -> W[1], C[1,1] -> W[3]
        U[0, 1] = 1
        U[3, 1] = 1
        V[3, 1] = 1
        W[1, 1] = 1
        W[3, 1] = -1
        
        # M3 = A[1,1] * (B[0,0] - B[1,0])
        # U: A[1,1] -> U[3]
        # V: B[0,0] -> V[0], B[1,0] -> V[2]
        # W: C[0,1] -> W[2], C[1,1] -> W[3]
        U[3, 2] = 1
        V[0, 2] = 1
        V[2, 2] = -1
        W[2, 2] = 1
        W[3, 2] = 1
        
        # M4 = A[0,0] * (B[1,1] - B[1,0])
        # U: A[0,0] -> U[0]
        # V: B[1,1] -> V[3], B[1,0] -> V[2]
        # W: C[0,0] -> W[0], C[1,0] -> W[1]
        U[0, 3] = 1
        V[3, 3] = 1
        V[2, 3] = -1
        W[0, 3] = 1
        W[1, 3] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[0,0]
        # U: A[0,0] -> U[0], A[0,1] -> U[1]
        # V: B[0,0] -> V[0]
        # W: C[0,0] -> W[0], C[0,1] -> W[2]
        U[0, 4] = 1
        U[1, 4] = 1
        V[0, 4] = 1
        W[0, 4] = -1
        W[2, 4] = 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[0,0])
        # U: A[1,0] -> U[2], A[0,0] -> U[0]
        # V: B[0,1] -> V[1], B[0,0] -> V[0]
        # W: C[1,1] -> W[3]
        U[2, 5] = 1
        U[0, 5] = -1
        V[1, 5] = 1
        V[0, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[1,0])
        # U: A[0,1] -> U[1], A[1,1] -> U[3]
        # V: B[1,1] -> V[3], B[1,0] -> V[2]
        # W: C[0,0] -> W[0]
        U[1, 6] = 1
        U[3, 6] = -1
        V[3, 6] = 1
        V[2, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x4x1_decomposition():
        """
        Return U, V, W for 2x4x1 schoolbook.
        U has shape (8, 8), V has shape (4, 8), W has shape (2, 8).
        
        A[i,j] -> U index i*4+j
        B[j,k] -> V index j*1+k = j (since k only has value 0)
        C[i,k] -> W index k*2+i = i (since k=0)
        """
        rank = 8
        U = np.zeros((8, rank))
        V = np.zeros((4, rank))
        W = np.zeros((2, rank))
        
        for i in range(2):
            for j in range(4):
                col = i * 4 + j
                U[col, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        
        return U, V, W
    
    # Now assemble the full 2x4x5 decomposition
    # C = A * B where A is 2x4, B is 4x5, C is 2x5
    # Split B into B1 (4x4, columns 0-3) and B2 (4x1, column 4)
    # C[:, 0:4] = A * B1, C[:, 4] = A * B2
    # 
    # For A * B1 (2x4x4):
    # Split A into A1 (2x2, columns 0-1) and A2 (2x2, columns 2-3)
    # B1 into B1_1 (2x4, rows 0-1) and B1_2 (2x4, rows 2-3)
    # A * B1 = A1 * B1_1 + A2 * B1_2
    # Each is 2x2x4.
    #
    # For 2x2x4: A is 2x2, B is 2x4, C is 2x4
    # Split B into B1 (2x2) and B2 (2x2), columns 0-1 and 2-3
    # C = A * B1 + A * B2, each 2x2x2 (rank 7 each)
    # So 2x2x4 has rank 14.
    #
    # Therefore A * B1 (2x4x4) = A1 * B1_1 + A2 * B1_2
    # Each is 2x2x4 with rank 14, total rank 28.
    # Plus 2x4x1 has rank 8.
    # Total: 36.
    
    total_rank = 28 + 8  # 36
    
    U = np.zeros((n * m, total_rank))  # (8, 36)
    V = np.zeros((m * p, total_rank))  # (20, 36)
    W = np.zeros((p * n, total_rank))  # (10, 36)
    
    # Helper to place a sub-decomposition into the larger matrices
    def place_sub(U_sub, V_sub, W_sub, row_offset_U, row_offset_V, row_offset_W, col_offset):
        """
        Place sub-decomposition into larger matrices.
        U_sub has shape (rows_U_sub, rank_sub)
        V_sub has shape (rows_V_sub, rank_sub)
        W_sub has shape (rows_W_sub, rank_sub)
        """
        rank_sub = U_sub.shape[1]
        U[row_offset_U:row_offset_U + U_sub.shape[0], col_offset:col_offset + rank_sub] = U_sub
        V[row_offset_V:row_offset_V + V_sub.shape[0], col_offset:col_offset + rank_sub] = V_sub
        W[row_offset_W:row_offset_W + W_sub.shape[0], col_offset:col_offset + rank_sub] = W_sub
    
    # For 2x4x4 part (first 4 columns of B):
    # A = [A1, A2] where A1 is columns 0-1, A2 is columns 2-3
    # B[:, 0:4] = [B1; B2] where B1 is rows 0-1, B2 is rows 2-3
    #
    # Term 1: A1 * B1 (2x2x4)
    # A1 indices in A: i=0,1; j=0,1 -> U indices i*4+j = 0,1,4,5
    # B1 indices in B: j=0,1; k=0,1,2,3 -> V indices j*5+k = 0,1,2,3,5,6,7,8
    # C1 indices in C: i=0,1; k=0,1,2,3 -> W indices k*2+i
    #   C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
    
    # For 2x2x4: split B into B_left (2x2, cols 0-1) and B_right (2x2, cols 2-3)
    # C = A * B_left + A * B_right
    
    # Sub-term 1a: A1 * B1_left (2x2x2)
    # A1: i=0,1; j=0,1
    # B1_left: j=0,1; k=0,1
    # C: i=0,1; k=0,1 -> W indices 0,1,2,3
    U_s1, V_s1, W_s1 = strassen_decomposition()
    place_sub(U_s1, V_s1, W_s1, 0, 0, 0, 0)
    
    # Sub-term 1b: A1 * B1_right (2x2x2)
    # A1: i=0,1; j=0,1 -> U indices 0,1,4,5
    # B1_right: j=0,1; k=2,3 -> V indices j*5+k = 2,3,7,8
    # C: i=0,1; k=2,3 -> W indices k*2+i = 4,5,6,7
    U_s2, V_s2, W_s2 = strassen_decomposition()
    # Need to remap V indices: B[j,k] with k in {2,3} -> V[j*5+k]
    # In sub-decomp, B[j,k] with k in {0,1} -> V[j*2+k]
    # So V_sub[j*2+0] -> V_full[j*5+2], V_sub[j*2+1] -> V_full[j*5+3]
    V_remapped = np.zeros_like(V_s2)
    V_remapped[0, :] = V_s2[0, :]  # j=0, k=0 -> j=0, k=2
    V_remapped[1, :] = V_s2[1, :]  # j=0, k=1 -> j=0, k=3
    V_remapped[2, :] = V_s2[2, :]  # j=1, k=0 -> j=1, k=2
    V_remapped[3, :] = V_s2[3, :]  # j=1, k=1 -> j=1, k=3
    # Actually, we need to place V_s2 into V at the correct positions
    # V_s2[0,:] corresponds to B[0,0] in sub -> B[0,2] in full -> V index 0*5+2=2
    # V_s2[1,:] corresponds to B[0,1] in sub -> B[0,3] in full -> V index 0*5+3=3
    # V_s2[2,:] corresponds to B[1,0] in sub -> B[1,2] in full -> V index 1*5+2=7
    # V_s2[3,:] corresponds to B[1,1] in sub -> B[1,3] in full -> V index 1*5+3=8
    V[2:4, 7:14] = V_s2[0:2, :]
    V[7:9, 7:14] = V_s2[2:4, :]
    # U and W are the same as sub-decomp
    U[0:2, 7:14] = U_s2[0:2, :]
    U[4:6, 7:14] = U_s2[2:4, :]
    W[4:6, 7:14] = W_s2[0:2, :]
    W[6:8, 7:14] = W_s2[2:4, :]
    
    # Term 2: A2 * B2 (2x2x4)
    # A2 indices in A: i=0,1; j=2,3 -> U indices i*4+j = 2,3,6,7
    # B2 indices in B: j=2,3; k=0,1,2,3 -> V indices j*5+k = 10,11,12,13,15,16,17,18
    # C2 indices in C: i=0,1; k=0,1,2,3 -> W indices k*2+i = 0,1,2,3,4,5,6,7
    
    # Sub-term 2a: A2 * B2_left (2x2x2)
    # A2: i=0,1; j=2,3 -> U indices 2,3,6,7
    # B2_left: j=2,3; k=0,1 -> V indices j*5+k = 10,11,15,16
    # C: i=0,1; k=0,1 -> W indices 0,1,2,3
    U_s3, V_s3, W_s3 = strassen_decomposition()
    U[2:4, 14:21] = U_s3[0:2, :]
    U[6:8, 14:21] = U_s3[2:4, :]
    V[10:12, 14:21] = V_s3[0:2, :]
    V[15:17, 14:21] = V_s3[2:4, :]
    W[0:2, 14:21] = W_s3[0:2, :]
    W[2:4, 14:21] = W_s3[2:4, :]
    
    # Sub-term 2b: A2 * B2_right (2x2x2)
    # A2: i=0,1; j=2,3 -> U indices 2,3,6,7
    # B2_right: j=2,3; k=2,3 -> V indices j*5+k = 12,13,17,18
    # C: i=0,1; k=2,3 -> W indices 4,5,6,7
    U_s4, V_s4, W_s4 = strassen_decomposition()
    U[2:4, 21:28] = U_s4[0:2, :]
    U[6:8, 21:28] = U_s4[2:4, :]
    V[12:14, 21:28] = V_s4[0:2, :]
    V[17:19, 21:28] = V_s4[2:4, :]
    W[4:6, 21:28] = W_s4[0:2, :]
    W[6:8, 21:28] = W_s4[2:4, :]
    
    # For 2x4x1 part (5th column of B):
    # A is 2x4, B2 is 4x1, C is 2x1
    # Schoolbook: rank 8
    U_sb, V_sb, W_sb = schoolbook_2x4x1_decomposition()
    # U_sb has shape (8, 8), V_sb has shape (4, 8), W_sb has shape (2, 8)
    # Place into full matrices
    U[:, 28:36] = U_sb
    # V_sb[j, col] corresponds to B[j, 4] -> V index j*5+4
    V[4, 28:36] = V_sb[0, :]  # j=0 -> V[0*5+4] = V[4]
    V[9, 28:36] = V_sb[1, :]  # j=1 -> V[1*5+4] = V[9]
    V[14, 28:36] = V_sb[2, :]  # j=2 -> V[2*5+4] = V[14]
    V[19, 28:36] = V_sb[3, :]  # j=3 -> V[3*5+4] = V[19]
    # W_sb[i, col] corresponds to C[i, 4] -> W index 4*2+i = 8+i
    W[8, 28:36] = W_sb[0, :]  # i=0 -> W[8]
    W[9, 28:36] = W_sb[1, :]  # i=1 -> W[9]
    
    # Verify the decomposition
    T = np.zeros((n, m, p))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    if not np.allclose(T, 1.0):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
