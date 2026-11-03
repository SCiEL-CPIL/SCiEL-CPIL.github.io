import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # For 2x4 times 4x6, we can use a tensor decomposition approach.
    # The schoolbook approach has rank 48 (2*4*6/2... no, it's 48 for the full tensor).
    # Actually, the standard multiplication tensor for C = A*B where A is n x m, B is m x p
    # has rank n*m*p for the trivial decomposition.
    # We can use Strassen-like or other fast multiplication algorithms.
    # 
    # Let's try a direct approach: 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can decompose this into tensor rank.
    #
    # A simpler approach: use the fact that we can split the 4x6 matrix into blocks
    # and use Strassen on 2x2 blocks.
    #
    # Actually, let's just use the schoolbook algorithm which gives rank 48.
    # But we can do better. Let's use a rank-36 algorithm based on
    # splitting the 4 intermediate indices into two groups of 2.
    #
    # C = A * B where A is 2x4, B is 4x6
    # Split B into B1 (4x3) and B2 (4x3)
    # C1 = A * B1 (2x3), C2 = A * B2 (2x3)
    #
    # For 2x4 * 4x3: 
    # We can use a rank-15 algorithm (from fast rectangular matrix multiplication)
    # or just use schoolbook: 2*4*3 = 24
    #
    # Let's try a different approach: use the Winograd-Friedman style or just
    # apply Strassen by padding to 2x2 blocks.
    #
    # Actually, the simplest improvement: 
    # The tensor T_{i,j,k} = A[i,j] * B[j,k] for C[i,k] = sum_j T_{i,j,k}
    # has a natural decomposition of rank n*m*p = 2*4*6 = 48 using unit tensors.
    #
    # We can reduce this by noting that we only need sums over j.
    # Using Strassen: pad A to 2x4 (already), B to 4x6 (already).
    # Split the 4 columns of A / 4 rows of B into two 2-element blocks.
    # 
    # Let A = [A1 | A2] where A1, A2 are 2x2
    # Let B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # 
    # Each 2x2 * 2x6 product can be done with rank 7*3 = 21 using Strassen
    # (split B into three 2x2 blocks, apply Strassen to each)
    # Total: 21 + 21 = 42
    #
    # This is what the current program does. Let's verify it's correct.
    #
    # The current program already implements this. Let's keep it but make
    # sure it's exactly correct.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7
    
    strassen_2x2 = [
        # (u, v, w) where u,v are 2x2 sub-matrices, w is 2x2 output
        # Indexing: 0=A11, 1=A12, 2=A21, 3=A22 for u
        #             0=B11, 1=B12, 2=B21, 3=B22 for v
        #             0=C11, 1=C12, 2=C21, 3=C22 for w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 1, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [1, 0, -1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    col = 0
    # Block 0: A_left (cols 0-1) * B_top (rows 0-1) -> contributes to C with cols 0-5
    # A_left is 2x2, B_top is 2x6
    # Split B_top into B_top1 (2x2, cols 0-1), B_top2 (2x2, cols 2-3), B_top3 (2x2, cols 4-5)
    for b_col_offset in [0, 2, 4]:
        if not _time_ok(deadline):
            break
        for u_vec, v_vec, w_vec in strassen_2x2:
            if not _time_ok(deadline):
                break
            # U indices: i*m + j for A[i,j] where i in 0-1, j in 0-1 (A_left)
            # V indices: j*p + k for B[j,k] where j in 0-1 (B_top), k in b_col_offset to b_col_offset+1
            # W indices: k*n + i for C[i,k]
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx][col] += u_vec[j * 2 + i] if False else 0
                    # Actually u_vec is indexed as [A11, A12, A21, A22]
                    # So u_vec[0]=A11, u_vec[1]=A12, u_vec[2]=A21, u_vec[3]=A22
                    # For A[i,j]: i=0,j=0 -> idx 0; i=0,j=1 -> idx 1; i=1,j=0 -> idx 2; i=1,j=1 -> idx 3
            # Let me redo this more carefully
            # u_vec[k] where k=0,1,2,3 corresponds to A[0,0], A[0,1], A[1,0], A[1,1]
            # v_vec[k] corresponds to B[0,0], B[0,1], B[1,0], B[1,1] (local)
            # w_vec[k] corresponds to C[0,0], C[0,1], C[1,0], C[1,1] (local)
            
            # U: A[i,j] for i in 0-1, j in 0-1 (global j = j_local since A_left is cols 0-1)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] += u_vec[j + i * 2]
            
            # V: B[j,k] for j in 0-1 (global), k in b_col_offset to b_col_offset+1
            for j in range(2):
                for k in range(2):
                    V[j * p + b_col_offset + k][col] += v_vec[k + j * 2]
            
            # W: C[i,k] for i in 0-1, k in b_col_offset to b_col_offset+1
            # W index: k * n + i
            for i in range(2):
                for k in range(2):
                    W[(b_col_offset + k) * n + i][col] += w_vec[k + i * 2]
            
            col += 1
    
    # Block 1: A_right (cols 2-3) * B_bot (rows 2-3) -> contributes to C with cols 0-5
    for b_col_offset in [0, 2, 4]:
        if not _time_ok(deadline):
            break
        for u_vec, v_vec, w_vec in strassen_2x2:
            if not _time_ok(deadline):
                break
            # U: A[i,j] for i in 0-1, j in 2-3 (global)
            for i in range(2):
                for j in range(2):
                    U[i * m + 2 + j][col] += u_vec[j + i * 2]
            
            # V: B[j,k] for j in 2-3 (global), k in b_col_offset to b_col_offset+1
            for j in range(2):
                for k in range(2):
                    V[(2 + j) * p + b_col_offset + k][col] += v_vec[k + j * 2]
            
            # W: C[i,k] for i in 0-1, k in b_col_offset to b_col_offset+1
            for i in range(2):
                for k in range(2):
                    W[(b_col_offset + k) * n + i][col] += w_vec[k + i * 2]
            
            col += 1
    
    # col should be 42
    if col != rank:
        # Pad if necessary
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
