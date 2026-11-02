"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Build the schoolbook decomposition as a baseline
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try block-Strassen approach:
    # Split 2x4x5 into blocks.
    # A is 2x4, B is 4x5, C is 2x5.
    # We can view this as:
    #   A = [A11 A12] where A11, A12 are 2x2
    #   B = [B11; B21] where B11, B21 are 2x5
    #   C = A11*B11 + A12*B21
    #
    # Each 2x2x5 product can be done with Strassen-like tricks.
    # Actually, let's think more carefully.
    #
    # Standard approach: treat as two 2x2x5 multiplications.
    # For 2x2x5: we can split the 5 columns.
    # 2x2x2 uses Strassen rank 7, 2x2x3 uses... let's think.
    #
    # Alternative: 2x4x5 = sum of two 2x2x5 products.
    # 2x2x5: split into 2x2x2 + 2x2x3.
    # 2x2x2: Strassen rank 7
    # 2x2x3: can use 3*2*2=12 schoolbook, or maybe better?
    #   Actually 2x2x3: we can use rank 6? Let's check.
    #   The tensor for 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0,1; j in 0,1; k in 0,1,2
    #   Schoolbook: 2*2*3 = 12.
    #   Can we do better? Known: 2x2x3 has rank 6? No, I think it's still 12 or maybe 8?
    #   Actually, for 2x2xn, the rank is known. For n=3, it might be 6? Let me think...
    #   Actually no. Let me just use a direct construction.
    #
    # Let me try a different block decomposition.
    # 2x4x5: 
    #   Split A into two 2x2 blocks: A = [A_L | A_R] (2x2 each)
    #   Split B into two 2x5 blocks: B = [B_T; B_B] (2x5 each)
    #   C = A_L * B_T + A_R * B_B
    #
    # Each 2x2x5 product: 
    #   Split B_T (2x5) into B_T1 (2x2) + B_T2 (2x2) + B_T3 (2x1)
    #   A_L * B_T = A_L*B_T1 + A_L*B_T2 + A_L*B_T3
    #   2x2x2: Strassen rank 7
    #   2x2x1: rank 2 (just two dot products)
    #   So 2x2x5: 7 + 7 + 2 = 16
    #   Total: 16 + 16 = 32
    #
    # Let me verify this construction works.

    def strassen_2x2x2(A_rows, B_cols, offset_U, offset_V, offset_W, U_mat, V_mat, W_mat):
        """
        Add Strassen 2x2x2 decomposition.
        A_rows: list of 2 row indices in U (for the 2x2 A block)
        B_cols: list of 2 row indices in V (for the 2x2 B block)  
        W_cols: list of 2*2=4 col indices in W (for the 2x2 C block)
        Returns number of columns used.
        """
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        a = A_rows  # [a0, a1] for A11, A22 conceptually... 
        # Actually I need to be more careful with indices.
        # Let me define: A is 2x2 with entries A[i,j] stored in U at row (base_U + i*2 + j)
        # B is 2x2 with entries B[j,k] stored in V at row (base_V + j*2 + k)
        # C is 2x2 with entries C[i,k] stored in W at row (base_W + k*2 + i)
        
        # Let me restructure: pass base indices
        pass

    # Let me just build the full decomposition explicitly.
    # 2x4x5 with block Strassen:
    # A (2x4): rows 0,1; cols 0,1,2,3
    # B (4x5): rows 0,1,2,3; cols 0,1,2,3,4  
    # C (2x5): rows 0,1; cols 0,1,2,3,4
    #
    # Split: A = [A_L | A_R], A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B = [B_T; B_B], B_T = B[0:2, :], B_B = B[2:4, :]
    # C = A_L @ B_T + A_R @ B_B
    #
    # For A_L @ B_T (2x2 @ 2x5):
    #   Split B_T into B_T[:, 0:2], B_T[:, 2:4], B_T[:, 4:5]
    #   = 2x2x2 + 2x2x2 + 2x2x1
    #   Strassen(7) + Strassen(7) + schoolbook(2) = 16
    #
    # Similarly for A_R @ B_B: 16
    # Total: 32

    # Let me build this explicitly.
    def build_block_strassen():
        """Build 2x4x5 decomposition using block Strassen, rank 32."""
        n, m, p = 2, 4, 5
        U = np.zeros((n * m, 32))  # 8 x 32
        V = np.zeros((m * p, 32))  # 20 x 32
        W = np.zeros((p * n, 32))  # 10 x 32
        
        col = 0
        
        def add_strassen_2x2x2(a_base_u, b_base_v, c_base_w):
            """
            Add Strassen 2x2x2 at given base indices.
            a_base_u: base row in U for A (2x2 block), A[i,j] at U[a_base_u + i*2 + j]
            b_base_v: base row in V for B (2x2 block), B[j,k] at V[b_base_v + j*2 + k]
            c_base_w: base row in W for C (2x2 block), C[i,k] at W[c_base_w + k*2 + i]
            Returns 7 columns used.
            """
            global col
            # M1 = (A11+A22)(B11+B22)
            # A11 at U[a_base_u+0], A22 at U[a_base_u+3]
            # B11 at V[b_base_v+0], B22 at V[b_base_v+3]
            U[a_base_u + 0, col] = 1
            U[a_base_u + 3, col] = 1
            V[b_base_v + 0, col] = 1
            V[b_base_v + 3, col] = 1
            # C contribution: C11 += M1, C22 += M1
            # C[i,k] at W[c_base_w + k*2 + i]
            # C11: i=0,k=0 -> W[c_base_w+0]
            # C22: i=1,k=1 -> W[c_base_w+3]
            W[c_base_w + 0, col] = 1
            W[c_base_w + 3, col] = 1
            col += 1
            
            # M2 = (A21+A22)B11
            # A21 at U[a_base_u+2], A22 at U[a_base_u+3]
            # B11 at V[b_base_v+0]
            U[a_base_u + 2, col] = 1
            U[a_base_u + 3, col] = 1
            V[b_base_v + 0, col] = 1
            # C21 += M2: i=1,k=0 -> W[c_base_w+1]
            W[c_base_w + 1, col] = 1
            col += 1
            
            # M3 = A11(B21-B22)
            # A11 at U[a_base_u+0]
            # B21 at V[b_base_v+2], B22 at V[b_base_v+3]
            U[a_base_u + 0, col] = 1
            V[b_base_v + 2, col] = 1
            V[b_base_v + 3, col] = -1
            # C12 += M3: i=0,k=1 -> W[c_base_w+2]
            # C22 += M3: i=1,k=1 -> W[c_base_w+3]
            W[c_base_w + 2, col] = 1
            W[c_base_w + 3, col] = 1
            col += 1
            
            # M4 = A22(B12-B11)
            # A22 at U[a_base_u+3]
            # B12 at V[b_base_v+1], B11 at V[b_base_v+0]
            U[a_base_u + 3, col] = 1
            V[b_base_v + 1, col] = 1
            V[b_base_v + 0, col] = -1
            # C11 += M4: W[c_base_w+0]
            # C21 += M4: W[c_base_w+1]
            W[c_base_w + 0, col] = 1
            W[c_base_w + 1, col] = 1
            col += 1
            
            # M5 = (A11+A12)B22
            # A11 at U[a_base_u+0], A12 at U[a_base_u+1]
            # B22 at V[b_base_v+3]
            U[a_base_u + 0, col] = 1
            U[a_base_u + 1, col] = 1
            V[b_base_v + 3, col] = 1
            # C11 -= M5: W[c_base_w+0]
            # C12 += M5: W[c_base_w+2]
            W[c_base_w + 0, col] = -1
            W[c_base_w + 2, col] = 1
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            # A21 at U[a_base_u+2], A11 at U[a_base_u+0]
            # B11 at V[b_base_v+0], B12 at V[b_base_v+1]
            U[a_base_u + 2, col] = 1
            U[a_base_u + 0, col] = -1
            V[b_base_v + 0, col] = 1
            V[b_base_v + 1, col] = 1
            # C22 += M6: W[c_base_w+3]
            W[c_base_w + 3, col] = 1
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            # A12 at U[a_base_u+1], A22 at U[a_base_u+3]
            # B21 at V[b_base_v+2], B22 at V[b_base_v+3]
            U[a_base_u + 1, col] = 1
            U[a_base_u + 3, col] = -1
            V[b_base_v + 2, col] = 1
            V[b_base_v + 3, col] = 1
            # C11 += M7: W[c_base_w+0]
            W[c_base_w + 0, col] = 1
            col += 1
            
            return 7
        
        def add_schoolbook_2x2x1(a_base_u, b_base_v, c_base_w):
            """
            Add schoolbook 2x2x1 at given base indices.
            A is 2x2, B is 2x1, C is 2x1.
            A[i,j] at U[a_base_u + i*2 + j]
            B[j,0] at V[b_base_v + j]
            C[i,0] at W[c_base_w + i]
            Returns 2 columns used.
            """
            global col
            for j in range(2):
                U[a_base_u + 0*2 + j, col] = 1
                U[a_base_u + 1*2 + j, col] = 1
                V[b_base_v + j, col] = 1
                W[c_base_w + 0, col] = 1
                W[c_base_w + 1, col] = 1
                col += 1
            return 2
        
        # A_L @ B_T: A_L is A[:, 0:2] (2x2), B_T is B[0:2, :] (2x5)
        # A_L[i,j] -> U[i*4 + j] for i in 0,1; j in 0,1
        # B_T[j,k] -> V[j*5 + k] for j in 0,1; k in 0,4
        # C[i,k] -> W[k*2 + i] for i in 0,1; k in 0,4
        
        # Split B_T into columns 0-1, 2-3, 4
        # B_T[:, 0:2]: B[j,k] at V[j*5+k] for k=0,1 -> V[0],V[1],V[5],V[6]
        #   This is a 2x2 block. Base for V: row 0 for j=0, row 5 for j=1
        #   But my add_strassen expects B[j,k] at V[b_base_v + j*2 + k]
        #   So I need V[b_base_v + 0*2 + 0] = V[b_base_v] = V[0] -> b_base_v=0, k=0
        #   V[b_base_v + 0*2 + 1] = V[b_base_v+1] = V[1] -> k=1
        #   V[b_base_v + 1*2 + 0] = V[b_base_v+2] = V[5] -> b_base_v+2=5 -> b_base_v=3? No...
        #   
        # The issue is that B_T[j,k] is at V[j*5+k], not contiguous in a 2x2 block.
        # I need to handle non-contiguous indexing. Let me modify the approach.
        
        # Actually, let me just do it more carefully with explicit index mapping.
        
        # Let me restart with a cleaner approach.
        U = np.zeros((n * m, 32))
        V = np.zeros((m * p, 32))
        W = np.zeros((p * n, 32))
        col = 0
        
        # Helper: add a Strassen 2x2x2 with arbitrary index mappings
        def strassen_2x2x2(a_idx, b_idx, c_idx):
            """
            a_idx: dict mapping (i,j) -> U row index for A[i,j]
            b_idx: dict mapping (j,k) -> V row index for B[j,k]
            c_idx: dict mapping (i,k) -> W row index for C[i,k]
            """
            nonlocal col
            # M1 = (A11+A22)(B11+B22)
            U[a_idx[(0,0)], col] += 1
            U[a_idx[(1,1)], col] += 1
            V[b_idx[(0,0)], col] += 1
            V[b_idx[(1,1)], col] += 1
            W[c_idx[(0,0)], col] += 1  # C11
            W[c_idx[(1,1)], col] += 1  # C22
            col += 1
            
            # M2 = (A21+A22)B11
            U[a_idx[(1,0)], col] += 1
            U[a_idx[(1,1)], col] += 1
            V[b_idx[(0,0)], col] += 1
            W[c_idx[(1,0)], col] += 1  # C21
            col += 1
            
            # M3 = A11(B21-B22)
            U[a_idx[(0,0)], col] += 1
            V[b_idx[(1,0)], col] += 1
            V[b_idx[(1,1)], col] += -1
            W[c_idx[(0,1)], col] += 1  # C12
            W[c_idx[(1,1)], col] += 1  # C22
            col += 1
            
            # M4 = A22(B12-B11)
            U[a_idx[(1,1)], col] += 1
            V[b_idx[(0,1)], col] += 1
            V[b_idx[(0,0)], col] += -1
            W[c_idx[(0,0)], col] += 1  # C11
            W[c_idx[(1,0)], col] += 1  # C21
            col += 1
            
            # M5 = (A11+A12)B22
            U[a_idx[(0,0)], col] += 1
            U[a_idx[(0,1)], col] += 1
            V[b_idx[(1,1)], col] += 1
            W[c_idx[(0,0)], col] += -1  # C11
            W[c_idx[(0,1)], col] += 1   # C12
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[a_idx[(1,0)], col] += 1
            U[a_idx[(0,0)], col] += -1
            V[b_idx[(0,0)], col] += 1
            V[b_idx[(0,1)], col] += 1
            W[c_idx[(1,1)], col] += 1  # C22
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[a_idx[(0,1)], col] += 1
            U[a_idx[(1,1)], col] += -1
            V[b_idx[(1,0)], col] += 1
            V[b_idx[(1,1)], col] += 1
            W[c_idx[(0,0)], col] += 1  # C11
            col += 1
        
        def schoolbook_2x2x1(a_idx, b_idx, c_idx):
            """
            a_idx: dict mapping (i,j) -> U row for A[i,j], i in 0,1; j in 0,1
            b_idx: dict mapping (j,) -> V row for B[j,0], j in 0,1
            c_idx: dict mapping (i,) -> W row for C[i,0], i in 0,1
            """
            nonlocal col
            for j in range(2):
                U[a_idx[(0,j)], col] += 1
                U[a_idx[(1,j)], col] += 1
                V[b_idx[(j,)], col] += 1
                W[c_idx[(0,)], col] += 1
                W[c_idx[(1,)], col] += 1
                col += 1
        
        # A[i,j] is at U[i*m + j] = U[i*4 + j]
        # B[j,k] is at V[j*p + k] = V[j*5 + k]
        # C[i,k] is at W[k*n + i] = W[k*2 + i]
        
        # A_L @ B_T:
        # A_L[i,j] = A[i, j] for j in 0,1 -> U[i*4 + j]
        # B_T[j,k] = B[j, k] for j in 0,1; k in 0,4 -> V[j*5 + k]
        # C[i,k] = W[k*2 + i]
        
        aL_idx = {(i,j): i*4 + j for i in range(2) for j in range(2)}
        
        # B_T columns 0,1:
        bT_01_idx = {(j,k): j*5 + k for j in range(2) for k in range(2)}
        c_01_idx = {(i,k): k*2 + i for i in range(2) for k in range(2)}
        strassen_2x2x2(aL_idx, bT_01_idx, c_01_idx)
        
        # B_T columns 2,3:
        bT_23_idx = {(j,k): j*5 + (k+2) for j in range(2) for k in range(2)}
        c_23_idx = {(i,k): (k+2)*2 + i for i in range(2) for k in range(2)}
        strassen_2x2x2(aL_idx, bT_23_idx, c_23_idx)
        
        # B_T column 4:
        bT_4_idx = {(j,): j*5 + 4 for j in range(2)}
        c_4_idx = {(i,): 4*2 + i for i in range(2)}
        schoolbook_2x2x1(aL_idx, bT_4_idx, c_4_idx)
        
        # A_R @ B_B:
        # A_R[i,j] = A[i, j+2] for j in 0,1 -> U[i*4 + j + 2]
        # B_B[j,k] = B[j+2, k] for j in 0,1; k in 0,4 -> V[(j+2)*5 + k]
        # C[i,k] = W[k*2 + i]
        
        aR_idx = {(i,j): i*4 + j + 2 for i in range(2) for j in range(2)}
        
        # B_B columns 0,1:
        bB_01_idx = {(j,k): (j+2)*5 + k for j in range(2) for k in range(2)}
        c_01_idx2 = {(i,k): k*2 + i for i in range(2) for k in range(2)}
        strassen_2x2x2(aR_idx, bB_01_idx, c_01_idx2)
        
        # B_B columns 2,3:
        bB_23_idx = {(j,k): (j+2)*5 + (k+2) for j in range(2) for k in range(2)}
        c_23_idx2 = {(i,k): (k+2)*2 + i for i in range(2) for k in range(2)}
        strassen_2x2x2(aR_idx, bB_23_idx, c_23_idx2)
        
        # B_B column 4:
        bB_4_idx = {(j,): (j+2)*5 + 4 for j in range(2)}
        c_4_idx2 = {(i,): 4*2 + i for i in range(2)}
        schoolbook_2x2x1(aR_idx, bB_4_idx, c_4_idx2)
        
        return U, V, W, col
    
    # Build the block-Strassen decomposition
    U_bs, V_bs, W_bs, rank_bs = build_block_strassen()
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat, rank):
        n, m, p = DIMENSIONS
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is a valid triple, else 0
        # Actually the tensor T[i,j,k] = delta(i,j,k) meaning T[i,j,k]=1 for all i,j,k
        # The decomposition satisfies: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # T[i,j,k] should be 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U_bs, V_bs, W_bs, rank_bs):
        best_rank = rank_bs
        best_U, best_V, best_W = U_bs.copy(), V_bs.copy(), W_bs.copy()
    else:
        print("Block Strassen verification failed, using schoolbook")
    
    # Try to find even better decompositions via random search
    # Try rank 30, 31, etc.
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            # Random initialization
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Use ALS (Alternating Least Squares) to optimize
            # Minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            for iteration in range(100):
                if time.time() > deadline - 20:
                    break
                # Update U
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            # Gradient w.r.t. U[i*m+j, c]
                            grad = 0.0
                            for k in range(p):
                                w_val = W_rand[k*n+i, c]
                                v_val = V_rand[j*p+k, c]
                                # Current value of sum
                                current = 0.0
                                for c2 in range(target_rank):
                                    current += U_rand[i*m+j, c2] * V_rand[j*p+k, c2] * W_rand[k*n+i, c2]
                                grad += (current - 1.0) * v_val * w_val
                            U_rand[i*m+j, c] -= 0.01 * grad
                
                # Update V
                for c in range(target_rank):
                    for j in range(m):
                        for k in range(p):
                            grad = 0.0
                            for i in range(n):
                                u_val = U_rand[i*m+j, c]
                                w_val = W_rand[k*n+i, c]
                                current = 0.0
                                for c2 in range(target_rank):
                                    current += U_rand[i*m+j, c2] * V_rand[j*p+k, c2] * W_rand[k*n+i, c2]
                                grad += (current - 1.0) * u_val * w_val
                            V_rand[j*p+k, c] -= 0.01 * grad
                
                # Update W
                for c in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            grad = 0.0
                            for j in range(m):
                                u_val = U_rand[i*m+j, c]
                                v_val = V_rand[j*p+k, c]
                                current = 0.0
                                for c2 in range(target_rank):
                                    current += U_rand[i*m+j, c2] * V_rand[j*p+k, c2] * W_rand[k*n+i, c2]
                                grad += (current - 1.0) * u_val * v_val
                            W_rand[k*n+i, c] -= 0.01 * grad
            
            # Check if we found a valid decomposition
            if verify(U_rand, V_rand, W_rand, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                found = True
                break
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
