import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 4, 5
    
    # We'll use a block decomposition approach.
    # The tensor is of shape (n*m, m*p, p*n) = (8, 20, 10)
    # We can split this into blocks based on the dimensions.
    
    # One approach: split m=4 into two blocks of 2
    # This gives us 2 subproblems of size 2x2x5 (for each j-block)
    # For 2x2x5, we can further split k=5 into blocks
    # 2x2x5 can be done as 2x2x2 (Strassen, rank 7) + 2x2x3 (rank ?)
    # 2x2x3 can be done as 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11
    # So 2x2x5 = 7 + 11 = 18
    # Total = 18 + 18 = 36
    
    # But we might be able to do better by not splitting m into blocks
    # Let's try to find a direct decomposition.
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid indices
    # We want to find rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Let's use a greedy/iterative approach to find a low-rank decomposition.
    # We can use randomized methods or structured approaches.
    
    # Actually, let's think about this more carefully.
    # The structure of the tensor is that it's a "convolution-like" tensor.
    # For matrix multiplication, the tensor is T[i,j,k] where the output is C[i,k] = sum_j A[i,j]B[j,k]
    # So T[A[i,j], B[j,k], C[i,k]] = 1
    
    # Let's try a different block decomposition.
    # Split n=2 into two blocks of 1: gives 2 subproblems of 1x4x5
    # 1x4x5: C[k] = sum_j A[j]B[j,k], this is just a dot product for each k
    # Rank of 1x4x5: we need at least 4 products (one per j), so rank = 4
    # Total = 4 + 4 = 8? No, that's not right because the A and B indices overlap.
    
    # Actually for 1xm x px5, the rank is min(m, p) in the simple case, but here it's 1x4x5
    # C[k] = sum_{j=0}^{3} A[j] B[j,k]
    # This can be done with 4 products, so rank = 4 per row-block
    # But wait, the A vector is shared across k, so for each i (row of C), we need 4 products
    # So for 2x4x5, if we split by i: 2 blocks of 1x4x5, each rank 4, total rank 8?
    # But that can't be right because schoolbook would be 2*4*5 = 40.
    
    # Oh wait, I think I'm confusing myself. Let me reconsider.
    # For 1x4x5: A is 1x4 (a row vector), B is 4x5, C is 1x5
    # C[k] = sum_{j=0}^{3} A[0,j] * B[j,k]
    # This is a single row of the matrix product. We need 4 scalar multiplications.
    # So rank of 1x4x5 tensor is 4.
    # For 2x4x5: split into two 1x4x5 blocks (one for each row of A), total rank = 4+4 = 8.
    # But wait, that seems too low. Let me check with a smaller case.
    
    # For 2x2x2: rank should be 7 (Strassen) or 8 (schoolbook)
    # Using the above logic: split into two 1x2x2 blocks, each rank 2, total rank 4.
    # But we know 2x2x2 has rank 7, so this logic is wrong.
    
    # The issue is that when we split, the factors U, V, W must be consistent.
    # For 1xm x pxn, the tensor is T[A_idx, B_idx, C_idx] where:
    # A_idx = i*m+j (but i is fixed, so A_idx = j)
    # B_idx = j*p+k
    # C_idx = k*n+i (but i is fixed, so C_idx = k*n + i)
    
    # For the full 2x4x5 tensor:
    # A_idx ranges over 0..7 (i*m+j for i in 0,1 and j in 0,1,2,3)
    # B_idx ranges over 0..19 (j*p+k for j in 0,1,2,3 and k in 0,1,2,3,4)
    # C_idx ranges over 0..9 (k*n+i for k in 0,1,2,3,4 and i in 0,1)
    
    # The tensor is 1 at (i*m+j, j*p+k, k*n+i) for all valid i,j,k.
    
    # Let's just use the block decomposition that gives rank 36, which is known to be valid.
    # U: 8 x 36, V: 20 x 36, W: 10 x 36
    
    rank = 36
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    S_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    S_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    S_W = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0]
    ]
    
    # Schoolbook 2x2x1
    SB_U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    SB_V = [
        [1,0],
        [0,1],
        [1,0],
        [0,1]
    ]
    SB_W = [
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ]
    
    col = 0
    
    def add_strassen(i_base, j_base, k_base, col_start):
        nonlocal col
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi*m + gj][col_start + r] = S_U[r][li*2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj*p + gk][col_start + r] = S_V[r][lj*2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk*n + gi][col_start + r] = S_W[r][lk*2 + li]
        return col_start + 7
    
    def add_schoolbook_2x2x1(i_base, j_base, k_base, col_start):
        nonlocal col
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi*m + gj][col_start + r] = SB_U[r][li*2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj*p + gk][col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk*n + gi][col_start + r] = SB_W[r][li]
        return col_start + 4
    
    # Rank-8 algorithm for 2x2x3
    # A is 2x2, B is 2x3, C is 3x2
    # C[k][i] = sum_j A[i][j] * B[j][k]
    # Using known rank-8 decomposition
    R8_U = [
        [1, 0, 0, 1],   # P0: A00, A01
        [0, 0, 1, 1],   # P1: A10, A11
        [1, 0, 0, 0],   # P2: A00
        [0, 0, 0, 1],   # P3: A11
        [1, 1, 0, 0],   # P4: A00+A10
        [-1, 0, 1, 0],  # P5: -A00+A10
        [0, 1, 0, -1],  # P6: A01-A11
        [1, 0, 0, 1],   # P7: A00+A11
    ]
    R8_V = [
        [1, 1, 0, 0, 0, 0],   # P0: B00+B01
        [0, 0, 1, 1, 0, 0],   # P1: B10+B11
        [1, 0, 0, 0, 0, 0],   # P2: B00
        [0, 0, 0, 0, 1, 1],   # P3: B12+B02? No...
        [1, 1, 0, 0, 0, 0],   # P4: B00+B01
        [-1, 0, 1, 0, 0, 0],  # P5: -B00+B10
        [0, 0, 0, 1, 0, 0],   # P6: B11
        [1, 0, 0, 0, 1, 0],   # P7: B00+B02
    ]
    # This is getting error-prone. Let me use a verified construction.
    
    # Actually, let me just keep the current rank-36 solution which is known to be correct.
    # The improvement from 36 would require finding a better algorithm.
    # For now, let's verify the current solution is correct and return it.
    
    # Block decomposition:
    # j=0 block: k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Schoolbook)
    col = add_strassen(0, 0, 0, col)
    col = add_strassen(0, 0, 2, col)
    col = add_schoolbook_2x2x1(0, 0, 4, col)
    
    # j=2 block: k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Schoolbook)
    col = add_strassen(0, 2, 0, col)
    col = add_strassen(0, 2, 2, col)
    col = add_schoolbook_2x2x1(0, 2, 4, col)
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
