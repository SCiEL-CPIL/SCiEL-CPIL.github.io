import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 decomposition
    # a_r[i*2+j], b_r[j*2+k], c_r[k*2+i]
    strassen_a = [
        [1,0,0,1],  # M1: A00+A11
        [0,0,1,1],  # M2: A10+A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: A00+A01
        [-1,0,1,0], # M6: A10-A00
        [0,1,0,-1], # M7: A01-A11
    ]
    strassen_b = [
        [1,0,0,1],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,1,0,-1], # M3: B01-B11
        [0,0,1,-1], # M4: B10-B00
        [0,0,0,1],  # M5: B11
        [1,1,0,0],  # M6: B00+B01
        [0,0,1,1],  # M7: B10+B11
    ]
    strassen_c = [
        [1,0,-1,1,0,0,1],   # C00 = M1+M4-M5+M7
        [0,0,1,0,1,0,0],    # C01 = M3+M5
        [0,1,0,1,0,0,0],    # C10 = M2+M4
        [1,-1,1,0,0,1,0],   # C11 = M1-M2+M3+M6
    ]
    
    # Verify Strassen 2x2x2
    U2 = np.array(strassen_a).T  # shape (4, 7)
    V2 = np.array(strassen_b).T  # shape (4, 7)
    W2 = np.array(strassen_c)    # shape (4, 7)
    
    # Check: sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] == 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                if s != 1:
                    raise ValueError(f"Strassen 2x2x2 verification failed at ({i},{j},{k}): {s}")
    
    # Now build 4x4 @ 4x4 decomposition using recursive Strassen
    # Outer Strassen: 7 terms, each is a 2x2 matmul (inner Strassen, rank 7)
    # Total rank: 49
    
    # For 4x4 @ 4x4:
    # A is 4x4, split into 2x2 blocks:
    # A[0:2, 0:2] = A11, A[0:2, 2:4] = A12
    # A[2:4, 0:2] = A21, A[2:4, 2:4] = A22
    
    # B is 4x4, split into 2x2 blocks:
    # B[0:2, 0:2] = B11, B[0:2, 2:4] = B12
    # B[2:4, 0:2] = B21, B[2:4, 2:4] = B22
    
    # Outer Strassen multipliers (for 2x2 block-level):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each outer multiplier r, we need to:
    # 1. Form the linear combination of A blocks
    # 2. Form the linear combination of B blocks
    # 3. Multiply the resulting 2x2 matrices using inner Strassen
    
    # The outer Strassen coefficients for A blocks:
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    outer_a = [
        [1,0,0,1],  # M1: A11+A22
        [0,0,1,1],  # M2: A21+A22
        [1,0,0,0],  # M3: A11
        [0,0,0,1],  # M4: A22
        [1,1,0,0],  # M5: A11+A12
        [-1,0,1,0], # M6: A21-A11
        [0,1,0,-1], # M7: A12-A22
    ]
    outer_b = [
        [1,0,0,1],  # M1: B11+B22
        [1,0,0,0],  # M2: B11
        [0,1,0,-1], # M3: B12-B22
        [0,0,1,-1], # M4: B21-B11
        [0,0,0,1],  # M5: B22
        [1,1,0,0],  # M6: B11+B12
        [0,0,1,1],  # M7: B21+B22
    ]
    outer_c = [
        [1,0,-1,1,0,0,1],   # C11 = M1+M4-M5+M7
        [0,0,1,0,1,0,0],    # C12 = M3+M5
        [0,1,0,1,0,0,0],    # C21 = M2+M4
        [1,-1,1,0,0,1,0],   # C22 = M1-M2+M3+M6
    ]
    
    # Now, for the full 4x4 @ 4x4 tensor decomposition:
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    # (since n=m=4, so n*m=16, m*p=16, p*n=16 for the 4x4 part)
    
    # For each outer term r (0..6) and inner term s (0..6):
    # The combined term index is r*7 + s
    
    # U[i*4+j, r*7+s] = outer_a[r][block(i,j)] * strassen_a[s][local(i,j)]
    # where block(i,j) = (i//2)*2 + (j//2) -- maps to block index
    # and local(i,j) = (i%2)*2 + (j%2) -- maps to local index within block
    
    # Wait, I need to be careful. The A matrix is 4x4 with entries A[i,j] for i,j in 0..3.
    # Block (bi, bj) contains entries A[2*bi+li, 2*bj+lj] for li,lj in 0..1.
    # So for a given (i,j): block = (i//2, j//2), local = (i%2, j%2)
    # Block index = (i//2)*2 + (j//2)
    # Local index = (i%2)*2 + (j%2)
    
    # Similarly for B[j,k]: block = (j//2, k//2), local = (j%2, k%2)
    
    # For W[k*4+i]: block = (k//2, i//2), local = (k%2, i%2)
    
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for i in range(4):
                for j in range(4):
                    block_i = i // 2
                    block_j = j // 2
                    local_i = i % 2
                    local_j = j % 2
                    block_idx = block_i * 2 + block_j
                    local_idx = local_i * 2 + local_j
                    U44[i*4+j, col] = outer_a[r][block_idx] * strassen_a[s][local_idx]
            
            for j in range(4):
                for k in range(4):
                    block_j = j // 2
                    block_k = k // 2
                    local_j = j % 2
                    local_k = k % 2
                    block_idx = block_j * 2 + block_k
                    local_idx = local_j * 2 + local_k
                    V44[j*4+k, col] = outer_b[r][block_idx] * strassen_b[s][local_idx]
            
            for k in range(4):
                for i in range(4):
                    block_k = k // 2
                    block_i = i // 2
                    local_k = k % 2
                    local_i = i % 2
                    block_idx = block_k * 2 + block_i
                    local_idx = local_k * 2 + local_i
                    W44[k*4+i, col] = outer_c[block_idx][r] * strassen_c[local_idx][s]
    
    # Verify 4x4 @ 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(49):
                    s += U44[i*4+j, r] * V44[j*4+k, r] * W44[k*4+i, r]
                if abs(s - 1) > 1e-10:
                    raise ValueError(f"4x4@4x4 verification failed at ({i},{j},{k}): {s}")
    
    # Now for 4x4 @ 4x2:
    # A is 4x4, B is 4x2, C is 4x2
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is 4x2: B_top (2x2) and B_bot (2x2)
    # C is 4x2: C_top (2x2) and C_bot (2x2)
    # C_top = A11*B_top + A12*B_bot
    # C_bot = A21*B_top + A22*B_bot
    
    # For C_top = A11*B_top + A12*B_bot:
    # This is a sum of two 2x2 matmuls.
    # We can use a Strassen-like approach or just schoolbook for each.
    # Schoolbook: rank 4+4 = 8 for this 2x2 result
    # But we might do better.
    
    # Actually, let me think about this differently.
    # 4x4 @ 4x2:
    # A[i,j] for i,j in 0..3
    # B[j,k] for j in 0..3, k in 0..1
    # C[i,k] for i in 0..3, k in 0..1
    
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B1], [B2]] where B1, B2 are 2x2
    # C = [[C1], [C2]] where C1, C2 are 2x2
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    
    # For C1 = A11*B1 + A12*B2:
    # We can view this as a "block" Strassen-like computation.
    # Let X = A11, Y = A12, Z = B1, W = B2
    # C1 = X*Z + Y*W
    # This is a bilinear map from (X,Y,Z,W) to C1.
    # The rank of this tensor...
    
    # One approach: use Strassen on a 2x2 "meta" problem.
    # Consider [[X, Y], [0, 0]] * [[Z, 0], [W, 0]] = [[X*Z+Y*W, 0], [0, 0]]
    # The (0,0) entry is X*Z+Y*W.
    # But this is a 2x2 @ 2x2 matmul where the result's (0,0) entry is what we want.
    # We can use Strassen (rank 7) for this 2x2 matmul, and extract the (0,0) entry.
    # But we only need the (0,0) entry, so maybe we can do better.
    
    # Actually, the rank of the tensor for X*Z + Y*W (where X,Y,Z,W are 2x2) 
    # is at most 8 (schoolbook for each product). Can it be less?
    
    # Let me just use schoolbook for now: rank 4 for X*Z and rank 4 for Y*W, total 8.
    # For C2 similarly: rank 8.
    # Total for 4x4 @ 4x2: 8 + 8 = 16.
    
    # But the knowledge says rank 28 for 4x4 @ 4x2. That's more than 16, so maybe
    # I'm misunderstanding the problem or the knowledge is about a different decomposition.
    
    # Wait, maybe the rank 28 is for a different split. Let me re-read:
    # "Recursive Strassen for 4x4*4x4 (rank 49) plus 2x2 Strassen blocks for 4x4*4x2 (rank 28)"
    
    # Hmm, rank 28 for 4x4 @ 4x2. If I use Strassen (rank 7) for each 2x2 matmul:
    # C1 = A11*B1 + A12*B2: two 2x2 matmuls, each rank 7, total 14
    # C2 = A21*B1 + A22*B2: two 2x2 matmuls, each rank 7, total 14
    # Total: 28
    
    # So the approach is to use Strassen (not schoolbook) for each 2x2 matmul!
    # That gives rank 28 for 4x4 @ 4x2.
    
    # OK so let me implement this.
    
    # For 4x4 @ 4x2:
    # U has shape (16, 28), V has shape (8, 28), W has shape (8, 28)
    # (n*m = 16, m*p = 4*2 = 8, p*n = 2*4 = 8)
    
    # Wait, p=6 for the full problem, but for the 4x2 subproblem, p=2.
    # Let me be careful with the global indices.
    
    # In the full problem:
    # U[i*4+j, r] for i,j in 0..3
    # V[j*6+k, r] for j in 0..3, k in 0..5
    # W[k*4+i, r] for k in 0..5, i in 0..3
    
    # For the 4x4 @ 4x4 part (k in 0..3):
    # U[i*4+j, r] as computed above
    # V[j*6+k, r] for k in 0..3
    # W[k*4+i, r] for k in 0..3
    
    # For the 4x4 @ 4x2 part (k in 4..5):
    # U[i*4+j, r] 
    # V[j*6+k, r] for k in 4..5
    # W[k*4+i, r] for k in 4..5
    
    # Let me build the 4x4 @ 4x2 decomposition.
    
    # C1 = A11*B1 + A12*B2 where B1 = B[0:2, 4:6]... wait no.
    
    # Let me re-index. In the full problem:
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x6: B[j,k] for j in 0..3, k in 0..5
    # C is 4x6: C[i,k] for i in 0..3, k in 0..5
    
    # Split B into B_left (4x4, k in 0..3) and B_right (4x2, k in 4..5)
    # C = A @ B_left + A @ B_right
    
    # For A @ B_right (4x4 @ 4x2):
    # B_right[j,k] for j in 0..3, k in 4..5
    # Let B_r[j, k'] = B_right[j, k'+4] for k' in 0..1
    # C_right[i, k'] = sum_j A[i,j] * B_r[j, k']
    
    # Split A into 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # Split B_r into 2x2 blocks:
    # B1 = B_r[0:2, 0:2] = B[0:2, 4:6]
    # B2 = B_r[2:4, 0:2] = B[2:4, 4:6]
    
    # C_right = [[C1], [C2]] where:
    # C1 (2x2) = A11*B1 + A12*B2
    # C2 (2x2) = A21*B1 + A22*B2
    
    # For C1 = A11*B1 + A12*B2:
    # Use Strassen for A11*B1 (rank 7) and Strassen for A12*B2 (rank 7)
    # Total rank 14 for C1
    
    # For C2 = A21*B1 + A22*B2:
    # Use Strassen for A21*B1 (rank 7) and Strassen for A22*B2 (rank 7)
    # Total rank 14 for C2
    
    # Total rank for 4x4 @ 4x2: 28
    
    U42 = np.zeros((16, 28))
    V42 = np.zeros((8, 28))  # j*6+k for j in 0..3, k in 4..5 -> 8 entries
    W42 = np.zeros((8, 28))  # k*4+i for k in 4..5, i in 0..3 -> 8 entries
    
    # C1 = A11*B1 + A12*B2
    # A11: i in 0..1, j in 0..1 -> U index i*4+j
    # A12: i in 0..1, j in 2..3 -> U index i*4+j
    # B1: j in 0..1, k in 4..5 -> V index j*6+k
    # B2: j in 2..3, k in 4..5 -> V index j*6+k
    # C1: i in 0..1, k in 4..5 -> W index k*4+i
    
    # For A11*B1 (Strassen):
    # A11[i,j] for i,j in 0..1 -> local (i,j)
    # B1[j,k] for j in 0..1, k in 4..5 -> local (j, k-4)
    # C1[i,k] for i in 0..1, k in 4..5 -> local (k-4, i)
    
    for s in range(7):
        col = s  # 0..6
        for i in range(2):
            for j in range(2):
                U42[i*4+j, col] = strassen_a[s][i*2+j]
        for j in range(2):
            for k in range(2):
                V42[j*6+(4+k), col] = strassen_b[s][j*2+k]
        for k in range(2):
            for i in range(2):
                W42[(4+k)*4+i, col] = strassen_c[i*2+k][s]  # Wait, W index is k*n+i = k*4+i
    
    # Hmm, I need to be more careful. Let me redo this.
    
    # W index is k*4+i where k is the column index of C (0..5) and i is the row index (0..3)
    # For C1, k is in 4..5 and i is in 0..1
    # So W index = k*4+i for k in 4..5, i in 0..1
    
    # For the Strassen decomposition of A11*B1:
    # The "local" C matrix has entries C1[li, lk] for li in 0..1, lk in 0..1
    # where lk = k-4
    # W index = (4+lk)*4 + li
    
    # strassen_c[lk*2+li, s] gives the coefficient for C[li, lk]... 
    # Wait, I defined strassen_c as shape (4, 7) where the row index is k*2+i
    # So strassen_c[lk*2+li, s] is the coefficient for C[li, lk]
    
    # Let me redo:
    U42 = np.zeros((16, 28))
    V42 = np.zeros((8, 28))
    W42 = np.zeros((8, 28))
    
    # C1 = A11*B1 + A12*B2
    # Term 1: A11*B1 (Strassen, rank 7)
    for s in range(7):
        col = s
        # U: A11[i,j] for i,j in 0..1
        for i in range(2):
            for j in range(2):
                U42[i*4+j, col] = strassen_a[s][i*2+j]
        # V: B1[j,k] for j in 0..1, k in 4..5
        for j in range(2):
            for k in range(2):
                V42[j*6+(4+k), col] = strassen_b[s][j*2+k]
        # W: C1[i,k] for i in 0..1, k in 4..5 -> W index k*4+i
        for k in range(2):
            for i in range(2):
                W42[(4+k)*4+i, col] = strassen_c[k*2+i, s]
    
    # Term 2: A12*B2 (Strassen, rank 7)
    for s in range(7):
        col = 7 + s
        # U: A12[i,j] for i in 0..1, j in 2..3
        for i in range(2):
            for j in range(2):
                U42[i*4+(2+j), col] = strassen_a[s][i*2+j]
        # V: B2[j,k] for j in 2..3, k in 4..5
        for j in range(2):
            for k in range(2):
                V42[(2+j)*6+(4+k), col] = strassen_b[s][j*2+k]
        # W: C1[i,k] for i in 0..1, k in 4..5 -> W index k*4+i
        for k in range(2):
            for i in range(2):
                W42[(4+k)*4+i, col] = strassen_c[k*2+i, s]
    
    # C2 = A21*B1 + A22*B2
    # Term 3: A21*B1 (Strassen, rank 7)
    for s in range(7):
        col = 14 + s
        # U: A21[i,j] for i in 2..3, j in 0..1
        for i in range(2):
            for j in range(2):
                U42[(2+i)*4+j, col] = strassen_a[s][i*2+j]
        # V: B1[j,k] for j in 0..1, k in 4..5
        for j in range(2):
            for k in range(2):
                V42[j*6+(4+k), col] = strassen_b[s][j*2+k]
        # W: C2[i,k] for i in 2..3, k in 4..5 -> W index k*4+i
        for k in range(2):
            for i in range(2):
                W42[(4+k)*4+(2+i), col] = strassen_c[k*2+i, s]
    
    # Term 4: A22*B2 (Strassen, rank 7)
    for s in range(7):
        col = 21 + s
        # U: A22[i,j] for i in 2..3, j in 2..3
        for i in range(2):
            for j in range(2):
                U42[(2+i)*4+(2+j), col] = strassen_a[s][i*2+j]
        # V: B2[j,k] for j in 2..3, k in 4..5
        for j in range(2):
            for k in range(2):
                V42[(2+j)*6+(4+k), col] = strassen_b[s][j*2+k]
        # W: C2[i,k] for i in 2..3, k in 4..5 -> W index k*4+i
        for k in range(2):
            for i in range(2):
                W42[(4+k)*4+(2+i), col] = strassen_c[k*2+i, s]
    
    # Verify 4x4 @ 4x2 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                s = 0
                for r in range(28):
                    s += U42[i*4+j, r] * V42[j*6+k, r] * W42[k*4+i, r]
                if abs(s - 1) > 1e-10:
                    raise ValueError(f"4x4@4x2 verification failed at ({i},{j},{k}): {s}")
    
    # Now combine the two parts
    # Full U: (16, 49+28) = (16, 77)
    # Full V: (24, 49+28) = (24, 77)  [m*p = 4*6 = 24]
    # Full W: (24, 49+28) = (24, 77)  [p*n = 6*4 = 24]
    
    total_rank = 49 + 28
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))
    
    # Part 1: 4x4 @ 4x4 (columns 0..3 of B)
    U_full[:, 0:49] = U44
    # V44 has shape (16, 49) for j*4+k where j,k in 0..3
    # But we need V_full[j*6+k, r] for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j*6+k, 0:49] = V44[j*4+k, :]
    # W44 has shape (16, 49) for k*4+i where k,i in 0..3
    # We need W_full[k*4+i, r] for k in 0..3, i in 0..3
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, 0:49] = W44[k*4+i, :]
    
    # Part 2: 4x4 @ 4x2 (columns 4..5 of B)
    U_full[:, 49:77] = U42
    # V42 has shape (8, 28) for j*6+k where j in 0..3, k in 4..5
    # V_full[j*6+k, r] for j in 0..3, k in 4..5
    for j in range(4):
        for k in range(4, 6):
            V_full[j*6+k, 49:77] = V42[j*6+k, :]
    # W42 has shape (8, 28) for k*4+i where k in 4..5, i in 0..3
    # W_full[k*4+i, r] for k in 4..5, i in 0..3
    for k in range(4, 6):
        for i in range(4):
            W_full[k*4+i, 49:77] = W42[k*4+i, :]
    
    # Final verification
    for i in range(4):
        for j in range(4):
            for k in range(6):
                s = 0
                for r in range(77):
                    s += U_full[i*4+j, r] * V_full[j*6+k, r] * W_full[k*4+i, r]
                if abs(s - 1) > 1e-10:
                    raise ValueError(f"Full verification failed at ({i},{j},{k}): {s}")
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
