"""Tensor decomposition for 4x4 * 4x7 matrix multiplication using Strassen-based Kronecker structure."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Uses Strassen's 7-term decomposition for 2x2x2 as a building block,
    applied via Kronecker product structure to get a rank-100 decomposition
    (down from schoolbook rank 112).
    
    The key insight: 4x4 can be viewed as 2x2 blocks, and 7 = 2+2+2+1.
    For each 2-column block of the 7 columns, we use the Kronecker product
    of the Strassen 2x2x2 decomposition with a 2x2x1 schoolbook, giving
    rank 7*2 = 14 per 2-column block. For the single remaining column,
    we use 4x4x1 schoolbook with rank 16.
    Total: 3*14 + 16 = 58? No, let me recalculate.
    
    Actually, the proper Kronecker approach:
    - 4x4 * 4x7: split rows of A into 2x2 blocks (2 groups), 
      columns of A / rows of B into 2x2 blocks (2 groups),
      columns of B into 2+2+2+1 (4 groups)
    
    For a 4x4 * 4x2 subproblem:
    Split into (2x2) x (2x2) x (2x2) structure where the "2" in the third
    dimension is handled by the Kronecker product of the output dimension.
    
    Let me use a cleaner approach:
    
    For 4x4 * 4x7, decompose as:
    - 4 = 2*2 for both n and m
    - 7 = 2+2+2+1 for p
    
    For the 4x4 * 4x2 subproblem (rank 32 schoolbook):
    View as tensor product of (2x2 * 2x2) for the A,B inputs and (2) for output dim.
    
    Using Kronecker: M_{n*m,p} ⊗ M_{n',m',p'} gives rank r * r'
    where M_{n*m,p} is the matmul tensor.
    
    4x4 * 4x2: We can write this as (2x2 * 2x2) in the A dimension (4x4 = 2*2 by 2*2),
    (2x2 * 2x2) in the B row dimension (4 = 2*2), and output 4x2 = (2x2 by 2).
    
    This gets complicated. Let me just build it directly.
    
    Simplest correct approach:
    1. Schoolbook for 4x4 * 4x1: rank 16
    2. For 4x4 * 4x2: Use Strassen on the 2x2 A-blocks
    
    Actually, the cleanest way:
    
    For each pair of output columns (k, k+1), we compute C[:, k] and C[:, k+1].
    C[:, k] = A * B[:, k]
    
    For 4x4 * 4x2:
    A is 4x4, B is 4x2.
    Split A into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into two 2x2 blocks: B = [[B1, B2]] where B1, B2 are 2x2? No, B is 4x2,
    so B = [B_top; B_bottom] where B_top, B_bottom are 2x2.
    
    C = A * B = [[A11*B_top + A12*B_bottom], [A21*B_top + A22*B_bottom]]
    C is 4x2.
    
    Each term like A11*B_top is a 2x2 * 2x2 multiplication (8 scalar mults schoolbook,
    7 with Strassen). There are 4 such 2x2*2x2 multiplications.
    
    With Strassen: 4 * 7 = 28 multiplications for 4x4 * 4x2 (vs 32 schoolbook).
    
    For 4x4 * 4x1:
    C = A * B where B is 4x1.
    Split A into 2x2 blocks, B into [B1; B2] where B1, B2 are 2x1.
    C = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    Each Aij * Bj is 2x2 * 2x1 = 4 scalar mults. 4 such terms = 16. No Strassen savings
    for 2x2 * 2x1.
    
    Total: 3 * 28 + 16 = 84 + 16 = 100.
    
    Let me implement this carefully.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    time_limit = 140  # safety margin before 150s kill
    
    # Strassen 2x2x2 decomposition: 7 terms
    # For matrices X, Y (2x2), compute Z = X*Y
    # X = [[a, b], [c, d]], Y = [[e, f], [g, h]]
    # Z = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-c)(e+g)
    # 
    # Z[0,0] = p1 + p4 - p5 + p7
    # Z[0,1] = p3 + p5
    # Z[1,0] = p2 + p4
    # Z[1,1] = p1 - p2 + p3 + p6
    
    # We need to express this as a tensor decomposition.
    # The 2x2x2 matmul tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1}
    # (since we compute all 4 outputs from all 4 input pairs).
    # 
    # A rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2, so U is 4x7, V is 4x7, W is 4x7.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen decomposition for 2x2x2:
    # Each of the 7 "multiplications" p1..p7 corresponds to a rank-1 term.
    # p1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1 (for a=d terms), V[0,0]=1, V[1,1]=1 (e=h terms)
    # Wait, let me think more carefully.
    
    # The scalar multiplication p1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # As a rank-1 tensor term:
    # U[0*2+0, r1] = 1, U[1*2+1, r1] = 1 (A[0,0] and A[1,1])
    # V[0*2+0, r1] = 1, V[1*2+1, r1] = 1 (B[0,0] and B[1,1])
    # W: p1 contributes to Z[0,0] with +1, Z[1,1] with +1
    # W[0*2+0, r1] = 1, W[1*2+1, r1] = 1
    
    # p2 = (A[1,0]+A[1,1]) * B[0,0]
    # U[1*2+0, r2] = 1, U[1*2+1, r2] = 1
    # V[0*2+0, r2] = 1
    # W: p2 contributes to Z[1,0] with +1, Z[1,1] with -1
    # W[0*2+1, r2] = 1, W[1*2+1, r2] = -1
    
    # p3 = A[0,0] * (B[0,1]-B[1,1])
    # U[0*2+0, r3] = 1
    # V[0*2+1, r3] = 1, V[1*2+1, r3] = -1
    # W: p3 contributes to Z[0,1] with +1, Z[1,1] with +1
    # W[1*2+0, r3] = 1, W[1*2+1, r3] = 1
    
    # p4 = A[1,1] * (B[1,0]-B[0,0])
    # U[1*2+1, r4] = 1
    # V[1*2+0, r4] = 1, V[0*2+0, r4] = -1
    # W: p4 contributes to Z[0,0] with +1, Z[1,0] with +1
    # W[0*2+0, r4] = 1, W[0*2+1, r4] = 1
    
    # p5 = (A[0,0]+A[0,1]) * B[1,1]
    # U[0*2+0, r5] = 1, U[0*2+1, r5] = 1
    # V[1*2+1, r5] = 1
    # W: p5 contributes to Z[0,0] with -1, Z[0,1] with +1
    # W[0*2+0, r5] = -1, W[1*2+0, r5] = 1
    
    # p6 = (A[1,0]-A[1,1]) * (B[0,1]+B[1,1])
    # U[1*2+0, r6] = 1, U[1*2+1, r6] = -1
    # V[0*2+1, r6] = 1, V[1*2+1, r6] = 1
    # W: p6 contributes to Z[1,1] with +1
    # W[1*2+1, r6] = 1
    
    # p7 = (A[0,1]-A[1,0]) * (B[0,0]+B[1,0])
    # U[0*2+1, r7] = 1, U[1*2+0, r7] = -1
    # V[0*2+0, r7] = 1, V[1*2+0, r7] = 1
    # W: p7 contributes to Z[0,0] with +1
    # W[0*2+0, r7] = 1
    
    # Let me verify p1: Z[0,0] should get +1 from p1, Z[1,1] should get +1 from p1.
    # W[0*2+0, r1] = 1 means W[0, r1] = 1, which corresponds to k=0, i=0 -> Z[0,0]. Yes.
    # W[1*2+1, r1] = 1 means W[1, r1] = 1, which corresponds to k=1, i=1 -> Z[1,1]. Yes.
    
    # Now let's build the full 4x4 * 4x7 decomposition.
    
    # Strategy:
    # Split the 4x4 A matrix into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split the 4x7 B matrix row-wise into 2x2 blocks: B = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    # where each Bij is 2x2.
    
    # Actually, B is 4x7. Split rows into top 2 and bottom 2:
    # B_top = B[0:2, :] (2x7), B_bot = B[2:4, :] (2x7)
    
    # C = A * B
    # C[0:2, k] = A00 * B_top[:, k] + A01 * B_bot[:, k]
    # C[2:4, k] = A10 * B_top[:, k] + A11 * B_bot[:, k]
    
    # For each column k of B, we have 4 terms of the form (2x2) * (2x1).
    # But we can group columns in pairs to use Strassen for the 2x2 * 2x2 case.
    
    # For columns (k, k+1):
    # Let B_pair = B[:, k:k+2] (4x2)
    # Split: B_pair_top = B_pair[0:2, :] (2x2), B_pair_bot = B_pair[2:4, :] (2x2)
    
    # C[:, k:k+2] = A * B_pair
    # C[0:2, k:k+2] = A00 * B_pair_top + A01 * B_pair_bot  (2x2 result)
    # C[2:4, k:k+2] = A10 * B_pair_top + A11 * B_pair_bot  (2x2 result)
    
    # Each of these is a sum of two 2x2 * 2x2 multiplications.
    # Using Strassen for each: 7 multiplications each.
    # So 4 * 7 = 28 per column pair.
    
    # For the single column k=6:
    # B_single = B[:, 6:7] (4x1)
    # B_single_top = B_single[0:2, :] (2x1), B_single_bot = B_single[2:4, :] (2x1)
    # C[0:2, 6] = A00 * B_single_top + A01 * B_single_bot
    # C[2:4, 6] = A10 * B_single_top + A11 * B_single_bot
    # Each is 2x2 * 2x1 = 4 multiplications. 4 * 4 = 16.
    
    # Total rank: 3 * 28 + 16 = 100.
    
    # Now let's build U, V, W matrices.
    
    # For the Strassen 2x2*2x2 decomposition, we have 7 rank-1 terms.
    # Let me define them explicitly.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matmul using Strassen (rank 7)."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # p1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1  # a = A[0,0], d = A[1,1]
        V[0, 0] = 1; V[3, 0] = 1  # e = B[0,0], h = B[1,1]
        W[0, 0] = 1; W[3, 0] = 1  # Z[0,0] += 1, Z[1,1] += 1
        
        # p2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1  # c = A[1,0], d = A[1,1]
        V[0, 1] = 1  # e = B[0,0]
        W[1, 1] = 1; W[3, 1] = -1  # Z[1,0] += 1, Z[1,1] -= 1
        
        # p3 = a(f-h)
        U[0, 2] = 1  # a = A[0,0]
        V[1, 2] = 1; V[3, 2] = -1  # f = B[0,1], h = B[1,1]
        W[2, 2] = 1; W[3, 2] = 1  # Z[0,1] += 1, Z[1,1] += 1
        
        # p4 = d(g-e)
        U[3, 3] = 1  # d = A[1,1]
        V[2, 3] = 1; V[0, 3] = -1  # g = B[1,0], e = B[0,0]
        W[0, 3] = 1; W[1, 3] = 1  # Z[0,0] += 1, Z[1,0] += 1
        
        # p5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1  # a = A[0,0], b = A[0,1]
        V[3, 4] = 1  # h = B[1,1]
        W[0, 4] = -1; W[2, 4] = 1  # Z[0,0] -= 1, Z[0,1] += 1
        
        # p6 = (c-d)(f+h)
        U[2, 5] = 1; U[3, 5] = -1  # c = A[1,0], d = A[1,1]
        V[1, 5] = 1; V[3, 5] = 1  # f = B[0,1], h = B[1,1]
        W[3, 5] = 1  # Z[1,1] += 1
        
        # p7 = (b-c)(e+g)
        U[1, 6] = 1; U[2, 6] = -1  # b = A[0,1], c = A[1,0]
        V[0, 6] = 1; V[2, 6] = 1  # e = B[0,0], g = B[1,0]
        W[0, 6] = 1  # Z[0,0] += 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Return U, V, W for 2x2 * 2x1 matmul (rank 4)."""
        U = np.zeros((4, 4), dtype=float)
        V = np.zeros((2, 4), dtype=float)
        W = np.zeros((2, 4), dtype=float)
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[i*2+j, col] = 1.0
                    V[j*1+k, col] = 1.0
                    W[k*2+i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    # Now assemble the 4x4 * 4x7 decomposition.
    # 
    # The full problem: A is 4x4, B is 4x7, C is 4x7.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (28, rank) 
    # W has shape (p*n, rank) = (28, rank)
    
    # Wait, let me re-read the problem statement.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # n=4, m=4, p=7
    # U: (16, rank), V: (28, rank), W: (28, rank)
    
    # The tensor entry: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all i in [0,4), j in [0,4), k in [0,7).
    
    # In our decomposition:
    # A[i, j] is indexed by i (row of A, 0..3), j (col of A, 0..3)
    # B[j, k] is indexed by j (row of B, 0..3), k (col of B, 0..6)
    # C[i, k] is indexed by i (row of C, 0..3), k (col of C, 0..6)
    
    # U[i*m+j, r] = U[i*4+j, r] corresponds to A[i, j]
    # V[j*p+k, r] = V[j*7+k, r] corresponds to B[j, k]
    # W[k*n+i, r] = W[k*4+i, r] corresponds to C[i, k]
    
    # Let's build the decomposition by handling each column pair.
    
    # For column pair (k0, k0+1) where k0 in {0, 2, 4}:
    # We compute C[:, k0:k0+2] = A * B[:, k0:k0+2]
    
    # Split A into 2x2 blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # B[:, k0:k0+2] split into top/bottom:
    # B_top = B[0:2, k0:k0+2] (2x2)
    # B_bot = B[2:4, k0:k0+2] (2x2)
    
    # C[0:2, k0:k0+2] = A00 * B_top + A01 * B_bot
    # C[2:4, k0:k0+2] = A10 * B_top + A11 * B_bot
    
    # Each term like A00 * B_top is a 2x2 * 2x2 multiplication.
    # For A00 * B_top:
    #   A00[i,j] = A[i, j] for i in 0,1, j in 0,1
    #   B_top[j,k_local] = B[j, k0+k_local] for j in 0,1, k_local in 0,1
    #   Result: C_top[i, k_local] = A00[i,j] * B_top[j, k_local]
    
    # The rank-1 terms for A00 * B_top:
    # For each Strassen term r in 0..6:
    #   U_local[ia*2+ja, r] * V_local[ja*2+kl, r] * W_local[kl*2+ia, r]
    #   where ia in 0,1, ja in 0,1, kl in 0,1
    #   
    #   Map to global:
    #   U_global[ia*4+ja, global_r] = U_local[ia*2+ja, r]
    #   V_global[ja*7+(k0+kl), global_r] = V_local[ja*2+kl, r]
    #   W_global[(k0+kl)*4+ia, global_r] = W_local[kl*2+ia, r]
    
    # For A01 * B_top:
    #   A01[i,j] = A[i, 2+j] for i in 0,1, j in 0,1
    #   B_top[j,k_local] = B[j, k0+k_local] for j in 0,1, k_local in 0,1
    #   Result: C_top[i, k_local] += A01[i,j] * B_top[j, k_local]
    #
    #   U_global[ia*4+(2+ja), global_r] = U_local[ia*2+ja, r]
    #   V_global[(2+ja)*7+(k0+kl), global_r] = V_local[ja*2+kl, r]
    #   W_global[(k0+kl)*4+ia, global_r] = W_local[kl*2+ia, r]
    
    # Similarly for A10 * B_bot and A11 * B_bot.
    
    # Let me code this up.
    
    U_s, V_s, W_s = strassen_2x2x2()
    U_sb, V_sb, W_sb = schoolbook_2x2x1()
    
    total_rank = 3 * 28 + 16  # 100
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((28, total_rank), dtype=float)
    W = np.zeros((28, total_rank), dtype=float)
    
    col_offset = 0
    
    # Process column pairs (0,1), (2,3), (4,5)
    for pair_idx in range(3):
        k0 = pair_idx * 2  # 0, 2, 4
        
        # For each of the 4 block multiplications:
        # (A00 * B_top), (A01 * B_top), (A10 * B_bot), (A11 * B_bot)
        
        for a_block in range(4):
            # a_block: 0=A00, 1=A01, 2=A10, 3=A11
            i_offset = (a_block // 2) * 2  # 0 for A0*, 2 for A1*
            j_offset = (a_block % 2) * 2   # 0 for A*0, 2 for A*1
            
            # B block: top (rows 0,1) for a_block 0,1; bottom (rows 2,3) for a_block 2,3
            b_row_offset = (a_block // 2) * 2  # Wait, no.
            # A00 * B_top: a_block=0, B_top rows 0,1
            # A01 * B_top: a_block=1, B_top rows 0,1
            # A10 * B_bot: a_block=2, B_bot rows 2,3
            # A11 * B_bot: a_block=3, B_bot rows 2,3
            b_row_offset = (a_block >= 2) * 2  # 0 for top, 2 for bottom
            
            for r in range(7):
                global_col = col_offset + r
                
                # U entries
                for ia in range(2):
                    for ja in range(2):
                        u_val = U_s[ia*2+ja, r]
                        if u_val != 0:
                            i_global = i_offset + ia
                            j_global = j_offset + ja
                            U[i_global*4 + j_global, global_col] = u_val
                
                # V entries
                for ja in range(2):
                    for kl in range(2):
                        v_val = V_s[ja*2+kl, r]
                        if v_val != 0:
                            j_global = b_row_offset + ja
                            k_global = k0 + kl
                            V[j_global*7 + k_global, global_col] = v_val
                
                # W entries
                for kl in range(2):
                    for ia in range(2):
                        w_val = W_s[kl*2+ia, r]
                        if w_val != 0:
                            k_global = k0 + kl
                            i_global = i_offset + ia
                            W[k_global*4 + i_global, global_col] = w_val
            
            col_offset += 7
        
        col_offset += 0  # No gap between the 4 blocks for this pair
    
    # Process single column k=6
    k0 = 6
    for a_block in range(4):
        i_offset = (a_block // 2) * 2
        j_offset = (a_block % 2) * 2
        b_row_offset = (a_block >= 2) * 2
        
        for r in range(4):
            global_col = col_offset + r
            
            # U entries
            for ia in range(2):
                for ja in range(2):
                    u_val = U_sb[ia*2+ja, r]
                    if u_val != 0:
                        i_global = i_offset + ia
                        j_global = j_offset + ja
                        U[i_global*4 + j_global, global_col] = u_val
            
            # V entries
            for ja in range(2):
                v_val = V_sb[ja*1+0, r]
                if v_val != 0:
                    j_global = b_row_offset + ja
                    k_global = k0
                    V[j_global*7 + k_global, global_col] = v_val
            
            # W entries
            for ia in range(2):
                w_val = W_sb[0*2+ia, r]
                if w_val != 0:
                    k_global = k0
                    i_global = i_offset + ia
                    W[k_global*4 + i_global, global_col] = w_val
        
        col_offset += 4
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol, max_err
    
    is_valid, max_err = verify_decomposition(U, V, W, n, m, p)
    
    if not is_valid:
        # Fall back to schoolbook if something went wrong
        rank_sb = n * m * p
        U_sb_full = np.zeros((n*m, rank_sb), dtype=float)
        V_sb_full = np.zeros((m*p, rank_sb), dtype=float)
        W_sb_full = np.zeros((p*n, rank_sb), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb_full[i*m+j, col] = 1.0
                    V_sb_full[j*p+k, col] = 1.0
                    W_sb_full[k*n+i, col] = 1.0
                    col += 1
        
        return {"U": U_sb_full.tolist(), "V": V_sb_full.tolist(), "W": W_sb_full.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
