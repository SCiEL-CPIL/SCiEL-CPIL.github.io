"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # We exploit the known result that rank(2x3x8) <= 44 via block decomposition.
    # Strategy:
    # 1. Split the 3 shared dimension into 1+2.
    # 2. Use Strassen (rank 7) for the 2x2x2 block.
    # 3. Use schoolbook (rank 4) for the 1x2x2 blocks.
    # Total rank = 7 + 4 + 4 = 15? No, wait.
    # Let's re-verify the block structure.
    # A is 2x3, B is 3x8.
    # Let A = [A1 | A2] where A1 is 2x1, A2 is 2x2.
    # Let B = [B1; B2] where B1 is 1x8, B2 is 2x8.
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x8. This is a sum of 2*1*8 = 16 scalar products? No, it's a matrix product of (2x1) and (1x8).
    # The tensor for (2x1) * (1x8) has rank 2*1*8 = 16?
    # Actually, the rank of the multiplication tensor T_{n,m,p} is the tensor rank.
    # For (2,1,8), the rank is 16? Yes, because it's just outer products.
    # For (2,2,8), the rank is 32?
    # Wait, the "44" claim in the prompt knowledge says: "Using block decomposition m=1+2 and Strassen for 2x2x2 blocks reduces rank from 48 to 44".
    # Let's trace that logic.
    # If we split m=3 into 1+2.
    # Term 1: A(2x1) * B(1x8). Rank = 2*1*8 = 16.
    # Term 2: A(2x2) * B(2x8).
    # If we use Strassen for the 2x2 part of the multiplication, we need to handle the 8 columns.
    # Strassen computes 2x2 = 2x2 * 2x2 in 7 multiplications.
    # Here we have 2x2 * 2x8. This is equivalent to 8 instances of 2x2 * 2x1? No.
    # It's equivalent to computing 8 columns of the result.
    # Each column is a 2x2 * 2x1 multiplication.
    # If we treat the 2x8 matrix B2 as 8 vectors of length 2.
    # We can compute A2 * B2 using the Strassen decomposition for the 2x2x2 tensor, applied to each column?
    # No, Strassen reduces the rank of the 2x2x2 tensor from 8 to 7.
    # The tensor for (2,2,8) can be seen as 8 copies of the (2,2,1) tensor? No, the output dimension is 8.
    # The tensor T_{2,2,8} has rank 32 (schoolbook).
    # Can we do better?
    # If we use Strassen, we compute 7 "large" products. But the inputs are not 2x2 matrices, they are vectors/matrices.
    # Let's look at the structure: C = A2 * B2.
    # A2 is 2x2, B2 is 2x8.
    # Let A2 = [[a11, a12], [a21, a22]].
    # Let B2 = [[b11..b18], [b21..b28]].
    # C11 = a11*b11 + a12*b21 ... a11*b18 + a12*b28.
    # This is a bilinear form.
    # The rank of T_{2,2,8} is known to be 28? Or can it be reduced?
    # The prompt says "reduces rank from 48 to 44".
    # 48 is the schoolbook rank: 2*3*8 = 48.
    # If the new rank is 44, we saved 4.
    # How?
    # Maybe the split is different.
    # Or maybe the Strassen application saves 1 rank per 2x2x2 block, and there are 4 such blocks?
    # If we view the 2x3x8 as four 1x1x1 blocks? No.
    # Let's assume the prompt's "44" is a valid target found by another program.
    # I will try to construct a rank 44 decomposition.
    #
    # Let's analyze the rank 44 construction.
    # If we split the middle dimension 3 into 1+2.
    # Part 1: (2,1,8). Rank 16.
    # Part 2: (2,2,8).
    # If Part 2 has rank 28, total is 16+28=44.
    # Is rank(2,2,8) = 28?
    # Schoolbook is 32.
    # Strassen for 2x2x2 is rank 7.
    # For 2x2x8, we can view it as 8 columns.
    # If we could use Strassen on the 2x2x2 core, we might save 1 multiplication per 2x2x2 block.
    # Since there are 8 columns, and the structure is linear, maybe we save 4 multiplications total?
    # 32 - 4 = 28.
    # This matches 16 + 28 = 44.
    #
    # So the plan is:
    # 1. Decompose (2,1,8) with rank 16 (schoolbook).
    # 2. Decompose (2,2,8) with rank 28 (Strassen-inspired).
    #
    # How to construct the rank 28 decomposition for (2,2,8)?
    # The tensor T_{2,2,8} maps (A2, B2) to C.
    # A2 is 2x2, B2 is 2x8.
    # We can think of B2 as 8 vectors b1..b8 in R^2.
    # C is 2x8, so columns c1..c8 in R^2.
    # c_j = A2 * b_j.
    # This is 8 separate 2x2 * 2x1 multiplications.
    # The tensor for 2x2 * 2x1 is T_{2,2,1} which has rank 4 (schoolbook).
    # Total rank 8*4 = 32.
    # Strassen reduces T_{2,2,2} (which is 2x2*2x2) from 8 to 7.
    # Does it help for T_{2,2,1}?
    # T_{2,2,1} is a sub-tensor of T_{2,2,2} (setting the second output column to 0).
    # If we use the 7 Strassen terms, we might need fewer than 4?
    # Actually, the rank of T_{2,2,1} is 4. It cannot be reduced below 4 because the output is 2x1, requiring 2 independent linear combinations for each of the 2 inputs?
    # Wait, the rank of the tensor for 2x2 * 2x1 is 4.
    # Proof: The tensor is sum_{i,j} e_i e_j e_k.
    # It is known that rank(T_{2,2,1}) = 4.
    # So we can't reduce the rank per column.
    #
    # Then how do we get 28?
    # Maybe the block decomposition is different.
    # What if we split the output dimension 8?
    # Or the input dimension 2?
    #
    # Let's reconsider the "44" claim.
    # Maybe the rank of (2,2,8) is indeed 28.
    # Is there a known result?
    # rank(2,2,2) = 7.
    # rank(2,2,3) ?
    # rank(2,2,4) ?
    #
    # Let's try to build the rank 44 solution explicitly.
    # If I can't find the exact 28-rank decomposition for (2,2,8), I should fall back to a safe, valid decomposition.
    # The current best is 48 (score 1.0).
    # If I can't beat 48, I should return 48.
    # But the prompt says "reduces rank from 48 to 44".
    # I will attempt to implement the 44-rank decomposition.
    #
    # Hypothesis: The (2,2,8) tensor can be decomposed using 28 terms.
    # How?
    # Perhaps by grouping columns?
    # If we group the 8 columns into 4 pairs.
    # Each pair is a 2x2 output.
    # So we have 4 instances of 2x2 * 2x2 (where the input B is 2x2 for each pair).
    # Wait, B is 2x8. If we take columns 1,2 as B12 (2x2), columns 3,4 as B34 (2x2), etc.
    # Then C = A2 * [B12, B34, B56, B78].
    # C12 = A2 * B12. This is a 2x2 * 2x2 multiplication.
    # There are 4 such blocks.
    # Each 2x2 * 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    # YES! This works!
    #
    # So the full decomposition:
    # 1. Split m=3 into 1+2.
    # 2. Term 1: A(2x1) * B(1x8). Rank 16.
    #    A1 is 2x1, B1 is 1x8.
    #    C1 = A1 * B1.
    #    This is 2*1*8 = 16 scalar products.
    #    U, V, W for this part:
    #    For each i in 0..1, k in 0..7:
    #       U[i, col] = A1[i]
    #       V[0, col] = B1[k]  (Note: B1 is 1x8, so index is just k)
    #       W[k, col] = C1[i,k]
    #    Wait, the indices for U, V, W must match the global indices.
    #    Global U size: n*m = 2*3 = 6.
    #    Global V size: m*p = 3*8 = 24.
    #    Global W size: p*n = 8*2 = 16.
    #
    #    Let's map the local indices to global.
    #    A is 2x3. A1 is columns 0. A2 is columns 1,2.
    #    B is 3x8. B1 is row 0. B2 is rows 1,2.
    #
    #    Part 1: A1 (2x1) * B1 (1x8).
    #    Local i (0..1), local k (0..7).
    #    Global A index: i (row), 0 (col). -> U index: i*m + 0 = i*3 + 0.
    #    Global B index: 0 (row), k (col). -> V index: 0*p + k = k.
    #    Global C index: k (row), i (col). -> W index: k*n + i = k*2 + i.
    #    There are 2*8 = 16 terms.
    #
    #    Part 2: A2 (2x2) * B2 (2x8).
    #    Split B2 into 4 blocks of 2 columns each.
    #    Block b (0..3): Columns 2b, 2b+1.
    #    Local A2 indices: r (0..1), c (0..1).
    #    Local B2 indices: r (0..1), c (0..1) within the block.
    #    Global A index: r (row), 1+c (col). -> U index: r*3 + 1 + c.
    #    Global B index: 1+r (row), 2b+c (col). -> V index: (1+r)*8 + 2b + c.
    #    Global C index: 2b+c (row), r (col). -> W index: (2b+c)*2 + r.
    #
    #    For each block b, we use Strassen to compute the 2x2 * 2x2 product.
    #    Strassen terms for M = A2 * B_block:
    #    Let A2 = [[a00, a01], [a10, a11]].
    #    Let B_block = [[b00, b01], [b10, b11]].
    #    M = [[m00, m01], [m10, m11]].
    #
    #    Strassen:
    #    M1 = (a00+a11)(b00+b11)
    #    M2 = (a10+a11)b00
    #    M3 = a00(b01-b11)
    #    M4 = a11(b10-b00)
    #    M5 = (a00+a01)b11
    #    M6 = (a10-a00)(b00+b01)
    #    M7 = (a01-a11)(b10+b11)
    #
    #    m00 = M1 + M4 - M5 + M7
    #    m01 = M3 + M5
    #    m10 = M2 + M4
    #    m11 = M1 - M2 + M3 + M6
    #
    #    We need to express these as sums of rank-1 terms.
    #    Each Mk is a product of a linear form in A and a linear form in B.
    #    So each Mk contributes 1 rank term.
    #    Total 7 terms per block.
    #    4 blocks * 7 = 28 terms.
    #
    #    Total rank = 16 + 28 = 44.
    #
    #    Now I need to construct U, V, W matrices.
    #    Rank R = 44.
    #    U is 6x44, V is 24x44, W is 16x44.
    #
    #    Initialize with zeros.
    #
    #    Part 1 (16 terms):
    #    col = 0
    #    for i in 0..1:
    #       for k in 0..7:
    #          U[i*3 + 0][col] = 1
    #          V[0*8 + k][col] = 1
    #          W[k*2 + i][col] = 1
    #          col += 1
    #
    #    Part 2 (28 terms):
    #    for b in 0..3:
    #       # Block b uses columns 2b, 2b+1 of B2 (rows 1,2 of B)
    #       # And all of A2 (cols 1,2 of A)
    #       
    #       # Define the linear forms for Strassen
    #       # A indices: (r, c) -> global U index r*3 + 1 + c
    #       # B indices: (r, c) -> global V index (1+r)*8 + 2b + c
    #       
    #       # M1 = (a00+a11)(b00+b11)
    #       # a00 is A[0,1], a11 is A[1,2]
    #       # b00 is B[1,2b], b11 is B[2,2b+1]
    #       
    #       # We add 1 term for M1.
    #       U[0*3 + 1][col] += 1  # a00
    #       U[1*3 + 2][col] += 1  # a11
    #       V[1*8 + 2b + 0][col] += 1  # b00
    #       V[2*8 + 2b + 1][col] += 1  # b11
    #       # W contributions:
    #       # m00 += M1
    #       # m11 += M1
    #       # m00 is C[2b, 0] -> W[2b*2 + 0]
    #       # m11 is C[2b+1, 1] -> W[(2b+1)*2 + 1]
    #       W[2b*2 + 0][col] += 1
    #       W[(2b+1)*2 + 1][col] += 1
    #       col += 1
    #
    #       # M2 = (a10+a11)b00
    #       # a10 is A[1,1], a11 is A[1,2]
    #       # b00 is B[1,2b]
    #       U[1*3 + 1][col] += 1
    #       U[1*3 + 2][col] += 1
    #       V[1*8 + 2b + 0][col] += 1
    #       # m10 += M2 -> C[2b+1, 0] -> W[(2b+1)*2 + 0]
    #       # m11 -= M2 -> C[2b+1, 1] -> W[(2b+1)*2 + 1]
    #       W[(2b+1)*2 + 0][col] += 1
    #       W[(2b+1)*2 + 1][col] -= 1
    #       col += 1
    #
    #       # M3 = a00(b01-b11)
    #       # a00 is A[0,1]
    #       # b01 is B[1,2b+1], b11 is B[2,2b+1]
    #       U[0*3 + 1][col] += 1
    #       V[1*8 + 2b + 1][col] += 1
    #       V[2*8 + 2b + 1][col] -= 1
    #       # m01 += M3 -> C[2b+1, 0] ?? No.
    #       # m01 is C[2b+1, 0]? No, C is 2x2.
    #       # m00 is C[0,0] of the block, which is global C[2b, 0].
    #       # m01 is C[0,1] of the block, which is global C[2b, 1].
    #       # m10 is C[1,0] of the block, which is global C[2b+1, 0].
    #       # m11 is C[1,1] of the block, which is global C[2b+1, 1].
    #       
    #       # Correction on indices:
    #       # Block output is 2x2.
    #       # Row 0 of block -> Global Row 2b.
    #       # Row 1 of block -> Global Row 2b+1.
    #       # Col 0 of block -> Global Col 0.
    #       # Col 1 of block -> Global Col 1.
    #       
    #       # So:
    #       # m00 -> W[2b*2 + 0]
    #       # m01 -> W[2b*2 + 1]
    #       # m10 -> W[(2b+1)*2 + 0]
    #       # m11 -> W[(2b+1)*2 + 1]
    #       
    #       # Re-evaluating M3:
    #       # m01 += M3
    #       # m11 += M3
    #       W[2b*2 + 1][col] += 1
    #       W[(2b+1)*2 + 1][col] += 1
    #       col += 1
    #
    #       # M4 = a11(b10-b00)
    #       # a11 is A[1,2]
    #       # b10 is B[2,2b], b00 is B[1,2b]
    #       U[1*3 + 2][col] += 1
    #       V[2*8 + 2b + 0][col] += 1
    #       V[1*8 + 2b + 0][col] -= 1
    #       # m00 += M4
    #       # m10 += M4
    #       W[2b*2 + 0][col] += 1
    #       W[(2b+1)*2 + 0][col] += 1
    #       col += 1
    #
    #       # M5 = (a00+a01)b11
    #       # a00 is A[0,1], a01 is A[0,2]
    #       # b11 is B[2,2b+1]
    #       U[0*3 + 1][col] += 1
    #       U[0*3 + 2][col] += 1
    #       V[2*8 + 2b + 1][col] += 1
    #       # m00 -= M5
    #       # m01 += M5
    #       W[2b*2 + 0][col] -= 1
    #       W[2b*2 + 1][col] += 1
    #       col += 1
    #
    #       # M6 = (a10-a00)(b00+b01)
    #       # a10 is A[1,1], a00 is A[0,1]
    #       # b00 is B[1,2b], b01 is B[1,2b+1]
    #       U[1*3 + 1][col] += 1
    #       U[0*3 + 1][col] -= 1
    #       V[1*8 + 2b + 0][col] += 1
    #       V[1*8 + 2b + 1][col] += 1
    #       # m11 += M6
    #       W[(2b+1)*2 + 1][col] += 1
    #       col += 1
    #
    #       # M7 = (a01-a11)(b10+b11)
    #       # a01 is A[0,2], a11 is A[1,2]
    #       # b10 is B[2,2b], b11 is B[2,2b+1]
    #       U[0*3 + 2][col] += 1
    #       U[1*3 + 2][col] -= 1
    #       V[2*8 + 2b + 0][col] += 1
    #       V[2*8 + 2b + 1][col] += 1
    #       # m00 += M7
    #       W[2b*2 + 0][col] += 1
    #       col += 1
    #
    #    End of Part 2.
    #
    #    Total col should be 16 + 28 = 44.
    #
    #    I will implement this and verify it.

    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: (2,1,8) schoolbook
    for i in range(n):  # i=0,1
        for k in range(p):  # k=0..7
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: (2,2,8) Strassen blocks
    for b in range(4):
        # M1
        U[0 * m + 1][col] += 1
        U[1 * m + 2][col] += 1
        V[1 * p + 2 * b + 0][col] += 1
        V[2 * p + 2 * b + 1][col] += 1
        W[2 * b * n + 0][col] += 1
        W[(2 * b + 1) * n + 1][col] += 1
        col += 1
        
        # M2
        U[1 * m + 1][col] += 1
        U[1 * m + 2][col] += 1
        V[1 * p + 2 * b + 0][col] += 1
        W[(2 * b + 1) * n + 0][col] += 1
        W[(2 * b + 1) * n + 1][col] -= 1
        col += 1
        
        # M3
        U[0 * m + 1][col] += 1
        V[1 * p + 2 * b + 1][col] += 1
        V[2 * p + 2 * b + 1][col] -= 1
        W[2 * b * n + 1][col] += 1
        W[(2 * b + 1) * n + 1][col] += 1
        col += 1
        
        # M4
        U[1 * m + 2][col] += 1
        V[2 * p + 2 * b + 0][col] += 1
        V[1 * p + 2 * b + 0][col] -= 1
        W[2 * b * n + 0][col] += 1
        W[(2 * b + 1) * n + 0][col] += 1
        col += 1
        
        # M5
        U[0 * m + 1][col] += 1
        U[0 * m + 2][col] += 1
        V[2 * p + 2 * b + 1][col] += 1
        W[2 * b * n + 0][col] -= 1
        W[2 * b * n + 1][col] += 1
        col += 1
        
        # M6
        U[1 * m + 1][col] += 1
        U[0 * m + 1][col] -= 1
        V[1 * p + 2 * b + 0][col] += 1
        V[1 * p + 2 * b + 1][col] += 1
        W[(2 * b + 1) * n + 1][col] += 1
        col += 1
        
        # M7
        U[0 * m + 2][col] += 1
        U[1 * m + 2][col] -= 1
        V[2 * p + 2 * b + 0][col] += 1
        V[2 * p + 2 * b + 1][col] += 1
        W[2 * b * n + 0][col] += 1
        col += 1

    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random test
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    # Compute C_decomposed
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ?
    # No, the tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Here indices are:
    # U is indexed by (i, j) where i is row of A, j is col of A.
    # V is indexed by (j, k) where j is row of B, k is col of B.
    # W is indexed by (k, i) where k is row of C, i is col of C.
    # Note: W[k*n+i] means row k, col i.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) ... No.
    # The standard form is:
    # C = sum_c U_c * V_c * W_c ?
    # Usually: C[i,k] = sum_c U[i, c] V[c, k] ...
    # But here we have 3 matrices U, V, W.
    # The tensor T is decomposed as sum_c u_c v_c w_c.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Wait, the indices must match.
    # U is (n*m, rank). Index (i*m+j, c).
    # V is (m*p, rank). Index (j*p+k, c).
    # W is (p*n, rank). Index (k*n+i, c).
    #
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # We want C[i,k] = sum_j A[i,j] B[j,k].
    # So we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{?}
    # Actually, the multiplication tensor T_{mul} has entries T_{i,j,k} = 1 if i,j,k are the indices for A[i,j]B[j,k] contributing to C[i,k]?
    # No, the tensor for matrix multiplication is defined such that:
    # C = A B  <=>  vec(C) = T (vec(A) \otimes vec(B))
    # The entries of T are T[i,j,k] where i is index of C, j of A, k of B?
    # Let's check the convention in the prompt: "W[k*n+i,r]".
    # This implies W is indexed by k (row of C) and i (col of C).
    # So the term is U[i,j,c] V[j,k,c] W[k,i,c].
    #
    # Let's verify with schoolbook:
    # U[i*m+j, c] = 1 for c=i*m+j*p+k? No, c is unique for each (i,j,k).
    # In schoolbook, for each (i,j,k), we have one term.
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Sum over c:
    # For a fixed i,k, sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Only terms where c corresponds to (i,j,k) are non-zero.
    # For such a term, U=1, V=1, W=1.
    # So the sum is sum_j 1 = m? No.
    # Wait.
    # The decomposition represents the tensor T such that:
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    #
    # We want the multiplication map.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a bilinear map.
    # The tensor T for this map has components T_{i,j,k} such that:
    # C[i,k] = sum_{j,l} A[i,j] B[l,k] T_{i,j,l,k}? No.
    #
    # Standard definition:
    # The multiplication tensor T_{n,m,p} is in V^* \otimes W^* \otimes Z.
    # T = sum_{i,j,k} e_i \otimes f_j \otimes g_k where e_i is basis for C, f_j for A, g_k for B?
    # No, T(e_i, f_j, g_k) = 1 if i,j,k form a valid product term?
    #
    # Let's stick to the indices.
    # If we plug in A and B:
    # sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_l B[l,k] V[l*p+k, c]) * W[k*n+i, c]
    # This doesn't look like C[i,k].
    #
    # Correct interpretation:
    # The rank-1 tensor is U_c V_c W_c.
    # The value is U_c[i,j] V_c[j,k] W_c[k,i].
    # Summing over c gives the tensor T[i,j,k].
    # Then C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]?
    # No, T[i,j,k] should be 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In matrix multiplication, A[i,j] and B[j,k] multiply to contribute to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k? No, that would be a constant.
    #
    # The tensor T is defined by T(i, j, k) = 1 if the indices match?
    # Actually, the tensor for matrix multiplication is:
    # T = sum_{i,j,k} E_{ik} \otimes E_{ij} \otimes E_{jk}
    # Where E_{ik} is the matrix unit for C, E_{ij} for A, E_{jk} for B.
    #
    # So T[i,k, j] ?
    # The prompt says W[k*n+i]. This is C index (k,i).
    # U is A index (i,j).
    # V is B index (j,k).
    #
    # So the term is U[i,j] * V[j,k] * W[k,i].
    # Summing over c:
    # S[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    #
    # We want S[i,j,k] = 1 for all i,j,k?
    # If S[i,j,k] = 1, then:
    # sum_j S[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k] = C[i,k].
    #
    # So the goal is to have S[i,j,k] = 1 for all i,j,k.
    #
    # Let's check my construction.
    # Part 1:
    # For term c (i,k), U[i,0]=1, V[0,k]=1, W[k,i]=1.
    # S[i,0,k] gets 1.
    # But we need S[i,j,k]=1 for all j.
    # Here j is fixed to 0.
    # So S[i,0,k] += 1.
    #
    # Part 2:
    # For block b, term M1 (c).
    # U[0,1]+=1, U[1,2]+=1.
    # V[1,2b]+=1, V[2,2b+1]+=1.
    # W[2b,0]+=1, W[2b+1,1]+=1.
    #
    # This contributes to S[0,1,2b], S[0,1,2b+1], S[1,2,2b], S[1,2,2b+1].
    # Specifically:
    # S[0,1,2b] += U[0,1] V[1,2b] W[2b,0] = 1*1*1 = 1.
    # S[0,1,2b+1] += U[0,1] V[1,2b+1] W[2b+1,0] ??
    # Wait, W index is k*n+i.
    # For W[2b,0], k=2b, i=0.
    # For W[2b+1,1], k=2b+1, i=1.
    #
    # Let's check S[0,1,2b+1].
    # U[0,1] is involved.
    # V[1,2b+1] is involved?
    # In M1, V[1,2b] and V[2,2b+1] are involved.
    # So V[1,2b+1] is 0 in M1.
    # So S[0,1,2b+1] gets 0 from M1.
    #
    # This is getting complicated to verify by hand.
    # I will write a quick verification in the code and print if it fails.
    # If it fails, I will return the schoolbook 48.

    # Verify
    S = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i, j, k] += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    # Check if S[i,j,k] == 1 for all i,j,k
    if not np.allclose(S, np.ones((n, m, p))):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
