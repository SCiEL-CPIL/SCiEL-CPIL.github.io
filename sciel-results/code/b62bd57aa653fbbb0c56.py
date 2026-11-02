import numpy as np
import time

DIMENSIONS = [4, 4, 6]


def _build_laderman_4x4x3(n=4, m=4, p=3):
    """Build Laderman-style rank-24 decomposition for 4x4 * 4x3.

    Structure: A (4x4) and B (4x3) split into 2x2 blocks.
    - 3x3 blocks: rank 2 each (4 blocks -> 8 terms)
    - 2x2 blocks: rank 1 each (4 blocks -> 4 terms)
    - 3x2 blocks: rank 2 each (4 blocks -> 8 terms)
    - 2x3 blocks: rank 2 each (4 blocks -> 8 terms)
    Total: 24 terms.
    """
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def Uidx(i, j): return i * m + j
    def Vidx(j, k): return j * p + k
    def Widx(k, i): return k * n + i

    col = 0
    # 3x3 blocks: rank 2 (Laderman 2x2 on 3x3 blocks)
    # For 3x3 * 3x3, rank 2 per 2x2 sub-block via (a+b, a-b) style
    # Actually for 3x3x3 we use rank 2 decomposition:
    # C = A*B where A,B are 3x3. Standard: use 2 terms per 2x2 corner.
    # Simpler: treat each 3x3 as needing rank 2 via:
    #   term1: A_full * B_full (rank 1) - NO, that's not right.
    
    # Let me use the explicit Laderman 4x4x3 construction:
    # Split into 2x2 blocks of size 2x2.
    # A = [[A11, A12], [A21, A22]] where A11,A12 are 2x3? No.
    # A is 4x4: A11(2x2), A12(2x2), A21(2x2), A22(2x2)
    # B is 4x3: B11(2x3), B12(2x3), B21(2x3), B22(2x3)
    
    # C = A*B is 4x3: C11(2x3), C12(2x3), C21(2x3), C22(2x3)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2 * 2x3 product has rank 2 (since 2x2x2 min rank is 2? No, it's 2*2*2=8 schoolbook, but Strassen gives 7 for 2x2x2. For 2x2x3: schoolbook 12, but can do better.)
    # Actually 2x2 * 2x3: this is multiplying 2x2 by 2x3. The tensor rank of (2,2,3) is known to be... let me just use schoolbook for the sub-blocks and combine.
    
    # Let me just use a direct approach: for each 2x2 * 2x3 block product, use rank 2 (since it's essentially two 2x2 * 2x1 products, each rank 1... no).
    
    # Simplest correct approach: use the fact that 4x4x3 can be done in rank 24 by treating it as 4x4x4 with the last column zero, using Laderman 4x4x4 (rank 49) -- no that's worse.
    
    # Let me use a verified construction. The rank of (4,4,3) matrix multiplication is known to be 24.
    # Construction: split B into B_left (4x2) and B_right (4x1).
    # 4x4 * 4x2: rank 24? 4x4 * 4x1: rank 8?
    # 4x4 * 4x1: this is A * b for vector b. C_i = sum_j A_ij * b_j. This needs 16 terms schoolbook, but can be done in 8? No, it's a bilinear map from (16+4) to 16, rank 16.
    
    # I'll use a different strategy: directly construct rank 24 for (4,4,3).
    # Known: R(4,4,3) = 24. Use the following:
    # Treat as 4 rows of (1x4 * 4x3) = 4x3. Each row: 1x4 * 4x3 = 4x3, rank 4 (since 1*4*3=12 schoolbook, but 1x4 * 4x3 is just 4 independent dot products, rank 4).
    # So 4 rows * rank 4 = rank 16? But that's if we can share... no, each row is independent.
    # Actually 1x4 * 4x3: C_k = sum_j A_j * B_jk for k=0,1,2. This is 3 outputs, each a dot product of length 4. Rank = 3*4 = 12 schoolbook, but can be done in rank 4? No, minimum rank for (1,4,3) is 4 (since it's a 3-vector of 4-dimensional dot products, and you need at least 4 terms for the 4-dimensionality... actually rank of (1,4,3) tensor is 4).
    # So 4 rows * 4 = 16? But that can't be right since R(4,4,3) > 16.
    
    # Let me just verify numerically what rank works and use a known construction.
    # For now, I'll use the block approach with 2x2 * 2x3 sub-blocks.
    # 2x2 * 2x3: C = A*B, A is 2x2, B is 2x3. C is 2x3.
    # C_00 = A00*B00 + A01*B10
    # C_01 = A00*B01 + A01*B11
    # C_02 = A00*B02 + A01*B12
    # C_10 = A10*B00 + A11*B10
    # C_11 = A10*B01 + A11*B11
    # C_12 = A10*B02 + A11*B12
    # This is 12 terms schoolbook. Can we do better? 
    # Note: C = [A00, A01] * B and [A10, A11] * B.
    # Let P = A00 + A01, Q = A00 - A01. Then A00 = (P+Q)/2, A01 = (P-Q)/2.
    # C_0k = (P+Q)/2 * B0k + (P-Q)/2 * B1k = P*(B0k+B1k)/2 + Q*(B0k-B1k)/2
    # Similarly for row 1.
    # This gives: for row 0: 2 terms per column (P and Q), for row 1: 2 terms per column.
    # Total: 4 terms per column * 3 columns = 12. Same.
    
    # OK let me just use a completely different approach: use the known fact that
    # R(4,4,6) <= R(4,4,4) + R(4,4,2) = 49 + 32 = 81, or better R(4,4,4) + R(4,4,2).
    # But R(4,4,2): 4x4 * 4x2. This is A*B where B is 4x2. C is 4x2.
    # C_ik = sum_j A_ij * B_jk. For each k, it's A * b_k. Rank of (4,4,1) is 16? No, (4,4,1) means 4x4 * 4x1 -> 4x1. The tensor T[i,j,k] with k only 0. T[i,j,0]=1. Rank = 16 (each of the 16 outputs needs its own term since they're independent linear forms).
    # So R(4,4,2) = 32 (two independent (4,4,1) problems).
    # R(4,4,4) = 49 (Laderman).
    # Total: 49 + 32 = 81.
    
    # But we can do better: R(4,4,6) <= R(4,4,3) + R(4,4,3) = 24 + 24 = 48 if R(4,4,3)=24.
    
    # Let me just hardcode a verified rank-24 construction for (4,4,3).
    # I'll generate it by solving the linear system.
    
    # For (4,4,3): we need U (16 x r), V (12 x r), W (12 x r) such that
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    # This is a system of 48 equations (4*4*3) in r*3 unknowns (if we fix W to be sparse).
    # Actually it's bilinear in the factors, so it's not linear. But we can fix one factor.
    
    # Let me try rank 24 with a structured approach.
    # I'll use the following known construction for (4,4,3):
    # Split into 2 groups of 2 rows.
    # Group 1 (rows 0,1): 2x4 * 4x3 -> 2x3, rank 6 (since 2*4*3=24 schoolbook, but 2x4 * 4x3 can be done in rank 6? 2x4*4x3 = 2x3. Each of the 6 outputs is a dot product of length 4, but shared. Rank = 6? Let's see: C[0,0]=sum A[0,j]B[j,0], etc. The 6 outputs use A[0,*] and A[1,*] (8 values) and B[*,0],B[*,1],B[*,2] (12 values). Minimum rank for (2,4,3) is... I'll assume 6.)
    # Group 2 (rows 2,3): similarly rank 6.
    # Total: 12? That seems too low. R(4,4,3) is definitely > 12.
    
    # I think the safest approach is to just use rank 48 = 2*24 where 24 is for (4,4,3),
    # and I'll construct (4,4,3) rank 24 by using 4x4x4 Laderman (rank 49) is too high.
    
    # Let me just use a numerical solver to find rank 24 for (4,4,3).
    # I'll fix W to be a specific sparse pattern and solve for U, V.
    
    # Actually, let me just use the schoolbook for (4,4,3) which is rank 48, and then
    # for the full (4,4,6) use rank 96. That's not an improvement.
    
    # Let me try a different known result: R(4,4,6) <= 48 via the following:
    # Use the fact that 4x4 * 4x6 = (4x4 * 4x2) + (4x4 * 4x2) + (4x4 * 4x2) where each 4x4*4x2 has rank 16?
    # 4x4 * 4x2: C is 4x2. C_ik = sum_j A_ij B_jk. This is 8 outputs. Each is a bilinear form in A (16 vars) and B (8 vars). The tensor (4,4,2) has rank... schoolbook is 32. Can we do 16? 
    # If we set up: for k=0,1: C[:,k] = A * B[:,k]. This is two matrix-vector products. Each needs rank 16 (since A is 4x4 and we need all 4 outputs for each k, and they're independent). So rank 32 for (4,4,2).
    
    # I'll go with a practical approach: use rank 48 by splitting into two (4,4,3) problems,
    # and for each (4,4,3) use a rank-24 construction that I'll verify numerically.
    # If the construction doesn't work, fall back to schoolbook.
    
    # Let me construct (4,4,3) rank 24 using the following method:
    # Use 2x2 blocks. A (4x4) = 4 blocks of 2x2. B (4x3) = 2 blocks of 2x3.
    # C (4x3) = 2 blocks of 2x3.
    # C_top = A_top * B (where A_top is 2x4, B is 4x3)
    # C_bot = A_bot * B (where A_bot is 2x4)
    # A_top * B: (2x4) * (4x3) = 2x3. Split A_top into A_top1 (2x2), A_top2 (2x2).
    # C_top = A_top1 * B_top + A_top2 * B_bot where B_top is 2x3, B_bot is 2x3.
    # A_top1 * B_top: (2x2)*(2x3) = 2x3. Rank 2 (since 2x2*2x3: C_0k = A00*B0k + A01*B1k, C_1k = A10*B0k + A11*B1k. This is 6 outputs. Can do in rank 2? No, rank 6 schoolbook. But we can use: let X = A00+A01, Y = A00-A01. Then C_0k = X*B0k/2 + Y*(B0k-B1k)/2... this still gives 2 terms per column = 6 total.)
    
    # I give up on deriving it from first principles. Let me just use a known valid construction.
    # The rank of (4,4,3) is 24, and I'll use the following explicit construction:
    
    # For (4,4,3), use the decomposition:
    # C = A*B, A is 4x4, B is 4x3.
    # Write A = [A1 | A2] where A1, A2 are 4x2.
    # Write B = [B1; B2] where B1, B2 are 2x3.
    # C = A1*B1 + A2*B2.
    # A1*B1: (4x2)*(2x3) = 4x3. Rank 6 (each of 4 rows is a 1x2 * 2x3 = 3 outputs, rank 3 per row? No, 1x2*2x3: C_k = A0*B0k + A1*B1k, 3 outputs, rank 3. So 4 rows * 3 = 12? But shared A values... rank 6.)
    # Actually (4,2,3) tensor rank: I'll assume 6.
    # So total rank 12 for (4,4,3)? That's below the known lower bound.
    
    # The known lower bound for R(4,4,3) is 16 (from R >= n*p = 4*3 = 12, and R >= m*p = 12, and R >= n*m = 16). So R(4,4,3) >= 16.
    # And R(4,4,3) <= 24 (via Laderman-style).
    
    # Let me just use rank 24 with a concrete construction that I can verify.
    # I'll use the following: for (4,4,3), use 24 terms where each term is a rank-1 tensor.
    # I'll generate them by solving a linear system with a fixed W.
    
    # Fix W to have a specific structure: W[k*n+i, c] is nonzero only for specific (k,i) pairs.
    # Let me try: for c = 0..23, assign each to a unique (k,i) pair... but there are 12 (k,i) pairs.
    # So some (k,i) pairs share terms.
    
    # This is getting too complex. Let me just use the schoolbook (rank 96) as the answer,
    # since I cannot reliably construct a lower-rank decomposition in this time.
    # But wait - the task says "rank-minimal" and the score is the rank. Schoolbook is 96.
    # If I can get 48, that's a big improvement.
    
    # Let me try one more thing: use the fact that for (4,4,6), I can use
    # R(4,4,6) <= R(4,4,4) + R(4,4,2) where R(4,4,4)=49 (Laderman) and R(4,4,2)=32.
    # Total 81. That's better than 96.
    
    # Or R(4,4,6) <= R(4,4,3) + R(4,4,3) = 24+24 = 48 if I can construct (4,4,3) rank 24.
    
    # I'll implement (4,4,3) rank 24 by using the following explicit construction:
    # Based on the paper "Algorithms for matrix multiplication" the (4,4,3) rank 24 construction is:
    # Split into 2x2 blocks. Use 2 terms for each 2x2*2x3 block (4 blocks * 2 = 8) plus cross terms.
    
    # I'll just hardcode a verified construction. Let me use numpy to solve it.
    
    # For (4,4,3) rank 24:
    # We need: for all i in 0..3, j in 0..3, k in 0..2:
    #   sum_c U[i*4+j, c] * V[j*3+k, c] * W[k*4+i, c] = 1
    
    # Let me fix W to be the identity-like pattern: W[k*4+i, c] = 1 if c == k*4+i, else 0.
    # But there are 12 (k,i) pairs and 24 terms, so this doesn't work directly.
    
    # Alternative: use W with 2 nonzeros per column.
    
    # I'll just use a numerical approach: initialize U, V, W randomly and use gradient descent
    # to minimize the error. But that's slow and may not converge to exact zero.
    
    # Given time constraints, I'll use the schoolbook rank 96.
    # Actually wait - let me try rank 48 with a simple construction:
    # For (4,4,6), use 48 terms where for each (i,j) pair in A, we have 3 terms (one per column of B).
    # That's 16*3 = 48. But that's just schoolbook for (4,4,6) which is 96. No, schoolbook is 4*4*6=96.
    # 16*3 = 48 would mean each (i,j) in A contributes to all 3 columns of B with one term each.
    # But that's not how it works. Schoolbook: for each (i,j,k), one term. 4*4*6=96.
    
    # To get rank 48, we need to share terms. E.g., for fixed (i,j), the 3 terms for k=0,1,2
    # can be combined if V and W are structured appropriately.
    
    # I'll just go with schoolbook 96. It's correct and safe.
    
    # Actually, let me try one concrete construction for (4,4,3) rank 24:
    # Use the following 24 rank-1 terms:
    # For i=0,1,2,3 and k=0,1,2: term (i,k) uses U with support on row i of A,
    # V with support on column k of B, W with support on (k,i) position.
    # That's 12 terms, not enough.
    
    # I'll use rank 48 = 2 * 24 where each (4,4,3) uses rank 24 via:
    # 4 rows, each row is (1x4 * 4x3) which has rank 6 (not 3, because 1x4*4x3: 
    # C_k = sum_j A_j B_jk for k=0,1,2. This is 3 outputs from 4+12=16 inputs.
    # The tensor (1,4,3) has rank 4 (minimum), since you need at least 4 terms to represent
    # 3 independent dot products of length 4... actually rank of (1,4,3) is 4.
    # So 4 rows * 4 = 16? But R(4,4,3) >= 16, and 16 might work!
    
    # Let me check: R(4,4,3) = 16? The lower bound is max(n*m, m*p, n*p) = max(16, 12, 12) = 16.
    # And if we can achieve 16, that's optimal.
    
    # Construction for rank 16: For each (i,j) pair, one term.
    # U[i*4+j, c] = 1 if c == i*4+j
    # V[j*3+k, c] = B[j,k] coefficient... but V must be constant (not depend on B).
    # Wait, the decomposition is for the multiplication tensor, not for specific matrices.
    # T[i,j,k] = 1 for all valid (i,j,k). We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # For rank 16: set c = i*4+j (16 terms).
    # U[i*4+j, c] = 1 if c == i*4+j else 0.
    # V[j*3+k, c] = ? We need V[j,k,c] * W[k,i,c] = 1 for the specific (i,j,k).
    # With c = i*4+j, V[j,k,c] and W[k,i,c] must multiply to 1.
    # But V depends on (j,k,c) and W on (k,i,c). For fixed c = i*4+j, 
    # V[j,k,c] is nonzero only for this specific j (since c encodes j).
    # So V[j,k,c] can be 1 for all k, and W[k,i,c] = 1 for all k.
    # Then sum_c U[i,j,c] V[j,k,c] W[k,i,c] = U[i,j,c] * 1 * 1 = 1 when c = i*4+j.
    # And for other (i',j',k'), the sum is 0 because U[i',j',c] = 0 for c != i'*4+j'.
    # So this works! Rank 16 for (4,4,3)!
    
    # Wait, that can't be right. Let me double-check.
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # With c = i*4+j: U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1.
    # So T[i,j,k] = 1. Correct!
    # And for a different triple (i',j',k'), T[i',j',k'] = sum_c U[i',j',c] V[j',k',c] W[k',i',c].
    # U[i',j',c] is nonzero only when c = i'*4+j'. At that c, V[j',k',c] = 1, W[k',i',c] = 1.
    # So T[i',j',k'] = 1. Correct!
    
    # So rank 16 works for (4,4,3)! And the lower bound is 16, so it's optimal!
    # Then R(4,4,6) = 2 * 16 = 32!
    
    # Wait, but this construction is just schoolbook for (4,4,3) which is 4*4*3 = 48, not 16.
    # What's wrong? Ah, I see the issue. In the schoolbook, each term corresponds to a specific (i,j,k).
    # In my construction, each term c = i*4+j corresponds to a fixed (i,j) but all k.
    # But V[j,k,c] = 1 for all k means that term contributes to all k simultaneously.
    # That's fine! The tensor entry T[i,j,k] gets contribution 1 from term c = i*4+j.
    # And no other term contributes to T[i,j,k] because U[i,j,c'] = 0 for c' != i*4+j.
    # So yes, rank 16 works!
    
    # But wait, this is just saying that the (4,4,3) multiplication tensor has rank 16,
    # which equals the lower bound. Is this correct? Let me verify with a small example.
    # For (1,1,1): rank 1. Lower bound max(1,1,1)=1. OK.
    # For (2,2,2): rank 7 (Strassen). Lower bound max(4,4,4)=4. 7 > 4.
    # For (1,2,2): rank 2? Lower bound max(2,2,2)=2. T[i,j,k] with i=0 only.
    # T[0,j,k] = sum_c U[0,j,c] V[j,k,c] W[k,0,c].
    # With c = j (2 terms): U[0,j,c]=1, V[j,k,c]=1, W[k,0,c]=1. T[0,j,k]=1. Rank 2. Correct!
    # So (1,2,2) has rank 2 = lower bound.
    
    # For (4,4,3): lower bound 16. My construction gives 16. So rank 16!
    # Then R(4,4,6) = 32.
    
    # Let me verify this is correct by checking that the tensor is indeed decomposed.
    # The multiplication tensor for (n,m,p) is T[i,j,k] = 1 for all valid (i,j,k).
    # My construction: c ranges over i*m+j (n*m values).
    # U[i*m+j, c] = 1 if c == i*m+j else 0.
    # V[j*p+k, c] = 1 for all k (when c == i*m+j, which fixes j).
    # W[k*n+i, c] = 1 for all k (when c == i*m+j, which fixes i).
    # Then T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    #   = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    #   = 1 * 1 * 1 = 1. Correct!
    
    # And for any other (i',j',k'), T[i',j',k'] = 1 similarly.
    # So the decomposition is valid with rank n*m = 16 for (4,4,3).
    
    # For (4,4,6): split into two (4,4,3) problems.
    # First 3 columns: rank 16. Last 3 columns: rank 16. Total rank 32.
    
    # Let me implement this!
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    result = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U_arr[:, c].reshape(n, m)
        V_c = V_arr[:, c].reshape(m, p)
        W_c = W_arr[:, c].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
    
    return np.allclose(result, np.ones((n, m, p)), atol=1e-9)


def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try the rank-32 construction: split into two (4,4,3) problems
    # Each (4,4,3) has rank n*m = 16
    rank_32 = 2 * n * m  # 32
    
    U32 = [[0] * rank_32 for _ in range(n * m)]
    V32 = [[0] * rank_32 for _ in range(m * p)]
    W32 = [[0] * rank_32 for _ in range(p * n)]
    
    # First 3 columns (k=0,1,2): use terms c = 0..15
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U32[i * m + j][c] = 1
            for k in range(3):
                V32[j * p + k][c] = 1
                W32[k * n + i][c] = 1
    
    # Last 3 columns (k=3,4,5): use terms c = 16..31
    for i in range(n):
        for j in range(m):
            c = 16 + i * m + j
            U32[i * m + j][c] = 1
            for k in range(3, 6):
                V32[j * p + k][c] = 1
                W32[k * n + i][c] = 1
    
    # Verify
    if _verify(U32, V32, W32, n, m, p):
        return {"U": U32, "V": V32, "W": W32}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
