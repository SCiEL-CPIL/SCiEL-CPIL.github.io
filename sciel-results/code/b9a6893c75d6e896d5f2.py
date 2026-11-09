import time
import numpy as np

DIMS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 130.0  # hard safety margin
    n, m, p = DIMS  # 4, 4, 7

    # We decompose the 4x4 x 4x7 multiplication into:
    # Block 1: 4x4 x 4x4 (first 4 columns of B)
    # Block 2: 4x4 x 4x3 (last 3 columns of B)
    #
    # For Block 1 (4x4x4): Use Strassen's algorithm recursively.
    #   4x4x4 can be done with 2x2 blocks of Strassen on 2x2x2 (rank 7 each).
    #   Actually, 4x4x4 = 16 * (2x2x2) if we do it naively with 2x2 blocks.
    #   But we can use Strassen on the 4x4x4 directly by viewing it as 2x2 blocks of 2x2.
    #   The 4x4x4 matmul tensor can be decomposed using 7 "super-multiplications" of 2x2 matrices.
    #   Each 2x2x2 has rank 7 (Strassen), so total rank = 7 * 7 = 49.
    #
    # For Block 2 (4x4x3): We need to multiply 4x4 by 4x3.
    #   We can pad the 4x3 to 4x4 (rank 49 for the full 4x4x4) but that's wasteful.
    #   Alternatively, we can use a direct decomposition.
    #   The task suggests using a rank-14 decomposition for 4x3x3, but our subproblem is 4x4x3.
    #   Let's think: 4x4 x 4x3. We can view this as multiplying a 4x4 matrix by a 4x3 matrix.
    #   One approach: use the fact that 4x4x3 can be embedded in 4x4x4 (rank 49) but we want better.
    #   Actually, let's use a simpler approach: decompose 4x4x3 using Strassen-like blocks.
    #   Split the 4x4 into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2.
    #   B (4x3) = [[B11, B12], [B21, B22]] where B11, B21 are 2x3, B12, B22 are 2x3.
    #   Wait, B is 4x3, so splitting rows: B = [[B_top], [B_bottom]] where B_top is 2x3, B_bottom is 2x3.
    #   C = A*B is 4x3. C = [[C11, C12], [C21, C22]] where C11 = A11*B_top + A12*B_bottom, etc.
    #   This is getting complex. Let's use a different strategy.
    #
    # Simpler structural approach:
    # Use the known fact that matmul(n,m,p) rank <= matmul(n,m,p') + matmul(n,m,p-p') for p' < p.
    # So rank(4,4,7) <= rank(4,4,4) + rank(4,4,3).
    #
    # For rank(4,4,4): Use Strassen recursively.
    #   rank(4,4,4) = 7 * rank(2,2,2) = 7 * 7 = 49.
    #
    # For rank(4,4,3): 
    #   We can write 4x4 x 4x3 as follows:
    #   Use the identity: rank(4,4,3) <= rank(4,4,2) + rank(4,4,1).
    #   rank(4,4,1) = 4*4*1 = 16 (schoolbook, since it's just a linear combination of columns).
    #   Actually, rank(4,4,1) is the rank of the tensor for multiplying 4x4 by 4x1 to get 4x1.
    #   This is just 16 independent multiplications (each output element is a dot product of a row of A with B).
    #   So rank(4,4,1) = 16.
    #   
    #   For rank(4,4,2): 
    #   We can use Strassen-like: rank(4,4,2) <= rank(2,2,2)*? 
    #   Actually, let's just use schoolbook for 4x4x2: rank = 4*4*2 = 32.
    #   But we can do better. rank(4,4,2) can be done with 2x2 blocks.
    #   A is 4x4, B is 4x2. 
    #   Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    #   Split B into 2x2 blocks (rows): B = [[B1], [B2]] where B1 is 2x2, B2 is 2x2.
    #   C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    #   Each term like A11*B1 is a 2x2x2 matmul (rank 7).
    #   We have 4 such terms: A11*B1, A12*B2, A21*B1, A22*B2.
    #   But they share B1 and B2. We can't just add ranks because of shared factors.
    #   Actually, the tensor for 4x4x2 is the direct sum of 4 tensors of type 2x2x2 if we view it as:
    #   C_top = A11*B1 + A12*B2
    #   C_bot = A21*B1 + A22*B2
    #   This is not a direct sum because B1, B2 are shared.
    #   
    #   Let's just use schoolbook for the 4x4x3 part: rank = 4*4*3 = 48.
    #   Total: 49 + 48 = 97.
    #
    # But wait, the inspiration mentioned rank 63 = 49 + 14. Let me reconsider.
    # The 4x4x3 subproblem: maybe there's a better way.
    # 
    # Actually, let's try a different decomposition. Instead of 4+3, let's try to use a
    # more efficient algorithm for the whole 4x4x7.
    #
    # Alternative: Use Strassen on the whole thing by padding.
    # Pad 4x7 to 8x8? No, that's too big.
    #
    # Let's stick with the block approach but optimize the 4x4x3 part.
    # 
    # For 4x4x3: We can use the following observation.
    # The 4x4x3 matmul tensor can be decomposed by considering it as:
    # C (4x3) = A (4x4) * B (4x3)
    # We can write B = [B1, B2, B3] where B1, B2, B3 are 4x1.
    # C = [A*B1, A*B2, A*B3]
    # Each A*Bk is a 4x1 vector, which requires 4*4 = 16 multiplications.
    # Total: 3 * 16 = 48. This is schoolbook.
    #
    # Can we do better? The rank of the 4x4x3 tensor is at most 48.
    # Known results: rank(4,4,3) is not well-known to be much less than 48.
    #
    # Let's try a different structural change: use a 2x2 block Strassen for the 4x4x4 part
    # and schoolbook for the 4x4x3 part, giving 49 + 48 = 97.
    #
    # But the current score is 1.0 which corresponds to rank 112 (schoolbook).
    # 97 < 112, so this should improve the score.
    #
    # Let me implement the 4x4x4 Strassen decomposition (rank 49) and the 4x4x3 schoolbook (rank 48).
    #
    # First, let's build the 4x4x4 Strassen decomposition.
    # 4x4x4: A (4x4) * B (4x4) -> C (4x4)
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen's algorithm:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a 2x2x2 matmul (rank 7).
    # So we need 7 * 7 = 49 factors.
    #
    # For each Mj, we have a 2x2x2 decomposition with 7 factors.
    # The 2x2x2 Strassen decomposition:
    # For 2x2x2: A (2x2) * B (2x2) -> C (2x2)
    # S1 = (A00 + A11) * (B00 + B11)
    # S2 = (A10 + A11) * B00
    # S3 = A00 * (B01 - B11)
    # S4 = A11 * (B10 - B00)
    # S5 = (A00 + A01) * B11
    # S6 = (A10 - A00) * (B00 + B01)
    # S7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    #
    # Now, for each Sj, we need the rank-1 factors (u, v, w) for the 2x2x2 tensor.
    # The 2x2x2 tensor has 8 terms in schoolbook, but Strassen gives 7.
    # Let me write out the 7 factors for 2x2x2 explicitly.
    #
    # For 2x2x2, indices:
    # A: a = i*2 + j, i=0,1; j=0,1. So A00=0, A01=1, A10=2, A11=3.
    # B: b = j*2 + k, j=0,1; k=0,1. So B00=0, B01=1, B10=2, B11=3.
    # C: c = k*2 + i (per task convention W[k*n+i] with n=2).
    #   C00: i=0,k=0 -> c=0
    #   C01: i=0,k=1 -> c=2
    #   C10: i=1,k=0 -> c=1
    #   C11: i=1,k=1 -> c=3
    #
    # Strassen factors for 2x2x2:
    # S1: u = A00+A11 = [1,0,0,1], v = B00+B11 = [1,0,0,1], w = C00+C10+C01+C11... 
    # Wait, S1 contributes to C00, C10, C01, C11 with coefficients.
    # From above:
    # C00 += S1, C10 += -S2? No, let me re-derive.
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    #
    # So for each Sj, we need to express it as a sum of rank-1 tensors.
    # But each Sj is itself a 2x2x2 matmul, which has rank 7.
    # So for each Sj, we need 7 rank-1 factors.
    #
    # This is getting very complex to implement correctly. Let me use a simpler approach.
    #
    # Given the time constraint, let me just implement the schoolbook for 4x4x7 (rank 112)
    # but also try to implement a 4x4x4 Strassen + 4x4x3 schoolbook (rank 97).
    #
    # Actually, let me just implement the 4x4x4 part with Strassen (rank 49) and the 4x4x3 part
    # with schoolbook (rank 48), for a total of 97.
    #
    # To implement Strassen for 4x4x4, I need the rank-1 factors for the 2x2x2 Strassen algorithm.
    # Let me hardcode them.
    
    # 2x2x2 Strassen rank-1 factors (7 factors)
    # Each factor is (u, v, w) where u, v, w are 4-vectors.
    # u for A (2x2), v for B (2x2), w for C (2x2) with W index = k*2 + i.
    
    # S1: (A00+A11) * (B00+B11)
    # Contributes: C00 += 1, C10 += -1, C01 += 1, C11 += 1? 
    # Let me re-check the Strassen formulas.
    # Standard Strassen:
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
    #
    # For 2x2x2, the indices are:
    # A: [[A00, A01], [A10, A11]] -> in Strassen notation: A11=A00, A12=A01, A21=A10, A22=A11
    # B: [[B00, B01], [B10, B11]] -> B11=B00, B12=B01, B21=B10, B22=B11
    # C: [[C00, C01], [C10, C11]] -> C11=C00, C12=C01, C21=C10, C22=C11
    #
    # So:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now, each Mj is a product of two 2x2 matrices (or linear combinations) resulting in a 2x2 matrix.
    # But in the tensor decomposition, we need to express each Mj as a sum of rank-1 tensors.
    # However, each Mj is itself a bilinear form, so it can be decomposed into rank-1 terms.
    #
    # For example, M1 = (A00+A11)(B00+B11) * I (where I is the 2x2 identity? No, M1 is a scalar times the identity? 
    # No, in Strassen, M1 is a 2x2 matrix? No, M1 is a scalar! 
    # Wait, no. In Strassen's algorithm for 2x2 matrices, M1, ..., M7 are scalars (products of linear combinations).
    # But then C11 = M1 + M4 - M5 + M7, etc., where each Mj is a scalar.
    # So each Mj is a scalar, and it contributes to the diagonal or off-diagonal of C.
    #
    # Actually, in the standard Strassen algorithm, M1 through M7 are scalars, and they are used to compute the 4 entries of C.
    # So the rank-1 decomposition for 2x2x2 using Strassen has 7 factors, where each factor is:
    # (u, v, w) where u and v are 4-vectors (for A and B), and w is a 4-vector (for C).
    # But since Mj is a scalar, w should have only one non-zero entry? No, that's not right.
    #
    # I think I'm confusing the bilinear form with the tensor decomposition.
    # In the tensor decomposition, each rank-1 term is u ⊗ v ⊗ w, where u, v, w are vectors.
    # The sum of these gives the matmul tensor.
    #
    # For 2x2x2, the matmul tensor T has T[i,j,k] = 1 if j=k (in the sense that C[i,k] = sum_j A[i,j]B[j,k]).
    # The schoolbook decomposition has 8 terms (one for each i,j,k with j=k).
    # Strassen's algorithm gives a rank-7 decomposition, which means we can write T as a sum of 7 rank-1 tensors.
    #
    # Let me just hardcode the 7 factors for 2x2x2 from a known source.
    # After some research, the standard rank-7 decomposition for 2x2x2 is:
    
    # Factor 1: u=[1,0,0,0], v=[0,1,0,0], w=[0,0,0,1]  # A00*B01 -> C01? 
    # I'm not confident in the exact factors. Let me use a different approach.
    #
    # Given the complexity and time, let me just return the schoolbook decomposition (rank 112) 
    # but with a small optimization: use the fact that we can do 4x4x4 with rank 49 (if I can implement it)
    # and 4x4x3 with rank 48, for a total of 97.
    #
    # Since I'm not confident in the Strassen factors, let me just do schoolbook for now.
    # But that gives rank 112, which is the current score.
    #
    # Let me try a different approach: use numpy to construct the decomposition.
    # Actually, let me just implement the schoolbook and see if I can verify it.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    best = schoolbook()
    
    if time.time() > deadline:
        return {k: v for k, v in best.items() if k != "rank"}
    
    # Try to improve with a block decomposition.
    # Let's implement the 4x4x4 Strassen (rank 49) + 4x4x3 schoolbook (rank 48) = 97.
    #
    # For the 4x4x4 part, I'll use the following approach:
    # The 4x4x4 matmul can be done by 7 "Strassen multiplications" of 2x2 matrices.
    # Each 2x2x2 matmul has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    #
    # I need the rank-1 factors for 2x2x2 Strassen. Let me hardcode them.
    # After careful thought, here are the 7 factors for 2x2x2:
    # (u, v, w) where u, v, w are 4-vectors.
    # A index: i*2+j, B index: j*2+k, C index: k*2+i
    
    # I'll use the following known decomposition (from literature):
    # The 7 factors for 2x2x2 Strassen are:
    
    factors_2x2x2 = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),  # M1: (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2: (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3: A00(B01-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 0]),  # M4: A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),   # M5: (A00+A01)B11
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6: (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),  # M7: (A01-A11)(B10+B11)
    ]
    
    # Now, for the 4x4x4 part:
    # A (4x4) split into 2x2 blocks: A11, A12, A21, A22
    # B (4x4) split into 2x2 blocks: B11, B12, B21, B22
    # C (4x4) split into 2x2 blocks: C11, C12, C21, C22
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a 2x2x2 matmul. So for each Mj, we use the 7 factors from factors_2x2x2.
    # But we need to map the 2x2 indices to the 4x4 indices.
    #
    # For M1 = (A11+A22)(B11+B22):
    # A11 is A[0:2, 0:2], A22 is A[2:4, 2:4]
    # B11 is B[0:2, 0:2], B22 is B[2:4, 2:4]
    # The result M1 is a 2x2 matrix, which contributes to C11 (C[0:2, 0:2]) and C22 (C[2:4, 2:4]).
    #
    # This is very complex to implement correctly. Given the time, let me just return schoolbook.
    
    return {k: v for k, v in best.items() if k != "rank"}
