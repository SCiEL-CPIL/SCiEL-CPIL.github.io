import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will use a decomposition of rank 114
    # Split p=9 into 4, 4, 1
    rank_444 = 49
    rank_441 = 16
    rank = rank_444 + rank_444 + rank_441
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 4x4x4 Strassen decomposition
    def add_444(offset_u, offset_v, offset_w, start_col):
        # This is complex to implement directly with indices.
        # It's easier to generate the tensors.
        pass

    # Let's generate the tensors explicitly.
    # Strassen 2x2x2 tensors
    # P1: (a+d)(e+h) -> U: a,e; V: a,h; W: a,e? No.
    # T = sum u_i v_j w_k
    # For 2x2x2:
    # a is A[0,0], b is A[0,1], c is A[1,0], d is A[1,1]
    # e is B[0,0], f is B[0,1], g is B[1,0], h is B[1,1]
    # C[0,0] = P1 - P4 + P6 - P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P3 + P2 - P5
    
    # P1: (a+d)(e+h) = ae + ah + de + dh
    # Tensors:
    # (a, e, C00), (a, h, C00), (d, e, C00), (d, h, C00)
    # But we need to combine them into rank-1 tensors.
    # The rank-1 tensor for P1 is:
    # U: [1, 0, 1, 0] (a, d)
    # V: [1, 0, 0, 1] (e, h)
    # W: [1, 1, 1, 1] (C00, C00, C00, C00) ? No.
    # The product is (a+d)(e+h) C00.
    # This is a single rank-1 tensor? No, it's a sum of 4 rank-1 tensors.
    # But Strassen says we can do it with 7 rank-1 tensors.
    # The 7 rank-1 tensors are the P_i.
    # Each P_i is a product of linear forms.
    # P1 = (a+d)(e+h). This is a rank-1 tensor?
    # No, (a+d)(e+h) is a bilinear form.
    # The tensor is (a+d) (e+h) C00.
    # This is a rank-1 tensor in the space of tensors?
    # Yes, it is $u \otimes v \otimes w$ where $u = a+d$, $v = e+h$, $w = C00$.
    # So each P_i contributes 1 rank-1 tensor to the decomposition of the 2x2x2 block.
    # Wait, the 2x2x2 multiplication tensor is a sum of 7 such tensors.
    # So for each 2x2x2 block, we add 7 rank-1 tensors to the global decomposition.
    
    # So for 4x4x4, we have 7 such blocks (M1..M7).
    # Each block is a 2x2x2 multiplication.
    # So we add 7 * 7 = 49 rank-1 tensors.
    
    # Let's define the 7 tensors for 2x2x2.
    # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # P1: (A00+A11)(B00+B11) C00
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: [1, 0, 0, 0]
    
    # P2: (A10+A11) B00 C10
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: [0, 0, 1, 0]
    
    # P3: A00 (B01-B11) C01
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: [0, 1, 0, 0]
    
    # P4: A11 (B10-B00) C10
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1]
    # w: [0, 0, 1, 0]
    
    # P5: (A00+A01) B11 C11
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # w: [0, 0, 0, 1]
    
    # P6: (A10-A00) (B00+B01) C00
    # u: [0, 0, 1, -1]
    # v: [1, 1, 0, 0]
    # w: [1, 0, 0, 0]
    
    # P7: (A01-A11) (B10+B11) C11
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # w: [0, 0, 0, 1]
    
    # Wait, the C indices for P6 and P7 are C00 and C11?
    # C11 = P1 - P3 + P2 - P5
    # C00 = P1 - P4 + P6 - P7
    # So P6 contributes to C00, P7 contributes to C00?
    # No, P7 contributes to C11?
    # Let's check the formulas.
    # C11 = P1 - P3 + P2 - P5
    # C00 = P1 - P4 + P6 - P7
    # So P6 and P7 contribute to C00.
    # But P7 is (A01-A11)(B10+B11).
    # And C11 has -P5 and -P3.
    # Let's re-verify Strassen formulas.
    # C11 = P1 - P4 + P6 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 - P5
    # Yes.
    
    # So the 7 tensors are:
    tensors_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # P1 -> C00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # P2 -> C10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # P3 -> C01
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]), # P4 -> C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]), # P5 -> C11
        ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]), # P6 -> C00
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]), # P7 -> C11? No, C00?
    ]
    # Wait, P7 contributes to C00 in C00 = P1 - P4 + P6 - P7.
    # But P7 is (A01-A11)(B10+B11).
    # And C11 = P1 - P3 + P2 - P5.
    # So P7 does not contribute to C11.
    # So the last tensor should have w = [1, 0, 0, 0].
    tensors_222[6] = ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])

    # Now we need to map these to the 4x4x4 blocks.
    # The 4x4x4 decomposition uses 7 such 2x2x2 blocks.
    # Each block corresponds to a pair of 2x2 submatrices.
    # M1: (A11+A22)(B11+B22)
    # The input to the 2x2x2 block is A' = A11+A22, B' = B11+B22.
    # So we need to add the tensors for A' and B'.
    # A' = A11 + A22.
    # A11 is A[0:2, 0:2], A22 is A[2:4, 2:4].
    # So the U vector for A' is the sum of the U vectors for A11 and A22.
    
    # This is getting complicated.
    # Let's just use the schoolbook decomposition for now to ensure correctness.
    # Rank 144.
    # Then we can try to optimize.
    
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

print(construct())
