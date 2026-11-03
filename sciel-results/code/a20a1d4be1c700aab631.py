import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ]

    # Helper to add a term
    def add_term(offset, a_idx, b_idx, w_idx, u_val, v_val, w_val):
        U[a_idx][offset] = u_val
        V[b_idx][offset] = v_val
        W[w_idx][offset] = w_val

    # A indices: i*m + j
    # B indices: j*p + k
    # W indices: k*n + i

    # A block indices (2x2 blocks of A):
    # A00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # A01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # A10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # A11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15

    # B block indices (2x2 blocks of B):
    # B00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # B01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # B10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # B11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15

    # W block indices (2x2 blocks of C):
    # C00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # C01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # C10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # C11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15

    # We will build the decomposition for 4x4x4 (rank 49) and then extend to 4x4x7.
    # For 4x4x4, we use the Kronecker product of Strassen 2x2x2 with itself? No, that gives rank 49.
    # Actually, the rank of 4x4x4 is at most 49 (using Strassen on 2x2 blocks).
    # The 49 terms come from the 7 terms of Strassen applied to the 2x2 blocks, but each term involves products of 2x2 matrices.
    # Wait, if we view 4x4x4 as (2x2) x (2x2) x (2x2), the rank is the tensor rank of the Kronecker product of the 2x2x2 tensor with itself?
    # No, the matrix multiplication tensor of size 4x4x4 is the tensor product of the 2x2x2 tensor with itself?
    # Let T be the 2x2x2 tensor. The 4x4x4 tensor is T \otimes T.
    # The rank of T \otimes T is at most rank(T) * rank(T) = 49.
    # So we can get rank 49 for 4x4x4.

    # How to construct it?
    # The 49 terms are indexed by (r1, r2) where r1, r2 in 0..6.
    # For each term (r1, r2):
    # U has non-zero entries at positions corresponding to A indices.
    # A is 4x4. We can view A as a 2x2 block matrix where each block is 2x2.
    # A = [[A00, A01], [A10, A11]]
    # The term (r1, r2) corresponds to the product of the r1-th term of Strassen on the "outer" 2x2 and the r2-th term on the "inner" 2x2?
    # Actually, the tensor T \otimes T has indices (i1, i2, j1, j2, k1, k2) where i1,i2,j1,j2,k1,k2 are 0 or 1.
    # The value is T[i1, j1, k1] * T[i2, j2, k2].
    # So for a fixed (r1, r2), the term is:
    # sum_{i1,j1,k1} U2[i1,r1] * sum_{i2,j2,k2} V2[j2,r2] * W2[k1,r1] * W2[k2,r2] * A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2]
    # Wait, the indices are:
    # A index: i = i1*2 + i2, j = j1*2 + j2.
    # B index: j = j1*2 + j2, k = k1*2 + k2.
    # C index: k = k1*2 + k2, i = i1*2 + i2.
    
    # So for term (r1, r2):
    # U[i1*2+i2, j1*2+j2] gets contribution U2[i1,r1] * V2[j2,r2]? No.
    # The term in the decomposition of T \otimes T is:
    # (sum_{i1,j1,k1} U2[i1,r1] V2[j1,r1] W2[k1,r1] A[i1,j1] B[j1,k1]) * (sum_{i2,j2,k2} U2[i2,r2] V2[j2,r2] W2[k2,r2] A[i2,j2] B[j2,k2])
    # This is not a rank-1 term in the 4x4x4 tensor. It's a product of two rank-1 terms.
    # So the 49 terms are not simply rank-1.
    
    # Actually, the rank of T \otimes T is not necessarily 49. It is at most 49, but it might be less.
    # However, we can use the 49 terms as a valid decomposition, even if they are not minimal.
    # But wait, the 49 terms I described are not rank-1 terms in the 4x4x4 tensor.
    # A rank-1 term in the 4x4x4 tensor is of the form u_i * v_j * w_k for all i,j,k.
    # The product of two rank-1 terms is not a rank-1 term.
    
    # So the Kronecker product approach does not directly give a rank-49 decomposition.
    # The rank of 4x4x4 is known to be at most 49, but the construction is non-trivial.
    
    # For now, let's use the schoolbook for 4x4x4 (rank 64) and then extend to 4x4x7.
    # 4x4x4 schoolbook: rank 64.
    # 4x4x3 schoolbook: rank 48.
    # Total: 112.
    
    # But we want to do better.
    # Let's use Strassen for the 4x4x4 part.
    # The rank of 4x4x4 is at most 49.
    # We can hardcode the 49 terms.
    
    # Alternatively, we can use the following approach:
    # For 4x4x4, use the 7 terms of Strassen on the 2x2 blocks.
    # Each term involves multiplying 2x2 matrices.
    # But we need to decompose the 2x2 matrix multiplication into rank-1 terms.
    # So each of the 7 terms expands into 4 rank-1 terms (schoolbook for 2x2x2).
    # Total: 7 * 4 = 28 terms? No, that's not right.
    
    # Actually, the standard way to get rank 49 for 4x4x4 is to use the 7 terms of Strassen, and for each term, use the 7 terms of Strassen for the 2x2 blocks? No.
    
    # Let's just use the schoolbook for simplicity and correctness, but with a better structure.
    # For 4x4x7, we can split B into 4x4 and 4x3.
    # 4x4x4: rank 64 (schoolbook)
    # 4x4x3: rank 48 (schoolbook)
    # Total: 112.
    
    # But we can do better for 4x4x4.
    # Let's use the known rank-49 decomposition.
    # I will hardcode it.
    
    # For now, let's return the schoolbook to be safe.
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
