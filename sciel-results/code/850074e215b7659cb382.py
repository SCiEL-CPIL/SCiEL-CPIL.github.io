"""
Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
Uses a hybrid approach: Strassen-style decomposition for a 2x2x2 sub-block
to save 1 rank, and schoolbook for the remaining terms.
"""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will use Strassen for the sub-block where i in {0,1}, j in {0,1}, k in {0,1}.
    # This sub-block has 8 terms in schoolbook, but can be done in 7.
    # The remaining terms are (n*m*p - 8) = 52.
    # Total rank = 7 + 52 = 59.
    
    # Let's verify if we can just use schoolbook for everything else.
    # Yes, the tensor decomposition is a direct sum of the sub-tensors if we partition the indices.
    # However, the indices i, j, k are not independent in the matrix multiplication tensor in a way that allows simple direct sum unless we are careful.
    # The tensor is T[i, j, k] = 1 if we are looking at the term a_ij * b_jk.
    # Actually, the matrix multiplication tensor T is defined by T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The rank of the tensor is the minimum number of rank-1 tensors that sum to T.
    # The standard schoolbook decomposition uses n*m*p rank-1 tensors:
    # T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C
    # where e_i^A has 1 at position i in the A-index space? No.
    # Let's look at the standard decomposition.
    # U has shape (n*m, rank). V has shape (m*p, rank). W has shape (p*n, rank).
    # The decomposition is: sum_{r=0}^{rank-1} U[:, r] \otimes V[:, r] \otimes W[:, r] = T
    # T is a 3-way tensor of dimensions (n*m, m*p, p*n).
    # Wait, the dimensions in the contract are:
    # U: (n*m, rank) -> indices (i, j) for A
    # V: (m*p, rank) -> indices (j, k) for B
    # W: (p*n, rank) -> indices (k, i) for C
    # So the tensor T has dimensions (n*m, m*p, p*n).
    # T[(i, j), (j', k), (k', i')] is non-zero only if i=i', j=j', k=k'.
    # Actually, T[(i, j), (j, k), (k, i)] = 1.
    # This is a "diagonal" tensor in the 3-way space.
    
    # The rank of this specific tensor (the matrix multiplication tensor for n x m by m x p) is known to be n*m*p for small dimensions, but can be lower for larger ones or with specific structures.
    # For 3x4 by 4x5, the schoolbook rank is 60.
    # Can we do better?
    # If we can find a sub-tensor that has lower rank than its size, we can replace it.
    # The 2x2 by 2x2 matrix multiplication tensor has rank 7 (Strassen).
    # The full tensor is not a direct sum of sub-tensors in the standard tensor product sense because the indices are linked.
    # However, we can partition the set of (i, j, k) tuples.
    # Let S be the set of all (i, j, k).
    # If we partition S into S1 and S2, the tensor T is the sum of T1 and T2, where T1 is supported on S1 and T2 on S2.
    # The rank of T is at most rank(T1) + rank(T2).
    # If we choose S1 to be the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}), then T1 is the 2x2x2 matrix multiplication tensor (embedded in the larger space).
    # The rank of T1 is 7.
    # T2 is the rest of the tensor, which has 52 non-zero entries.
    # The rank of T2 is at most 52 (schoolbook).
    # So the total rank is at most 7 + 52 = 59.
    
    # Let's implement this.
    # We need to construct U, V, W for rank 59.
    # Columns 0-6: Strassen decomposition for the 2x2x2 block.
    # Columns 7-58: Schoolbook for the remaining 52 terms.
    
    # Strassen decomposition for 2x2x2:
    # Let the indices for the sub-block be i in {0,1}, j in {0,1}, k in {0,1}.
    # We need to express the 8 terms as a sum of 7 rank-1 tensors.
    # The standard Strassen algorithm computes 7 products M1..M7.
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to our U, V, W matrices.
    # For the sub-block, the indices in U are (i, j) where i, j in {0,1}.
    # In the full U matrix, these are at rows:
    # i=0, j=0 -> row 0
    # i=0, j=1 -> row 1
    # i=1, j=0 -> row 4 (since m=4, row = i*m + j = 1*4 + 0 = 4)
    # i=1, j=1 -> row 5
    
    # For V, indices (j, k) where j, k in {0,1}.
    # In the full V matrix, these are at rows:
    # j=0, k=0 -> row 0
    # j=0, k=1 -> row 1
    # j=1, k=0 -> row 5 (since p=5, row = j*p + k = 1*5 + 0 = 5)
    # j=1, k=1 -> row 6
    
    # For W, indices (k, i) where k, i in {0,1}.
    # In the full W matrix, these are at rows:
    # k=0, i=0 -> row 0
    # k=0, i=1 -> row 3 (since n=3, row = k*n + i = 0*3 + 1 = 1? No, row = k*n + i. Wait, the contract says W[k*n+i, r].
    # So for k=0, i=0 -> row 0.
    # k=0, i=1 -> row 1.
    # k=1, i=0 -> row 3.
    # k=1, i=1 -> row 4.
    
    # Let's re-verify the W indexing.
    # Contract: W[k*n+i, r].
    # n=3.
    # k=0, i=0 -> 0*3+0 = 0.
    # k=0, i=1 -> 0*3+1 = 1.
    # k=1, i=0 -> 1*3+0 = 3.
    # k=1, i=1 -> 1*3+1 = 4.
    
    # Now, let's define the 7 rank-1 tensors for the sub-block.
    # Each rank-1 tensor is defined by a column in U, V, W.
    # We will use columns 0 to 6 for Strassen.
    
    # M1: (A00 + A11) * (B00 + B11)
    # U column 0: A00 (row 0) = 1, A11 (row 5) = 1. Others 0.
    # V column 0: B00 (row 0) = 1, B11 (row 6) = 1. Others 0.
    # W column 0: C00 (row 0) = 1, C11 (row 4) = 1. Others 0.
    # Check: C00 gets +1, C11 gets +1. Correct.
    
    U[0][0] = 1
    U[5][0] = 1
    V[0][0] = 1
    V[6][0] = 1
    W[0][0] = 1
    W[4][0] = 1
    
    # M2: (A10 + A11) * B00
    # U column 1: A10 (row 4) = 1, A11 (row 5) = 1.
    # V column 1: B00 (row 0) = 1.
    # W column 1: C10 (row 3) = 1. (C10 is k=1, i=1? No. C10 is i=1, k=0.
    # Wait, C_{ik}. i=1, k=0.
    # W index is k*n + i. So k=0, i=1 -> row 1.
    # Let's re-check C10.
    # C10 means row 1, col 0 of C.
    # So i=1, k=0.
    # W[k*n+i] = W[0*3+1] = W[1].
    # So W[1][1] = 1.
    # Check: C10 gets +1. Correct.
    
    U[4][1] = 1
    U[5][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00 * (B01 - B11)
    # U column 2: A00 (row 0) = 1.
    # V column 2: B01 (row 1) = 1, B11 (row 6) = -1.
    # W column 2: C01 (row 1) = 1, C11 (row 4) = 1.
    # C01: i=0, k=1. W[1*3+0] = W[3].
    # C11: i=1, k=1. W[1*3+1] = W[4].
    # So W[3][2] = 1, W[4][2] = 1.
    
    U[0][2] = 1
    V[1][2] = 1
    V[6][2] = -1
    W[3][2] = 1
    W[4][2] = 1
    
    # M4: A11 * (B10 - B00)
    # U column 3: A11 (row 5) = 1.
    # V column 3: B10 (row 5) = 1, B00 (row 0) = -1.
    # W column 3: C10 (row 1) = 1, C00 (row 0) = 1.
    # C10: i=1, k=0. W[1].
    # C00: i=0, k=0. W[0].
    
    U[5][3] = 1
    V[5][3] = 1
    V[0][3] = -1
    W[1][3] = 1
    W[0][3] = 1
    
    # M5: (A00 + A01) * B11
    # U column 4: A00 (row 0) = 1, A01 (row 1) = 1.
    # V column 4: B11 (row 6) = 1.
    # W column 4: C00 (row 0) = -1, C01 (row 3) = 1.
    
    U[0][4] = 1
    U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1
    W[3][4] = 1
    
    # M6: (A10 - A00) * (B00 + B01)
    # U column 5: A10 (row 4) = 1, A00 (row 0) = -1.
    # V column 5: B00 (row 0) = 1, B01 (row 1) = 1.
    # W column 5: C11 (row 4) = 1.
    
    U[4][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[4][5] = 1
    
    # M7: (A01 - A11) * (B10 + B11)
    # U column 6: A01 (row 1) = 1, A11 (row 5) = -1.
    # V column 6: B10 (row 5) = 1, B11 (row 6) = 1.
    # W column 6: C00 (row 0) = 1.
    
    U[1][6] = 1
    U[5][6] = -1
    V[5][6] = 1
    V[6][6] = 1
    W[0][6] = 1
    
    # Now, for the remaining 52 terms, we use schoolbook.
    # We iterate through all i, j, k and skip the ones already handled.
    # Handled: i in {0,1}, j in {0,1}, k in {0,1}.
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # The total rank should be 59.
    # Let's verify col == 59.
    # 8 terms in 2x2x2 block. 60 - 8 = 52.
    # 7 + 52 = 59.
    # So col should be 59.
    
    # We need to truncate U, V, W to rank 59.
    # Currently they have size 60.
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
