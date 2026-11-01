"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Recursive Strassen decomposition of rank 7^2 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for 2x2
    S = [
        (1, 1), (0, 1), (1, 0), (0, 1),
        (1, 0), (1, 0), (1, 0), (0, 1)
    ]
    # For each (r, s) pair, we need U[i*m+j, r*7+s], V[j*p+k, r*7+s], W[k*n+i, r*7+s]
    # The 7 terms for Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M2 + M4 - M6
    # C12 = M5 + M6
    # C21 = M3 + M4
    # C22 = M1 - M3 - M5 + M7

    # For each of the 49 rank-1 terms, we have (i_block, j_block, k_block, r, s)
    # where i_block, j_block, k_block in {0,1} and r, s in {0,1}
    # The coefficient in U[i*m+j, c] depends on (i_block, j_block, r)
    # The coefficient in V[j*p+k, c] depends on (j_block, k_block, s)
    # The coefficient in W[k*n+i, c] depends on (k_block, i_block, r, s)

    # Let's build the 7-term Strassen decomposition as a list of (u_coeffs, v_coeffs, w_coeffs)
    # where u_coeffs[i_block, j_block] gives the coefficient for U[i*m+j, c]
    # v_coeffs[j_block, k_block] gives the coefficient for V[j*p+k, c]
    # w_coeffs[k_block, i_block] gives the coefficient for W[k*n+i, c]

    # Term 0: M1 = (A11+A22)(B11+B22)
    # u: A11=1, A22=1 => u[[0,0]=1, [1,1]=1]
    # v: B11=1, B22=1 => v[[0,0]=1, [1,1]=1]
    # w: C11 += M1, C22 += M1 => w[[0,0]=1, [1,1]=1]

    # Term 1: M2 = (A21+A22)B11
    # u: A21=1, A22=1 => u[[1,0]=1, [1,1]=1]
    # v: B11=1 => v[[0,0]=1]
    # w: C11 -= M2 => w[[0,0]]=-1

    # Term 2: M3 = A11(B21-B11)
    # u: A11=1 => u[[0,0]=1]
    # v: B21=1, B11=-1 => v[[0,1]]=-1, [1,0]=1
    # w: C21 += M3 => w[[1,1]]=1 (since C21 corresponds to k_block=1, i_block=1... wait)
    # Actually C[i,j] = sum_k A[i,k]B[k,j]. For 2x2:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # So C[i_block, j_block] where i_block is row, j_block is col.
    # But our W is indexed by [k*n + i] which is [k_block, i_block] in the 2x2 block sense.
    # Wait, the output is C[i,j] = sum_k A[i,k] B[k,j].
    # In the tensor decomposition, W[k*n+i, r] is the coefficient for the (k, i) pair.
    # So for a 2x2 block, W[(k_block)*2 + i_block, c] is the coefficient.
    # And the output C[i,j] = sum_k W[(k)*2+i, c] * U[i*2+j, c] * V[j*2+k, c]
    # Hmm, actually the standard formulation is:
    # C[i,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ... no.
    # Let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... but k is summed over?
    # No, the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = delta[i,j,k] ... no.
    # For matrix multiplication: (AB)[i,j] = sum_k A[i,k] B[k,j]
    # The multiplication tensor has T[i,j,k] = 1 if we're computing (AB)[i,j] and the k-th term is A[i,k]*B[k,j].
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that each (i,j,k) triple contributes).
    # Wait no. The tensor is defined by T[i,j,k] = A[i,k] * B[k,j] but that's not a tensor in i,j,k alone.
    # The multiplication tensor T satisfies (AB)[i,j] = sum_k A[i,k] B[k,j]
    # And the tensor decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (since the "input" is just the identity, and we're decomposing the structure).
    # Actually, I think the convention is that T[i,j,k] = 1 for the multiplication tensor, and the decomposition
    # U, V, W satisfies sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # No wait, that can't be right either because then it's just a rank-1 decomposition of the all-ones tensor.
    # Let me re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
    # I think the tensor is: T[i, j, k] = 1 (the structure of matrix multiplication), and we need
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to give T[i,j,k] = 1.
    # Hmm, but that would mean we're decomposing the all-ones tensor, which has rank n*m*p in the trivial way.
    # 
    # Actually, I think I'm overcomplicating this. The schoolbook decomposition in the seed program does:
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # And the "tensor value" T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # = 1 (only the term where r corresponds to (i,j,k) contributes).
    # So yes, T[i,j,k] = 1 for all i,j,k. We're decomposing the all-ones tensor of size (n*m)*(m*p)*(p*n).
    # Wait, that doesn't make sense for matrix multiplication.
    #
    # Let me reconsider. In tensor decomposition for matrix multiplication, the tensor is:
    # M[i, j, k] = delta_{i,k'} * delta_{j',k} ... no.
    # The multiplication tensor M has M[i, j, k] = 1, and the decomposition gives:
    # (AB)[i,j] = sum_k A[i,k] B[k,j] = sum_r (sum_k A[i,k] V[j*p+k, r]) * (sum_k W[k*n+i, r] * ...)
    # I think the correct interpretation is:
    # T[i, j, k] = 1, and we need rank-r decomposition such that T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # But that's just decomposing the all-ones tensor, which has tensor rank n*m*p in general.
    #
    # I think I need to just trust the seed program and the observed improvement from Strassen.
    # The observed score went from 1.0 to 1.306 with rank 49 vs 64, so 64/49 ≈ 1.306.
    # So the score is seed_rank / current_rank = 64 / rank.
    #
    # Let me just implement Strassen correctly. The key insight is that Strassen decomposes the 2x2 multiplication
    # into 7 rank-1 terms, and we can Kronecker this for 4x4.

    # Strassen 2x2 decomposition:
    # 7 terms, each with (U_block, V_block, W_block) where each block is 2x2
    strassen_2x2 = [
        # (U_block, V_block, W_block)
        # M1 = (A11+A22)(B11+B22)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # M2 = (A21+A22)B11
        ([[0,0],[1,1]], [[1,0],[0,0]], [[-1,0],[0,0]]),
        # M3 = A11(B21-B11)
        ([[1,0],[0,0]], [[0,-1],[1,0]], [[0,0],[0,1]]),
        # M4 = A22(B22-B11)
        ([[0,0],[0,1]], [[0,-1],[0,1]], [[0,0],[0,1]]),  # Wait, C11 -= M2, C21 += M4? Let me check.
        # C11 = M1 - M2 + M4 - M6
        # C12 = M5 + M6
        # C21 = M3 + M4
        # C22 = M1 - M3 - M5 + M7
        # So W for M4: C11 += M4 => W[0,0] += 1? No, C11 is [i=0,j=0], and W[k*n+i, r] so for C11, we need k such that...
        # Actually W[k*n+i, r] contributes to C[i, j] via summing over k and j... I'm confusing myself.
        # Let me just use the known Strassen W matrix.
    ]

    # I'll use a different approach: directly construct the rank-49 decomposition using the
    # known Strassen 7-term formulas and Kronecker products.

    # For 2x2 Strassen, the 7 multiplications are:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)(B11)
    # P3 = (A11)(B21-B11)
    # P4 = (A22)(B22-B11)
    # P5 = (A11+A12)(B22)
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 - P2 + P4 - P6
    # C12 = P5 + P6
    # C21 = P3 + P4
    # C22 = P1 - P3 - P5 + P7

    # Each P corresponds to a rank-1 term in the 2x2 tensor.
    # For P1: A_factor = [[1,0],[0,1]], B_factor = [[1,0],[0,1]], C_contribution = [[1,0],[0,1]]
    # For P2: A_factor = [[0,0],[1,1]], B_factor = [[1,0],[0,0]], C_contribution = [[-1,0],[0,0]]
    # etc.

    # For 4x4, we use Kronecker product of the 2x2 Strassen decomposition with itself.
    # The 4x4 matrix is viewed as a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    # Using Strassen on the 2x2 block level:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B10-B00)
    # P4 = (A11)(B11-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 - P2 + P4 - P6
    # C01 = P5 + P6
    # C10 = P3 + P4
    # C11 = P1 - P3 - P5 + P7

    # Each Pj is a 2x2 matrix product of 2x2 matrices, which itself requires 7 multiplications (Strassen) or 4 (schoolbook).
    # To get rank 49, we use Strassen at both levels: 7 * 7 = 49.

    # Let's define the 7 Strassen terms for 2x2 as (A_coeff, B_coeff, C_coeff) where each is a 2x2 matrix of coefficients.
    # Then for 4x4, each term (j, k) where j is the outer Strassen term and k is the inner Strassen term,
    # the U factor is kron(A_coeff_outer[j], A_coeff_inner[k]), etc.

    S2 = [
        # (A_coeff, B_coeff, C_coeff)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),   # P1
        ([[0,0],[1,1]], [[1,0],[0,0]], [[-1,0],[0,0]]),   # P2
        ([[1,0],[0,0]], [[0,-1],[1,0]], [[0,0],[0,1]]),   # P3
        ([[0,0],[0,1]], [[0,-1],[0,1]], [[1,0],[0,1]]),   # P4: C11 += P4, C21 += P4? 
        # Wait: C11 = P1 - P2 + P4 - P6, so C11 gets +P4. C21 = P3 + P4, so C21 gets +P4.
        # C_coeff for P4 should have [0,0]=1 (for C11) and [1,0]=1 (for C21). But C_coeff[i,j] means C[i,j] gets this coefficient.
        # C_coeff[0,0] = 1 (C11), C_coeff[1,0] = 1 (C21). So C_coeff = [[1,0],[1,0]].
        ([[0,0],[0,1]], [[0,-1],[0,1]], [[1,0],[1,0]]),   # P4 corrected
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,1],[0,0]]),   # P5: C12 = P5 + P6, so C12 gets P5. C_coeff[0,1]=1.
        ([[0,0],[1,-1]], [[1,1],[0,0]], [[-1,1],[0,0]]), # P6: C11 = P1 - P2 + P4 - P6, C12 = P5 + P6. C_coeff[0,0]=-1, C_coeff[0,1]=1.
        ([[0,1],[0,-1]], [[1,0],[1,1]], [[0,0],[0,1]]),   # P7: C11 = P1 - P3 - P5 + P7? No, C22 = P1 - P3 - P5 + P7. C_coeff[1,1]=1.
    ]

    # Let me verify the Strassen formulas more carefully.
    # Standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)(B11)
    # P3 = (A11)(B21 - B11)
    # P4 = (A22)(B22 - B11)
    # P5 = (A11 + A12)(B22)
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # C11 = P1 - P2 + P4 - P6
    # C12 = P5 + P6
    # C21 = P3 + P4
    # C22 = P1 - P3 - P5 + P7

    # A_coeff for each P (2x2 matrix where A_coeff[i,j] is the coefficient of A[i,j]):
    # P1: A11+A22 => [[1,0],[0,1]]
    # P2: A21+A22 => [[0,0],[1,1]]
    # P3: A11 => [[1,0],[0,0]]
    # P4: A22 => [[0,0],[0,1]]
    # P5: A11+A12 => [[1,1],[0,0]]
    # P6: A21-A11 => [[-1,0],[1,0]]
    # P7: A12-A22 => [[0,1],[0,-1]]

    # B_coeff:
    # P1: B11+B22 => [[1,0],[0,1]]
    # P2: B11 => [[1,0],[0,0]]
    # P3: B21-B11 => [[-1,0],[1,0]]
    # P4: B22-B11 => [[-1,0],[0,1]]
    # P5: B22 => [[0,0],[0,1]]
    # P6: B11+B12 => [[1,1],[0,0]]
    # P7: B21+B22 => [[0,0],[1,1]]

    # C_coeff (contribution to C[i,j]):
    # P1: C11+=P1, C22+=P1 => [[1,0],[0,1]]
    # P2: C11-=P2 => [[-1,0],[0,0]]
    # P3: C21+=P3 => [[0,0],[1,0]]
    # P4: C11+=P4, C21+=P4 => [[1,0],[1,0]]
    # P5: C12+=P5 => [[0,1],[0,0]]
    # P6: C11-=P6, C12+=P6 => [[-1,1],[0,0]]
    # P7: C22+=P7 => [[0,0],[0,1]]

    S2_correct = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[-1,0],[0,0]]),
        ([[1,0],[0,0]], [[-1,0],[1,0]], [[0,0],[1,0]]),
        ([[0,0],[0,1]], [[-1,0],[0,1]], [[1,0],[1,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,1],[0,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[-1,1],[0,0]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[0,0],[0,1]]),
    ]

    # Now for 4x4, we Kronecker product each 2x2 Strassen term with itself.
    # The 4x4 matrix A is partitioned into 2x2 blocks: A00, A01, A10, A11 (each 2x2).
    # In the full 4x4 matrix, A[i,j] where i = 2*i_block + i_inner, j = 2*j_block + j_inner.
    # So U[i*m+j, r] where i = 2*i_b + i_i, j = 2*j_b + j_i.
    # U = kron(A_coeff_outer, A_coeff_inner) where A_coeff_outer is 2x2 (block level) and A_coeff_inner is 2x2 (inner level).
    # kron(A_outer, A_inner)[i, j] where i = 2*i_b + i_i, j = 2*j_b + j_j = A_outer[i_b, j_b] * A_inner[i_i, j_j].

    # For the 49 rank-1 terms, term index r = 7*outer_idx + inner_idx.
    for outer_idx in range(7):
        A_outer, B_outer, C_outer = S2_correct[outer_idx]
        for inner_idx in range(7):
            A_inner, B_inner, C_inner = S2_correct[inner_idx]
            r = 7 * outer_idx + inner_idx

            # U: 4x4 matrix, U[i*4+j, r] = A_outer[i_b, j_b] * A_inner[i_i, j_j]
            # where i = 2*i_b + i_i, j = 2*j_b + j_j
            for i_b in range(2):
                for i_i in range(2):
                    i = 2 * i_b + i_i
                    for j_b in range(2):
                        for j_j in range(2):
                            j = 2 * j_b + j_j
                            U[i * m + j][r] = A_outer[i_b][j_b] * A_inner[i_i][j_j]

            # V: 4x4 matrix, V[j*4+k, r] = B_outer[j_b, k_b] * B_inner[j_i, k_i]
            # where j = 2*j_b + j_i, k = 2*k_b + k_i
            for j_b in range(2):
                for j_i in range(2):
                    j = 2 * j_b + j_i
                    for k_b in range(2):
                        for k_i in range(2):
                            k = 2 * k_b + k_i
                            V[j * p + k][r] = B_outer[j_b][k_b] * B_inner[j_i][k_i]

            # W: 4x4 matrix, W[k*4+i, r] = C_outer[k_b, i_b] * C_inner[k_i, i_i]
            # where k = 2*k_b + k_i, i = 2*i_b + i_i
            for k_b in range(2):
                for k_i in range(2):
                    k = 2 * k_b + k_i
                    for i_b in range(2):
                        for i_i in range(2):
                            i = 2 * i_b + i_i
                            W[k * n + i][r] = C_outer[k_b][i_b] * C_inner[k_i][i_i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
