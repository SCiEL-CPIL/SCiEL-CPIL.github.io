"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Use the rank-23 decomposition of the 3x3x3 matrix multiplication tensor
    # (Bayer, Strassen, and others; best known upper bound).
    # 6x6 = 2x2 blocks of 3x3, so 8 block multiplications, each of rank 23.
    # Total rank = 8 * 23 = 184.
    #
    # We use the explicit rank-23 decomposition from the literature.
    # The coefficients are integers.

    # The 23 rank-1 tensors for 3x3 matrix multiplication.
    # Each is (U_3x3, V_3x3, W_3x3) where C = sum_k (U_k * A) * (V_k * B) is not quite right.
    # Actually: C[i,j] = sum_k U_k[i,j] * (sum_{a,b} U_k[i,a] * A[a,b]) * (sum_{a,b} V_k[j,b] * B[a,b])
    # No, that's not the right form either.
    #
    # The tensor decomposition: T_{i,j,k} = sum_r U_r[i,k] * V_r[j,k] * W_r[i,j]
    # where T is the matrix multiplication tensor: T_{i,j,k} = 1 if j=k (i.e., C[i,j] = sum_k A[i,k]*B[k,j])
    # Wait, the standard form: (AB)_{i,j} = sum_k A_{i,k} B_{k,j}
    # So the tensor T has indices (i, k, j) and T_{i,k,j} = delta_{k, k'}... 
    # Actually T_{i,k,j} = 1 if we're computing C_{i,j} = sum_k A_{i,k} B_{k,j}
    # The tensor is T_{i,k,j} with T_{i,k,j} = 1 for all valid i,k,j? No.
    # T_{i,k,j} = A_{i,k} * B_{k,j} contributes to C_{i,j}.
    # So the multiplication tensor has T_{i,k,j} = 1 (it's the identity in the "contraction" sense).
    #
    # Rank-1 decomposition: T_{i,k,j} = sum_r u_r[i] * v_r[k] * w_r[j] ... no.
    #
    # Let me use the standard formulation. The matrix multiplication tensor m_{n} has
    # entries (m_n)_{i,j,k} = 1 if i=j=k... no.
    #
    # (AB)_{i,j} = sum_k A_{i,k} B_{k,j}
    # The tensor T_{i,k,j} = A_{i,k} B_{k,j} is a rank-1 tensor in the Kronecker product sense.
    # The multiplication tensor is sum_{i,k,j} e_i ⊗ e_k ⊗ e_j (with appropriate indices).
    #
    # For the rank decomposition: we need U, V, W such that:
    # T_{i,k,j} = sum_r U_r[i,k] * V_r[k,j] * W_r[i,j]  ... this doesn't work dimensionally.
    #
    # Let me use the interface directly. The interface says:
    # U: shape (n*m, rank) = (36, rank)
    # V: shape (m*p, rank) = (36, rank)  
    # W: shape (p*n, rank) = (36, rank)
    # with W[k*n+i, r] convention.
    #
    # The decomposition: for each r, U[:,r], V[:,r], W[:,r] define a rank-1 term.
    # U is indexed by (i,k) pairs: row = i*m + k (for A_{i,k})
    # V is indexed by (k,j) pairs: row = k*p + j (for B_{k,j})
    # W is indexed by (k,i) pairs with W[k*n+i, r] (for C_{i,j})
    #
    # The rank-1 term contributes: C_{i,j} += U[i*m+k, r] * V[k*p+j, r] * W[k*n+i, r] ... 
    # No, that's not right either. The standard tensor decomposition of matrix mult:
    #
    # C_{i,j} = sum_k A_{i,k} B_{k,j}
    # = sum_r [sum_k U_r[i,k] A_{i,k}] * [sum_k V_r[k,j] B_{k,j}] ... no.
    #
    # Actually the correct form for the bilinear decomposition:
    # C_{i,j} = sum_r W_r[i,j] * (sum_{a,b} U_r[i,a] * A[a,b]) * (sum_{a,b} V_r[j,b] * B[a,b])
    # No, that would be trilinear, not bilinear.
    #
    # The bilinear decomposition (which is what tensor rank means for matrix multiplication):
    # C_{i,j} = sum_r W_r[i,j] * M_r(A) * N_r(B)
    # where M_r(A) = sum_{a,b} U_r[a,b] * A[a,b] and N_r(B) = sum_{a,b} V_r[a,b] * B[a,b]
    #
    # So: U has shape (n*m, rank) where U[a*m+b, r] is the coefficient for A[a,b] in M_r
    # V has shape (m*p, rank) where V[a*p+b, r] is the coefficient for B[a,b] in N_r
    # W has shape (p*n, rank) where W[k*n+i, r] is the coefficient for C[i,j] (with k as row of C? no...)
    #
    # Wait, the interface says W[k*n+i, r] with shape (p*n, rank).
    # If C is p x n... but C should be n x p. Let me re-read.
    # "W[k*n+i, r]" - so W is indexed by (k, i) where k ranges over p and i ranges over n.
    # This suggests C has dimensions... hmm.
    #
    # Actually for 6x6 * 6x6 = 6x6, n=m=p=6.
    # U: (n*m, rank) = (36, rank) - indexed by (i,k) for A[i,k]
    # V: (m*p, rank) = (36, rank) - indexed by (k,j) for B[k,j]
    # W: (p*n, rank) = (36, rank) - indexed by (k,i) with W[k*n+i, r]
    #
    # The reconstruction: C[i,j] = sum_r W[j*n+i, r] * (sum_{a,b} U[a*m+b, r]*A[a,b]) * (sum_{a,b} V[a*p+b, r]*B[a,b])
    #
    # Wait, that still doesn't look right. Let me think about it differently.
    #
    # The standard Strassen-like decomposition for 2x2:
    # M0 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M1 = (A10+A11)B00 -> contributes to C10, C11
    # etc.
    #
    # So the form is: C[i,j] = sum_r W_r[i,j] * (linear form in A) * (linear form in B)
    #
    # For the 3x3 rank-23 decomposition, I'll use the known coefficients.
    # Let me hardcode them.

    # Rank-23 decomposition of 3x3 matrix multiplication
    # From Bayer (1989) / Strassen.
    # U_r: 3x3 matrix, V_r: 3x3 matrix, W_r: 3x3 matrix
    # C[i,j] = sum_r W_r[i,j] * (U_r : A) * (V_r : B)
    # where (U_r : A) = sum_{a,b} U_r[a,b] * A[a,b]

    # The 23 terms (using 0-indexed):
    # I'll use the decomposition from the literature.
    
    # Actually, let me use a simpler approach: use the known rank-23 coefficients.
    # These are well-known. Let me define them.
    
    # Each term r has:
    # U_r as a 9-element vector (flattened 3x3)
    # V_r as a 9-element vector (flattened 3x3)
    # W_r as a 9-element vector (flattened 3x3, with W_r[k*3+i] for C[i,j]... 
    # wait, W index is k*n+i where n=3 for the block, so W_r[k*3+i])
    
    # The 23 rank-1 terms for 3x3 matrix multiplication:
    # I'll use the coefficients from the known decomposition.
    
    # Term 0: U = [[1,0,0],[0,0,0],[0,0,1]], V = [[1,0,0],[0,0,0],[0,0,1]], W = [[1,0,0],[0,0,0],[0,0,1]]
    # This gives: (A00+A22)(B00+B22) -> C00 += ..., C22 += ...
    # Hmm, that's not quite right for the standard form.
    
    # Let me just use the explicit 23-term decomposition.
    # I'll encode it as lists of (u_flat, v_flat, w_flat) where each is a list of 9 ints.
    
    # Using the decomposition where:
    # C[i,j] = sum_r W_r[i,j] * (sum_{a,b} U_r[a,b]*A[a,b]) * (sum_{a,b} V_r[a,b]*B[a,b])
    
    # The 23 terms (I'm using a known valid decomposition):
    terms_3x3 = []
    
    # I'll construct these from the known rank-23 algorithm.
    # The key insight: for 3x3, rank 23 is achievable.
    # Let me use the coefficients from the literature.
    
    # Actually, I realize I should just use the same Strassen-on-2x2-blocks approach but
    # verify it's correct, since I'm not 100% sure of the rank-23 coefficients from memory.
    # The current program uses rank 189 (27 * 7) which is valid.
    # Rank 184 (8 * 23) would be better but requires the 3x3 rank-23 coefficients.
    
    # Let me try a different approach: use 2x2 Strassen but optimize the block structure.
    # Actually, 6 = 2*3, so we can do 3x3 blocks of 2x2, giving 27*7 = 189.
    # Or we can do 2x2 blocks of 3x3, giving 8*23 = 184.
    
    # Since I'm not confident in the rank-23 coefficients from memory, let me stick with
    # the known-correct Strassen 2x2 approach but make the code more robust.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                if time.time() > deadline:
                    break
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
