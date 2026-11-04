"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _laderman_4x4():
    """
    Return the Laderman rank-19 bilinear forms for 4x4 matrix multiplication.
    Each form is (u_vec, v_vec, w_vec) where u, v are length-16 (row-major 4x4),
    w is length-16 (row-major 4x4) representing the contribution to C = A*B.
    C[i,j] = sum_c u[c,i] * v[c,j] * w[c,i,j] ... actually we use the standard
    tensor decomposition: C_{ij} = sum_c U_{ic} * V_{jc} * W_{ijc}
    where U, V, W are the factor matrices.
    """
    # Laderman's 19 bilinear forms for 4x4x4
    # Using the well-known decomposition.
    # The 19 terms are listed as (u_indices, v_indices, w_indices)
    # where each index is a (i,j) pair in 0..15 for 4x4 matrices.
    
    # Let's define the 19 bilinear forms explicitly.
    # Each form contributes u * v * w to specific (i,j) positions.
    
    # I'll use the standard Laderman decomposition.
    # The forms are:
    forms = []
    
    # Form 1: (A00+A11)(B00+B11) - A00*B11 - A11*B00
    # Actually, let me use the explicit 19-term Laderman decomposition.
    
    # The Laderman decomposition uses these 19 bilinear forms:
    # I'll represent each as (u, v, w) where u, v are vectors in R^16
    # and w is a matrix in R^{4x4} (flattened to 16).
    
    # Using the known Laderman construction:
    # Let's index A and B as 4x4 matrices with entries A[i][j], B[j][k].
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # The 19 bilinear forms:
    # I'll use the formulation from Laderman (1976).
    
    # Let me use a cleaner representation. Each bilinear form is:
    # u * v where u is a linear form in A entries, v is a linear form in B entries,
    # and the product contributes to specific C[i][k] entries.
    
    # Actually, for the tensor decomposition, we need:
    # C[i,k] = sum_c U[i,c] * V[k,c] * ... no wait.
    
    # The standard form is: C = A*B, and we write
    # C[i,k] = sum_c M_c(A) * N_c(B) * P_{i,k,c}
    # where M_c, N_c are linear forms and P_{i,k,c} are constants.
    
    # But the task asks for U, V, W such that:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Wait, let me re-read the task. The task says:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Hmm, that doesn't look right either. Let me think again.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 if we're computing C[i,k].
    # Actually, the multiplication tensor has entries T[i,j,k] = 1 for all valid i,j,k
    # in the sense that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    
    # A rank-r decomposition writes T as sum of r rank-1 tensors:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (or similar)
    
    # The task specifies:
    # U has shape (n*m, rank) -> U[i*m+j, c]
    # V has shape (m*p, rank) -> V[j*p+k, c]
    # W has shape (p*n, rank) -> W[k*n+i, c]
    
    # And the decomposition should satisfy:
    # For the multiplication tensor: sum_j A[i,j]*B[j,k] = C[i,k]
    # This is achieved by: C[i,k] = sum_c (sum over appropriate indices)
    
    # Actually, I think the convention is:
    # The bilinear form for computing C is:
    # C[i,k] = sum_c U[i*m + j_i, c] * V[j_v * p + k_v, c] * W[k_w * n + i_w, c]
    # where the sums are over c, and the indices are structured such that
    # each term contributes to exactly one (i,k) pair.
    
    # Let me look at the seed program more carefully:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This means for each (i,j,k), we have a rank-1 term where:
    # U picks out A[i,j], V picks out B[j,k], and W contributes to C[k*n+i]
    # But W[k*n+i] is a single entry, so this means each rank-1 term
    # contributes A[i,j]*B[j,k] to position (k*n+i) in the output.
    
    # But the output C is n x p, so it has n*p entries. The index k*n+i
    # ranges over k in [0,p) and i in [0,n), so k*n+i is in [0, p*n).
    # This matches W having shape (p*n, rank).
    
    # So the convention is: C is stored as a vector of length p*n with
    # C_flat[k*n+i] = C[i,k] (column-major in some sense, or rather
    # the output index is k*n+i where k is the column and i is the row).
    
    # Wait, the task says "The output factor follows the official trace convention W[k*n+i,r]".
    # So W[k*n+i, r] corresponds to the (i,k) entry of C (row i, column k).
    
    # OK so for the 4x4x4 case (n=4, m=4, p=4):
    # U has shape (16, rank), V has shape (16, rank), W has shape (16, rank).
    # U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] summed over c gives C[i,k].
    
    # For Laderman, we need 19 such terms.
    
    # Let me use the explicit Laderman decomposition. I'll look up the exact forms.
    
    # The Laderman decomposition for 4x4:
    # There are 19 bilinear forms. Let me denote them as (u, v, w) where
    # u is a vector in R^{16} (coefficients for A entries),
    # v is a vector in R^{16} (coefficients for B entries),
    # w is a matrix in R^{4x4} (coefficients for C entries, flattened as w[k*4+i]).
    
    # The condition is: for all i,k: C[i,k] = sum_c u_c[A] * v_c[B] * w_c[i,k]
    # where u_c[A] = sum_{i',j'} u_c[i'*4+j'] * A[i',j']
    # and similarly for v.
    
    # Actually, I realize I should just directly construct U, V, W.
    
    # Let me use the known Laderman 19-term decomposition.
    # I'll define the 19 terms explicitly.
    
    # From Laderman (1976), the 19 bilinear forms for 4x4 matrix multiplication:
    
    # I'll use a formulation where each term c has:
    # - a set of A-entries with coefficients (defining U[:,c])
    # - a set of B-entries with coefficients (defining V[:,c])
    # - a set of C-entries with coefficients (defining W[:,c])
    
    # The 19 terms (using 0-indexed (i,j) for 4x4 matrices):
    
    laderman_forms = [
        # Each form: (u_dict, v_dict, w_dict)
        # u_dict: {(i,j): coeff} for A
        # v_dict: {(j,k): coeff} for B
        # w_dict: {(i,k): coeff} for C (stored as k*4+i)
        
        # Term 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}),
        # Term 2
        ({(0,1):1, (1,0):1}, {(0,1):1, (1,0):1}, {(0,1):1, (1,0):1}),
        # Term 3
        ({(0,0):1, (1,0):1}, {(0,0):1, (0,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # Term 4
        ({(0,1):1, (1,1):1}, {(1,0):1, (1,1):1}, {(0,1):1, (0,0):1, (1,1):1, (1,0):1}),
        # Term 5
        ({(2,2):1, (3,3):1}, {(2,2):1, (3,3):1}, {(2,2):1, (3,3):1, (2,3):-1, (3,2):-1}),
        # Term 6
        ({(2,3):1, (3,2):1}, {(2,3):1, (3,2):1}, {(2,3):1, (3,2):1}),
        # Term 7
        ({(2,2):1, (3,2):1}, {(2,2):1, (2,3):1}, {(2,2):1, (2,3):1, (3,2):1, (3,3):1}),
        # Term 8
        ({(2,3):1, (3,3):1}, {(3,2):1, (3,3):1}, {(2,3):1, (2,2):1, (3,3):1, (3,2):1}),
        # Term 9
        ({(0,2):1, (1,3):1}, {(2,0):1, (3,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}),
        # Term 10
        ({(0,3):1, (1,2):1}, {(3,0):1, (2,1):1}, {(0,1):1, (1,0):1}),
        # Term 11
        ({(0,2):1, (1,2):1}, {(2,0):1, (2,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # Term 12
        ({(0,3):1, (1,3):1}, {(3,0):1, (3,1):1}, {(0,1):1, (0,0):1, (1,1):1, (1,0):1}),
        # Term 13
        ({(2,0):1, (3,1):1}, {(0,2):1, (1,3):1}, {(2,2):1, (3,3):1, (2,3):-1, (3,2):-1}),
        # Term 14
        ({(2,1):1, (3,0):1}, {(1,2):1, (0,3):1}, {(2,3):1, (3,2):1}),
        # Term 15
        ({(2,0):1, (3,0):1}, {(0,2):1, (0,3):1}, {(2,2):1, (2,3):1, (3,2):1, (3,3):1}),
        # Term 16
        ({(2,1):1, (3,1):1}, {(1,2):1, (1,3):1}, {(2,3):1, (2,2):1, (3,3):1, (3,2):1}),
        # Term 17
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1, (0,2):-1, (2,0):-1}),
        # Term 18
        ({(0,2):1, (2,0):1}, {(0,2):1, (2,0):1}, {(0,2):1, (2,0):1}),
        # Term 19
        ({(0,0):1, (0,2):1}, {(0,0):1, (0,2):1}, {(0,0):1, (0,2):1, (2,0):1, (2,2):1}),
    ]
    
    # Wait, I'm not confident these are correct. Let me verify by checking
    # if the decomposition actually works. But first, let me reconsider.
    
    # Actually, I think I should just use the Kronecker product approach with
    # two levels of Strassen, which gives rank 49 for 4x4x4, and then extend
    # to 4x4x8 by applying to two 4x4 blocks, giving rank 98.
    
    # The knowledge says: "Kronecker product of two Strassen levels gives rank-49
    # for 4x4x4 instead of the 8-block rank-56, reducing total rank from 112 to 98
    # for 4x4x8 Measured: 1 -> 1.306."
    
    # So let me implement the Kronecker product of two Strassen levels.
    
    return None  # Placeholder, I'll implement Strassen instead

def _strassen_2x2():
    """
    Return the Strassen rank-7 bilinear forms for 2x2 matrix multiplication.
    Each form is (u, v, w) where u, v are length-4 vectors (2x2 flattened),
    and w is a length-4 vector (2x2 flattened).
    """
    # Strassen's 7 bilinear forms for 2x2:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each bilinear form M_c contributes:
    # M_c = u_c * v_c where u_c is a linear form in A, v_c in B.
    # And M_c contributes to C entries with coefficients from w_c.
    
    # In our convention: C_flat[k*2+i] = C[i,k] for 2x2.
    # So C00 -> index 0*2+0 = 0, C01 -> index 1*2+0 = 2,
    # C10 -> index 0*2+1 = 1, C11 -> index 1*2+1 = 3.
    
    # Wait, for 2x2: n=2, p=2. W[k*n+i] = W[k*2+i].
    # C[i,k] is stored at W[k*2+i].
    # C00: i=0,k=0 -> W[0]
    # C01: i=0,k=1 -> W[2]
    # C10: i=1,k=0 -> W[1]
    # C11: i=1,k=1 -> W[3]
    
    # A[i,j] stored at U[i*2+j]:
    # A00: U[0], A01: U[1], A10: U[2], A11: U[3]
    
    # B[j,k] stored at V[j*2+k]:
    # B00: V[0], B01: V[1], B10: V[2], B11: V[3]
    
    forms = []
    
    # M1 = (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1]  # A00 + A11
    v1 = [1, 0, 0, 1]  # B00 + B11
    w1 = [1, 0, 0, 1]  # C00 + C11
    forms.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00
    u2 = [0, 0, 1, 1]  # A10 + A11
    v2 = [1, 0, 0, 0]  # B00
    w2 = [0, 1, 0, 0]  # C10
    forms.append((u2, v2, w2))
    
    # M3 = A00(B10-B11)
    u3 = [1, 0, 0, 0]  # A00
    v3 = [0, 0, 1, -1]  # B10 - B11
    w3 = [0, 0, 1, 0]  # C01
    forms.append((u3, v3, w3))
    
    # M4 = A11(B01-B00)
    u4 = [0, 0, 0, 1]  # A11
    v4 = [-1, 1, 0, 0]  # B01 - B00
    w4 = [1, 0, 0, 0]  # C00
    forms.append((u4, v4, w4))
    
    # M5 = (A00+A01)B11
    u5 = [1, 1, 0, 0]  # A00 + A01
    v5 = [0, 0, 0, 1]  # B11
    w5 = [0, 0, 0, 1]  # C11 ... wait
    # C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    # So M5 contributes -1 to C00 and +1 to... let me recheck.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M5 contributes: -1 to C00, +1 to C01
    w5 = [-1, 0, 1, 0]  # -C00 + C01
    forms.append((u5, v5, w5))
    
    # M6 = (A10-A00)(B00+B01)
    u6 = [-1, 0, 1, 0]  # A10 - A00
    v6 = [1, 1, 0, 0]  # B00 + B01
    w6 = [0, 0, 0, 1]  # C11
    forms.append((u6, v6, w6))
    
    # M7 = (A01-A11)(B10+B11)
    u7 = [0, 1, 0, -1]  # A01 - A11
    v7 = [0, 0, 1, 1]  # B10 + B11
    w7 = [1, 0, 0, 0]  # C00
    forms.append((u7, v7, w7))
    
    return forms

def _kron_strassen_4x4():
    """
    Build the Kronecker product of two Strassen levels for 4x4x4.
    This gives rank 7*7 = 49.
    """
    s2 = _strassen_2x2()
    
    # For 4x4, we view A as a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # Similarly for B and C.
    
    # The Kronecker product approach:
    # We first apply Strassen at the block level (2x2 blocks of 2x2 matrices),
    # then apply Strassen within each block multiplication.
    
    # Actually, the Kronecker product of two Strassen decompositions works as follows:
    # If we have a rank-r1 decomposition for n1 x m1 x p1 and a rank-r2 decomposition
    # for n2 x m2 x p2, then the Kronecker product gives a rank-r1*r2 decomposition
    # for (n1*n2) x (m1*m2) x (p1*p2).
    
    # For 4x4x4 = (2*2) x (2*2) x (2*2), we use two levels of 2x2x2 Strassen.
    
    # The Kronecker product of two rank-7 decompositions gives rank 49.
    
    # Let's construct it.
    # For the first level (block level), we have:
    # C_block[i,k] = sum_c1 U1[i*2+j, c1] * V1[j*2+k, c1] * W1[k*2+i, c1]
    # where i,j,k are in {0,1} and the blocks are 2x2.
    
    # For the second level (within blocks), we have:
    # C_within[i',k'] = sum_c2 U2[i'*2+j', c2] * V2[j'*2+k', c2] * W2[k'*2+i', c2]
    # where i',j',k' are in {0,1}.
    
    # The combined decomposition has rank 7*7 = 49, with terms indexed by (c1, c2).
    
    # The full index mapping:
    # A is 4x4, indexed as A[I,J] where I = i*2+i', J = j*2+j'
    # B is 4x4, indexed as B[J,K] where J = j*2+j', K = k*2+k'
    # C is 4x4, indexed as C[I,K] where I = i*2+i', K = k*2+k'
    
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for c1 in range(7):
        u1, v1, w1 = s2[c1]
        for c2 in range(7):
            u2, v2, w2 = s2[c2]
            c = c1 * 7 + c2
            
            # For each (i,j,k) in {0,1}^3:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # The block-level contribution is U1[i*2+j, c1] * V1[j*2+k, c1] * W1[k*2+i, c1]
                        # The within-block contribution is U2[i'*2+j', c2] * V2[j'*2+k', c2] * W2[k'*2+i', c2]
                        
                        # Combined: for each (i', j', k') in {0,1}^3:
                        for i_p in range(2):
                            for j_p in range(2):
                                for k_p in range(2):
                                    # Full indices:
                                    I = i * 2 + i_p
                                    J = j * 2 + j_p
                                    K = k * 2 + k_p
                                    
                                    # U index: I*4 + J = (i*2+i_p)*4 + (j*2+j_p)
                                    u_idx = (i * 2 + i_p) * 4 + (j * 2 + j_p)
                                    # V index: J*4 + K = (j*2+j_p)*4 + (k*2+k_p)
                                    v_idx = (j * 2 + j_p) * 4 + (k * 2 + k_p)
                                    # W index: K*4 + I = (k*2+k_p)*4 + (i*2+i_p)
                                    w_idx = (k * 2 + k_p) * 4 + (i * 2 + i_p)
                                    
                                    # The coefficient is:
                                    # U1[i*2+j, c1] * V1[j*2+k, c1] * W1[k*2+i, c1] *
                                    # U2[i_p*2+j_p, c2] * V2[j_p*2+k_p, c2] * W2[k_p*2+i_p, c2]
                                    
                                    coeff = (u1[i*2+j] * v1[j*2+k] * w1[k*2+i] *
                                             u2[i_p*2+j_p] * v2[j_p*2+k_p] * w2[k_p*2+i_p])
                                    
                                    if coeff != 0:
                                        U[u_idx][c] += coeff
                                        V[v_idx][c] += coeff
                                        W[w_idx][c] += coeff
    
    return U, V, W

def construct():
    """
    Construct a rank-98 decomposition for 4x4x8 matrix multiplication
    using the Kronecker product of two Strassen levels applied to
    two 4x4 column blocks of the B matrix.
    """
    n, m, p = 4, 4, 8
    
    # Get the 4x4x4 Kronecker-Strassen decomposition (rank 49)
    U44, V44, W44 = _kron_strassen_4x4()
    
    # For 4x4x8, we split B into two 4x4 column blocks: B = [B1 | B2]
    # where B1 is columns 0-3 and B2 is columns 4-7.
    # C = A * B = [A*B1 | A*B2]
    
    # We apply the 4x4x4 decomposition independently to each block.
    # Total rank = 49 + 49 = 98.
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]  # (16, 98)
    V = [[0]*rank for _ in range(m*p)]  # (32, 98)
    W = [[0]*rank for _ in range(p*n)]  # (32, 98)
    
    # Block 1: B columns 0-3
    for c in range(49):
        for i in range(n*m):
            U[i][c] = U44[i][c]
        for j in range(m):
            for k in range(4):  # k = 0..3
                V[j*p + k][c] = V44[j*4 + k][c]
        for k in range(4):  # k = 0..3
            for i in range(n):
                W[k*n + i][c] = W44[k*4 + i][c]
    
    # Block 2: B columns 4-7
    for c in range(49):
        c2 = c + 49
        for i in range(n*m):
            U[i][c2] = U44[i][c]
        for j in range(m):
            for k in range(4, 8):  # k = 4..7
                V[j*p + k][c2] = V44[j*4 + (k-4)][c]
        for k in range(4, 8):  # k = 4..7
            for i in range(n):
                W[k*n + i][c2] = W44[(k-4)*4 + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
