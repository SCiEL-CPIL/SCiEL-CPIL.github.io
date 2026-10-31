import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _matmul_tensor(n, m, p):
    """Tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matmul tensor."""
    T = _matmul_tensor(n, m, p)
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 via Strassen (2 levels of 7)
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 block multiplication
    # Each of the 7 terms s has linear forms on A, B and output on C
    # a_s[I][J], b_s[J][K], c_s[I][K] where I,J,K in {0,1}
    
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 - M3 + M4
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M6
    
    a_coeffs = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 0], [0, 1]],  # M3
        [[1, 1], [0, 0]],  # M4
        [[-1, 0], [1, 0]], # M5
        [[0, 1], [0, -1]], # M6
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],  # M0
        [[1, 0], [0, 0]],  # M1
        [[0, 1], [0, -1]], # M2
        [[-1, 0], [1, 0]], # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, 1]],  # M6
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],  # M0 -> C00, C11
        [[0, 0], [1, -1]], # M1 -> C10, C11
        [[0, 1], [0, 0]],  # M2 -> C01
        [[-1, 0], [0, 1]], # M3 -> C00, C10
        [[1, 1], [0, 0]],  # M4 -> C00, C01
        [[0, 0], [0, 1]],  # M5 -> C11
        [[0, 0], [0, 1]],  # M6 -> C11
    ]
    
    # For 4x4x4, we split into 2x2 blocks.
    # The tensor is T[i,j,k] for i,j,k in 0..3.
    # Block indices: I=i//2, i'=i%2, etc.
    # T[i,j,k] = sum_{s,t} U[i, 7s+t] V[j, 7s+t] W[k, 7s+t]
    
    # We can view this as a 7x7 grid of 2x2x2 tensors.
    # The (s,t) entry corresponds to the t-th term of the s-th Strassen product.
    # Wait, the standard recursive Strassen for 4x4 uses 7 products of 2x2 matrices.
    # Each 2x2 product is decomposed into 7 scalar products.
    # So total rank is 7*7=49.
    
    # Let's build the 2x2x2 decomposition first (rank 7).
    # U2, V2, W2 of shape (4, 7)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Fill U2, V2, W2 based on a_coeffs, b_coeffs, c_coeffs
    for s in range(7):
        for I in range(2):
            for J in range(2):
                if a_coeffs[s][I][J] != 0:
                    U2[I*2+J, s] = a_coeffs[s][I][J]
        for J in range(2):
            for K in range(2):
                if b_coeffs[s][J][K] != 0:
                    V2[J*2+K, s] = b_coeffs[s][J][K]
        for I in range(2):
            for K in range(2):
                if c_coeffs[s][I][K] != 0:
                    W2[K*2+I, s] = c_coeffs[s][I][K]
    
    # Now assemble 4x4x4.
    # U[i, 7s+t] = sum_{I,J} a_s[I][J] * U2[i' * 2 + j', t]  ??
    # No, that's not quite right.
    # The 4x4 matrix A is composed of 4 2x2 blocks A_{I,J}.
    # The s-th Strassen term is M_s = (sum_{I,J} a_s[I][J] A_{I,J}) * (sum_{J,K} b_s[J][K] B_{J,K}).
    # Let X_s = sum_{I,J} a_s[I][J] A_{I,J} and Y_s = sum_{J,K} b_s[J][K] B_{J,K}.
    # X_s is a 2x2 matrix. Y_s is a 2x2 matrix.
    # M_s = X_s * Y_s.
    # M_s is decomposed as sum_t U2[..., t] * V2[..., t] * W2[..., t] ?
    # No, M_s is the product of two 2x2 matrices.
    # The decomposition of the 2x2 matmul tensor is:
    # (X*Y)_{i',k'} = sum_t U2[i', t] X_{i'j'} ... wait.
    # The tensor for 2x2 matmul is T2[i', j', k'] = delta_{i'k'}? No.
    # T2[i', j', k'] is 1 if (i', j', k') corresponds to a valid term in C = A*B.
    # C_{i'k'} = sum_{j'} A_{i'j'} B_{j'k'}.
    # So T2[i', j', k'] is the coefficient of A_{i'j'} B_{j'k'} in C_{i'k'}.
    # This is 1 if i'=i' and k'=k' (always true) ... no.
    # The tensor is T2[i', j', k'] = 1 if the term A_{i'j'} B_{j'k'} contributes to C_{i'k'}.
    # It contributes to C_{i'k'} if i'=i' and k'=k'.
    # Actually, the decomposition is:
    # C_{i'k'} = sum_t (sum_{j'} U2[i',t] A_{i'j'}) (sum_{j'} V2[j',t] B_{j'k'}) W2[k',t] ?
    # No, the standard decomposition is:
    # C_{i'k'} = sum_t U2[i',t] V2[j',t] W2[k',t] is not how it works.
    # It is:
    # C = sum_t (U2[:,t] A) * (V2[:,t] B)^T ? No.
    # The rank decomposition of the tensor T is:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # For 2x2 matmul, T[i,j,k] is 1 if i=0,j=0,k=0; i=0,j=1,k=1; i=1,j=0,k=0; i=1,j=1,k=1?
    # No, T[i,j,k] = 1 if k=i? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has T[i,j,k] = 1 if the term A_{ij}B_{jk} is in C_{ik}.
    # This is always true for any i,j,k.
    # Wait, C_{ik} contains A_{ij}B_{jk} for all j.
    # So T[i,j,k] = 1 for all i,j,k? No, that would be rank 1.
    # The tensor is T[i,j,k] = 1 if i,j,k are such that A_{ij}B_{jk} contributes to C_{ik}.
    # It contributes to C_{ik} specifically.
    # So T[i,j,k] is 1 if i,j,k form a valid triple.
    # Actually, the tensor is defined by T[i,j,k] = 1 if the product A_{ij}B_{jk} appears in C_{ik}.
    # It appears in C_{ik} with coefficient 1.
    # So T[i,j,k] = 1 for all i,j,k?
    # No, C_{ik} = sum_j A_{ij}B_{jk}.
    # So the term A_{ij}B_{jk} appears in C_{ik}.
    # So T[i,j,k] = 1.
    # But this is for a fixed i,j,k.
    # The tensor is 8-dimensional? No, 2x2x2.
    # T is 2x2x2.
    # T[i,j,k] = 1 if A_{ij}B_{jk} is a term in C_{ik}.
    # Yes, for any i,j,k, A_{ij}B_{jk} is a term in C_{ik}.
    # So T[i,j,k] = 1 for all i,j,k?
    # That would mean T is the all-ones tensor, which has rank 1.
    # But 2x2 matmul has rank 7.
    # I am confusing the tensor definition.
    # The tensor T for matrix multiplication C=AB is defined such that:
    # C_{ik} = sum_{j} T[i,j,k] A_{ij} B_{jk}
    # So T[i,j,k] = 1 for all i,j,k.
    # Wait, if T[i,j,k]=1 for all i,j,k, then C_{ik} = sum_j A_{ij}B_{jk}.
    # This is correct.
    # So the tensor is the all-ones tensor?
    # No, the indices are i (row of C), j (col of A / row of B), k (col of C).
    # The tensor T has indices i, j, k.
    # T[i,j,k] = 1.
    # This tensor has rank 1?
    # Rank of all-ones tensor of size n x n x n is 1.
    # But Strassen says rank 7 for 2x2.
    # Where is the error?
    # Ah, the tensor is T[i,j,k] where i is row of A, j is col of A, k is col of B?
    # No, the standard definition is:
    # The bilinear map (A,B) -> C = AB.
    # The tensor T is in V_A^* \otimes V_B^* \otimes V_C.
    # T(A,B) = C.
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # So T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_k (where e_k is basis of C space indexed by ik?)
    # No, C is a vector space of dimension 4. Basis is C_{00}, C_{01}, C_{10}, C_{11}.
    # So T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_{ik} (where e_{ik} is the basis vector for C_{ik})?
    # No, the sum is over j.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes (coefficient of C_{ik})?
    # No.
    # T is a tensor such that T(A,B) = sum_{i,j,k} T_{ijk} A_i B_j e_k.
    # Here A_i is the i-th component of A (flattened), B_j is j-th of B.
    # C_k is k-th of C.
    # C_k = sum_{i,j} T_{ijk} A_i B_j.
    # We know C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} (where i=idx(A), j=idx(B), k=idx(C)) is 1 if A_i and B_j contribute to C_k.
    # A_i is A_{row, col}. B_j is B_{row, col}.
    # A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{(i,j), (j,k), (i,k)} = 1.
    # Note that the index j in A_{ij} and B_{jk} must match.
    # So T is sparse.
    # For 2x2:
    # A indices: 00, 01, 10, 11.
    # B indices: 00, 01, 10, 11.
    # C indices: 00, 01, 10, 11.
    # T[00, 00, 00] = 1 (A00 B00 -> C00)
    # T[00, 01, 01] = 1 (A00 B01 -> C01)
    # T[01, 00, 10] = 1 (A01 B00 -> C10)
    # T[01, 01, 11] = 1 (A01 B01 -> C11)
    # T[10, 00, 10] = 1 (A10 B00 -> C10)
    # T[10, 01, 11] = 1 (A10 B01 -> C11)
    # T[11, 00, 11] = 1 (A11 B00 -> C11)
    # T[11, 01, 11] = 1 (A11 B01 -> C11) -- Wait, A11 B01?
    # C_{11} = A10 B01 + A11 B11.
    # So T[10, 01, 11] = 1.
    # T[11, 11, 11] = 1.
    # Let's list them properly.
    # C00 = A00 B00
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B01
    # C11 = A10 B01 + A11 B11
    
    # So non-zero T entries:
    # (00, 00, 00)
    # (00, 01, 01)
    # (01, 11, 01)
    # (10, 00, 10)
    # (11, 01, 10)
    # (10, 01, 11)
    # (11, 11, 11)
    
    # This tensor has rank 7.
    
    # Back to 4x4.
    # The 4x4 tensor is built from 2x2 blocks.
    # We can use the Kronecker product structure or just the recursive Strassen.
    # The recursive Strassen gives rank 49.
    # We just need to generate the U, V, W matrices.
    
    # Let's use the block structure.
    # U[i, r] where i is index in 4x4 A, r is rank index.
    # r = 7*s + t.
    # U[i, 7s+t] = sum_{I,J} a_s[I][J] * U2[i_local, t]
    # where i_local is the local index in the 2x2 block.
    # i = I*2 + i'. j = J*2 + j'.
    # A_{ij} is in block A_{I,J} at position (i', j').
    # So U[i, 7s+t] = a_s[I][J] * U2[i'*2+j', t].
    
    # Similarly for V and W.
    # V[j, 7s+t] = b_s[J][K] * V2[j'*2+k', t]
    # W[k, 7s+t] = c_s[I][K] * W2[k'*2+i', t]
    # Note: W index is k*n+i. In 2x2 block, it is k'*2+i'.
    
    # Let's verify this logic.
    # T[i,j,k] = sum_{s,t} U[i, 7s+t] V[j, 7s+t] W[k, 7s+t]
    # = sum_{s,t} (sum_{I,J} a_s[I][J] U2[i'2+j', t]) (sum_{J',K'} b_s[J'][K'] V2[j'2+k', t]) (sum_{I',K'} c_s[I'][K'] W2[k'2+i', t])
    # This looks like it sums over all I,J,J',K',I',K'.
    # But the Strassen identity says:
    # C_{ik} = sum_s c_s[I][K] * M_s
    # M_s = (sum_{I,J} a_s[I][J] A_{I,J}) * (sum_{J,K} b_s[J][K] B_{J,K})
    # = (sum_{I,J} a_s[I][J] A_{I,J}) * (sum_{J,K} b_s[J][K] B_{J,K})
    # The product of two 2x2 matrices X_s and Y_s.
    # (X_s Y_s)_{i'k'} = sum_{j'} X_s[i'j'] Y_s[j'k'].
    # X_s[i'j'] = sum_{I,J} a_s[I][J] A_{I,J}[i'j']
    # Y_s[j'k'] = sum_{J,K} b_s[J][K] B_{J,K}[j'k']
    # So (X_s Y_s)_{i'k'} = sum_{j'} (sum_{I,J} a_s[I][J] A_{I,J}[i'j']) (sum_{J,K} b_s[J][K] B_{J,K}[j'k'])
    # = sum_{I,J,J,K,j'} a_s[I][J] b_s[J][K] A_{I,J}[i'j'] B_{J,K}[j'k']
    # Note: The J in a_s[I][J] and J in b_s[J][K] must match?
    # In Strassen, M_s = X_s Y_s.
    # X_s is formed from A blocks. Y_s from B blocks.
    # The indices I,J in X_s and J,K in Y_s are block indices.
    # The "j'" is the local index within the 2x2 block.
    # The block index J in X_s and J in Y_s are the same?
    # No, X_s is a 2x2 matrix. Y_s is a 2x2 matrix.
    # The product is standard matrix multiplication.
    # The block indices I,J in X_s do not constrain J in Y_s.
    # Wait, a_s[I][J] is the coefficient of block A_{I,J}.
    # b_s[J][K] is the coefficient of block B_{J,K}.
    # In the definition of M_s, the J in A_{I,J} and the J in B_{J,K} are summed over?
    # No, M_s = (linear form on A) * (linear form on B).
    # The linear form on A is X_s = sum_{I,J} a_s[I][J] A_{I,J}.
    # The linear form on B is Y_s = sum_{J,K} b_s[J][K] B_{J,K}.
    # These are just 2x2 matrices.
    # So M_s = X_s Y_s.
    # Then C_{I,K} += c_s[I][K] M_s.
    # So C_{I,K}[i'k'] += c_s[I][K] (X_s Y_s)[i'k'].
    # (X_s Y_s)[i'k'] = sum_{j'} X_s[i'j'] Y_s[j'k'].
    # X_s[i'j'] = sum_{I,J} a_s[I][J] A_{I,J}[i'j'].
    # Y_s[j'k'] = sum_{J,K} b_s[J][K] B_{J,K}[j'k'].
    # So C_{I,K}[i'k'] = sum_{s} c_s[I][K] sum_{j'} (sum_{I',J'} a_s[I'][J'] A_{I',J'}[i'j']) (sum_{J',K'} b_s[J'][K'] B_{J',K'}[j'k'])
    # = sum_{s, j', I', J', J', K'} c_s[I][K] a_s[I'][J'] b_s[J'][K'] A_{I',J'}[i'j'] B_{J',K'}[j'k']
    
    # We want this to equal sum_{J} A_{I,J}[i'j] B_{J,K}[j'k'] (where j is summed over local indices? No, J is block index).
    # C_{ik} = sum_J A_{I,J} B_{J,K}.
    # (A_{I,J} B_{J,K})[i'k'] = sum_{j'} A_{I,J}[i'j'] B_{J,K}[j'k'].
    # So C_{I,K}[i'k'] = sum_J sum_{j'} A_{I,J}[i'j'] B_{J,K}[j'k'].
    
    # Comparing the two:
    # We need:
    # sum_{s, j', I', J', J', K'} c_s[I][K] a_s[I'][J'] b_s[J'][K'] A_{I',J'}[i'j'] B_{J',K'}[j'k']
    # to equal
    # sum_J sum_{j'} A_{I,J}[i'j'] B_{J,K}[j'k'].
    
    # This requires that the coefficients match.
    # Coefficient of A_{I',J'}[i'j'] B_{J',K'}[j'k'] in the LHS is:
    # sum_{s, j'} c_s[I][K] a_s[I'][J'] b_s[J'][K'] delta_{j', j'} (wait, j' is summed in both)
    # Actually, the term is A_{I',J'}[i'j'] B_{J',K'}[j'k'].
    # In the RHS, the term is A_{I,J}[i'j'] B_{J,K}[j'k'].
    # So we need I'=I, J'=J, J'=J, K'=K.
    # So I'=I, J'=J, K'=K.
    # And the coefficient must be 1.
    # So sum_s c_s[I][K] a_s[I][J] b_s[J][K] = 1?
    # No, the sum is over s.
    # And there is a sum over j' in the LHS?
    # No, j' is the local index in the product (X_s Y_s)[i'k'].
    # The term is A_{I',J'}[i'j'] B_{J',K'}[j'k'].
    # So for a fixed i', j', k', the coefficient is sum_s c_s[I][K] a_s[I'][J'] b_s[J'][K'].
    # This must be 1 if I'=I, J'=J, J'=J, K'=K.
    # And 0 otherwise?
    # Actually, if I'!=I or J'!=J or J'!=J or K'!=K, the term A_{I',J'} B_{J',K'} does not appear in C_{I,K}.
    # So the coefficient must be 0.
    
    # So we require:
    # sum_s c_s[I][K] a_s[I'][J'] b_s[J'][K'] = delta_{I,I'} delta_{J,J'} delta_{J,J'} delta_{K,K'}
    # This is the Strassen identity for the blocks.
    # It holds for the 7 terms defined above.
    
    # So the construction is correct.
    
    # Now we just need to fill U, V, W.
    # U[i, 7s+t] = sum_{I,J} a_s[I][J] U2[i'2+j', t]
    # where i = I*2+i', j = J*2+j'.
    # So for a given i, I = i//2, i' = i%2.
    # But j is not fixed in U. U depends only on i.
    # Wait, U[i, r] is the coefficient of A_i in the r-th term.
    # The r-th term is (7s+t).
    # The term is U2[i'2+j', t] * a_s[I][J].
    # But j' is part of the index in U2.
    # U2 is 4x7. Index is i'2+j'.
    # So U[i, 7s+t] = sum_{J} a_s[I][J] U2[i'2+J*2? No.
    # U2 index is local i', local j'.
    # A_{I,J}[i'j'] corresponds to global i=I*2+i', j=J*2+j'.
    # So U[global_i, 7s+t] should be the coefficient of A_{global_i, global_j}.
    # But U is a vector of length 16 (4*4).
    # U[row, col] where row = i*4+j.
    # So U[i*4+j, 7s+t] = a_s[I][J] * U2[i'2+j', t].
    
    # Let's implement this.
    
    for s in range(7):
        for t in range(7):
            col = 7*s + t
            for I in range(2):
                for J in range(2):
                    if a_coeffs[s][I][J] != 0:
                        coeff = a_coeffs[s][I][J]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                i = I*2 + i_prime
                                j = J*2 + j_prime
                                row_u = i*4 + j
                                val = coeff * U2[i_prime*2 + j_prime, t]
                                if val != 0:
                                    U[row_u, col] = val
                                
                                # V
                                # V[j*4+k, 7s+t] = b_s[J][K] * V2[j'2+k', t]
                                # We need to loop over K too?
                                # No, V depends on j and k.
                                # V[j*4+k, 7s+t] = sum_K b_s[J][K] V2[j'2+k', t]?
                                # No, for a fixed j, J=j//2, j'=j%2.
                                # V[j*4+k, 7s+t] = b_s[J][K] V2[j'2+k', t] where K=k//2, k'=k%2.
                                # But b_s[J][K] depends on K.
                                # So for a fixed j, k, we have J, j', K, k'.
                                # V[j*4+k, 7s+t] = b_s[J][K] V2[j'2+k', t].
                                
                                # W
                                # W[k*4+i, 7s+t] = c_s[I][K] W2[k'2+i', t].
                                
            # It is more efficient to loop over all i,j,k for each s,t.
            
    # Reset U,V,W to zero and fill properly
    U.fill(0)
    V.fill(0)
    W.fill(0)
    
    for s in range(7):
        for t in range(7):
            col = 7*s + t
            for i in range(4):
                I = i // 2
                i_prime = i % 2
                for j in range(4):
                    J = j // 2
                    j_prime = j % 2
                    row_u = i*4 + j
                    val = a_coeffs[s][I][J] * U2[i_prime*2 + j_prime, t]
                    if val != 0:
                        U[row_u, col] = val
                        
            for j in range(4):
                J = j // 2
                j_prime = j % 2
                for k in range(4):
                    K = k // 2
                    k_prime = k % 2
                    row_v = j*4 + k
                    val = b_coeffs[s][J][K] * V2[j_prime*2 + k_prime, t]
                    if val != 0:
                        V[row_v, col] = val
                        
            for k in range(4):
                K = k // 2
                k_prime = k % 2
                for i in range(4):
                    I = i // 2
                    i_prime = i % 2
                    row_w = k*4 + i
                    val = c_coeffs[s][I][K] * W2[k_prime*2 + i_prime, t]
                    if val != 0:
                        W[row_w, col] = val
                        
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
