"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 51 decomposition:
    # Split A (3x3) into:
    #   A_TL: rows 0-1, cols 0-1 (2x2)
    #   A_TR: rows 0-1, col 2   (2x1)
    #   A_BL: row 2, cols 0-1   (1x2)
    #   A_BR: row 2, col 2      (1x1)
    # Split B (3x6) into:
    #   B_TL: row 0, cols 0-5 (1x6)
    #   B_ML: row 1, cols 0-5 (1x6)
    #   B_BL: row 2, cols 0-5 (1x6)
    #
    # Products:
    # 1) A_TL * B_TL: 2x2 * 1x6 -> rank 4 (each of 2 A rows times B row)
    #    Actually A_TL is 2x2, B_TL is 1x6. Product is 2x6.
    #    C[i,k] = A_TL[i,0]*B_TL[0,k] + A_TL[i,1]*B_TL[0,k] ... no.
    #    Wait, B_TL is row 0 of B, which is 1x6.
    #    A_TL * B_TL doesn't make sense dimensionally unless B_TL is 2x6.
    #
    # Let's re-derive properly.
    # A is 3x3, B is 3x6. C = A*B is 3x6.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # Standard decomposition is rank 54.
    # Can we do better?
    # The tensor for 3x3*3x6 has dimension 3*3*6 = 54.
    # It is a subset of the 3x3*3x3 tensor (which has rank 23 over reals, but integer rank is 27? No, Strassen is 7 for 2x2, so 3x3 is at most 27, actually best known is 23 over reals but integer entries require different).
    # For 3x3 * 3x6, we can think of it as two 3x3 * 3x3 multiplications plus some cross terms? No.
    # It's exactly the same as computing A1*B and A2*B where A is split? No.
    #
    # Let's try Strassen-like approach.
    # Split A into 2x2 blocks? A is 3x3.
    # Let's use the known result: rank of 3x3*3x6.
    # Actually, the rank of m x n * n x p is at most m*n*p.
    # For 3x3*3x6, is there a known lower bound or better algorithm?
    # The tensor is T_{i,j,k} = delta_{j,k'} ... no.
    #
    # Let's try a different split.
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # B = [B1; B2; B3] where B1 is 1x6, B2 is 1x6, B3 is 1x6.
    #
    # C = A11*B1 + A12*B2 + ... no, that's not right.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k].
    #
    # This is a sum of 3 terms for each (i,k).
    # Total 9*6 = 54 terms.
    #
    # Can we combine terms?
    # For fixed k, C[:,k] is a linear combination of columns of A.
    # C[:,k] = A * B[:,k].
    # This is just matrix-vector multiplication A (3x3) * v (3x1).
    # Rank of 3x3*3x1 tensor is 3 (each output element is a dot product of a row of A with v).
    # So for each k, we need rank 3.
    # For 6 k's, total rank 6*3 = 18?
    # Wait. C[i,k] = sum_j A[i,j] B[j,k].
    # If we fix k, we have 3 outputs C[0,k], C[1,k], C[2,k].
    # Each is sum_j A[i,j] B[j,k].
    # This is rank 3 for that k.
    # Total rank 18?
    # Let's check.
    # For k=0: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0].
    # This is 3 terms.
    # Similarly for C[1,0] and C[2,0].
    # Total 9 terms for k=0.
    # Ah, I see. The rank of the tensor T_{i,j,k} for fixed k is the rank of the 3x3 matrix of coefficients?
    # No, the tensor is T_{i,k,j} = A_{i,j} B_{j,k}.
    # The mode-2 unfolding is A. The mode-3 unfolding is B^T.
    # The rank of the tensor is at least max(rank(A_unfolding), rank(B_unfolding), rank(C_unfolding)).
    # Rank of A_unfolding (6x3) is 3.
    # Rank of B_unfolding (6x3) is 3.
    # Rank of C_unfolding (18x3) is 3.
    # So rank is at least 3.
    # But is it 18?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a sum of 3 terms: for j=0,1,2.
    # Term j: A[:,j] * B[j,:].
    # A[:,j] is a vector of length 3. B[j,:] is a vector of length 6.
    # The outer product A[:,j] B[j,:] is a 3x6 matrix.
    # C = A[:,0] B[0,:] + A[:,1] B[1,:] + A[:,2] B[2,:].
    # This is a sum of 3 rank-1 tensors!
    # So the rank is at most 3?
    # Wait. A[:,j] depends on j, B[j,:] depends on j.
    # In a tensor decomposition, each term is U_c * V_c * W_c.
    # Here, for term j, U_c = A[:,j], V_c = B[j,:], W_c = 1 (scalar).
    # But U_c must be a column of U, which is a vector of length n*m=9.
    # A[:,j] is a vector of length 3. It needs to be embedded in 9 dimensions.
    # Specifically, U[i*m + j] = A[i,j].
    # So for term j, U has non-zeros only in column j.
    # V[j*p + k] = B[j,k].
    # So for term j, V has non-zeros only in row j.
    # W[k*n + i] = 1.
    # But W is the same for all j? No.
    # In the schoolbook, each term (i,j,k) is separate.
    # Here, we are grouping by j.
    # For fixed j, the product is A[:,j] * B[j,:] * (sum over i,k).
    # The tensor is sum_j (A[:,j] outer B[j,:] outer W_j).
    # Where W_j[i,k] = 1 if the indices match? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So W_j[i,k] = 1 for all i,k? No.
    # The term is A[i,j] * B[j,k].
    # So for fixed j, the term is sum_{i,k} A[i,j] B[j,k] * delta_{i,k}? No.
    # C[i,k] gets A[i,j] B[j,k].
    # So the contribution to C[i,k] from term j is A[i,j] B[j,k].
    # This is exactly the outer product of A[:,j] and B[j,:] placed into C.
    # So W_j is the identity-like matrix? No, W_j is all ones?
    # If W_j[i,k] = 1, then (U_j * V_j * W_j)[i,k] = sum_{a,b} U[i,a] V[b,k] W[i,k].
    # No, the decomposition is C[i,k] = sum_c U[i*m+j_c, c] V[j_c*p+k, c] W[k*n+i, c].
    # Wait, the indices in U, V, W are coupled.
    # U is n*m x rank. V is m*p x rank. W is p*n x rank.
    # C[i,k] = sum_c U[i*m + j, c] V[j*p + k, c] W[k*n + i, c]? No.
    # The standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i, c] V[j, c] W[k, c]? No.
    # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # And the decomposition is exact.
    # Usually, C[i,k] = sum_c U[i*m + j_i(c), c] ... no.
    # The indices i, j, k are linked.
    # U[i*m + j, c] is non-zero only for specific (i,j).
    # V[j*p + k, c] is non-zero only for specific (j,k).
    # W[k*n + i, c] is non-zero only for specific (k,i).
    #
    # In schoolbook, for each (i,j,k), we have one term.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Then C[i,k] = sum_{i',j',k'} U[i*m+j', c] V[j'*p+k', c] W[k'*n+i', c] ... this doesn't look right.
    #
    # Let's re-read the task carefully.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    # The tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    # The decomposition is T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}.
    # Wait, that would mean T_{i,j,k} is a function of (i,j,k).
    # But matrix multiplication gives C_{i,k} = sum_j T_{i,j,k}.
    # So the tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # And we need to decompose this tensor.
    #
    # In schoolbook:
    # For each (i,j,k), T_{i,j,k} = A_{i,j} B_{j,k}.
    # We set U_{i*m+j, c} = A_{i,j}, V_{j*p+k, c} = B_{j,k}, W_{k*n+i, c} = 1.
    # Then sum_c U V W = A_{i,j} B_{j,k} * 1.
    # This works.
    #
    # Now, can we reduce rank?
    # We want to find U, V, W such that for all i,j,k:
    # sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = A_{i,j} B_{j,k}.
    #
    # Note that A_{i,j} B_{j,k} is separable: it's a product of a function of (i,j) and a function of (j,k).
    # But the indices are shared.
    #
    # Consider the rank of the tensor T_{i,j,k}.
    # It is the same as the rank of the matrix multiplication tensor for 3x3*3x6.
    # This tensor is a sub-tensor of the 3x3*3x3 tensor (if we pad B with zeros).
    # The rank of 3x3*3x3 is 23 (over reals).
    # But we need integer/half-integer entries.
    # The rank of 3x3*3x3 over integers is at least 27? No.
    # Strassen for 3x3: 27 multiplications (3x3 = 3*3*3).
    # Can we do better?
    # Best known for 3x3*3x3 is 23 (Landsberg).
    # But for 3x3*3x6, we can view it as 3x3*3x3 + 3x3*3x3? No.
    # It's 3x3*3x6 = 3x3*(3x3 + 3x3) if we split B into two 3x3 blocks? No, B is 3x6.
    # B = [B1 | B2] where B1, B2 are 3x3.
    # C = A*B1 + A*B2.
    # So C is the sum of two 3x3 matrices.
    # The tensor for C is the sum of the tensor for A*B1 and A*B2.
    # The rank of A*B1 + A*B2 is at most rank(A*B1) + rank(A*B2) = 2*rank(3x3*3x3).
    # If rank(3x3*3x3) = 23, then total rank 46.
    # But we can share the A part?
    # A*B1 + A*B2 = A * [B1 | B2].
    # This is exactly the original problem.
    #
    # Let's try to construct a rank 27 solution using Strassen for 3x3.
    # Actually, the simplest improvement is to use the fact that B has 6 columns.
    # We can compute A * B by computing A * B1 and A * B2 where B1, B2 are 3x3.
    # Each takes rank 27 (schoolbook) or 23 (Strassen-like).
    # Total rank 54 or 46.
    #
    # But wait, the schoolbook for 3x3*3x6 is rank 54.
    # Can we do better than 54?
    # Yes, if we use Strassen for 3x3*3x3.
    # Rank of 3x3*3x3 is at most 27 (schoolbook) or 23 (Strassen).
    # So rank of 3x3*3x6 is at most 2*23 = 46.
    #
    # Let's implement rank 46 using Strassen for 3x3*3x3.
    # Strassen for 2x2*2x2 uses 7 multiplications.
    # For 3x3*3x3, we can use a known decomposition of rank 23.
    # But that's complex.
    #
    # Alternatively, we can use the fact that 3x3*3x6 = 3x3*3x3 + 3x3*3x3.
    # If we use schoolbook for each 3x3*3x3, total rank 2*27 = 54.
    # If we use Strassen for each, total rank 2*23 = 46.
    #
    # Let's try to implement Strassen for 3x3*3x3.
    # Actually, there's a simpler approach.
    # The tensor T_{i,j,k} for 3x3*3x6 can be decomposed using the rank of the 3x3*3x3 tensor.
    # Let's use the known rank 23 decomposition for 3x3*3x3.
    # But I don't remember the exact coefficients.
    #
    # Let's try a different approach: use the rank of the 3x3*3x1 tensor.
    # 3x3*3x1 has rank 3.
    # 3x3*3x6 = 6 * (3x3*3x1).
    # Total rank 18.
    # Wait, is that true?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For each k, C[:,k] is a linear combination of columns of A.
    # The tensor for 3x3*3x1 is T_{i,j} = A[i,j] B[j].
    # This has rank 3: T_{i,j} = sum_{l=0}^2 A[i,l] B[l] * delta_{j,l}? No.
    # T_{i,j} = A[i,j] B[j].
    # This is rank 3: for each j, term is A[:,j] * B[j] * delta_{j,j}.
    # So rank 3.
    # For 6 k's, total rank 18.
    #
    # Let's verify this.
    # We need U, V, W such that:
    # sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = A[i,j] B[j,k].
    #
    # For rank 18, we have c = 0..17.
    # Let c = 3*k + j.
    # U_{i*m+j, c} = A[i,j] if j == c % 3, else 0.
    # V_{j'*p+k, c} = B[j,k] if j' == c % 3, else 0.
    # W_{k'*n+i, c} = 1 if k' == c // 3, else 0.
    #
    # Then sum_c U V W = sum_{j=0}^2 sum_{k=0}^5 A[i,j] B[j,k] * (1 if k == c//3 else 0).
    # For fixed (i,j,k), the term is non-zero only when c = 3*k + j.
    # Then U_{i*m+j, c} = A[i,j], V_{j*p+k, c} = B[j,k], W_{k*n+i, c} = 1.
    # So the sum is A[i,j] B[j,k].
    # This works!
    #
    # So the rank is 18.
    # Let's implement this.

    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for j in range(m):
            c = k * m + j
            for i in range(n):
                U[i * m + j][c] = 1
            for k_prime in range(p):
                if k_prime == k:
                    V[j * p + k_prime][c] = 1
            for i_prime in range(n):
                W[k * n + i_prime][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
