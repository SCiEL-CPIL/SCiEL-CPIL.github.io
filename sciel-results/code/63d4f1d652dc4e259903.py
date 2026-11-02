"""Rewritten program: exploit direct-sum structure M_{2,2,8} = 4 x M_{2,2,2} with Strassen."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # The 2x8 matrix B can be viewed as four 2x2 blocks: B[:, 2t:2t+2] for t=0..3.
    # The product C = A @ B is a 2x8 matrix; each 2x2 block C[:, 2t:2t+2] = A @ B_t.
    # Thus the tensor M_{2,2,8} is the direct sum of four copies of M_{2,2,2}.
    # We use Strassen's rank-7 algorithm for each 2x2 block.
    #
    # Strassen's 7 products for 2x2 matrix multiplication:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # For the bilinear form, we need U, V, W such that:
    #   C[k*n+i, r] = sum_c U[i*m+j, c] * V[j*p+k, c] ... wait, let me re-read the convention.
    #
    # The convention is: W[k*n+i, r] is the output factor.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    #
    # For matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p:
    #   C[i, k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T[i,k,j] = delta (i.e., T is the multiplication tensor).
    # In the trace convention W[k*n+i, r], the output index is (k, i) mapped to k*n+i.
    # So W[k*n+i, r] corresponds to coefficient of C[i,k] (row i, col k).
    #
    # U[i*m+j, r] corresponds to coefficient of A[i,j] (row i, col j).
    # V[j*p+k, r] corresponds to coefficient of B[j,k] (row j, col k).
    #
    # So the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # No, that's not right. The standard tensor decomposition of matrix mult:
    #   T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    #
    # Actually, the multiplication tensor for C = A*B (n x m times m x p):
    #   T[i, k, j] = 1  (for all valid i, j, k)
    # meaning C[i,k] = sum_j A[i,j] B[j,k]
    #
    # In the factorization: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has index (i*m+j), V has index (j*p+k), W has index (k*n+i).
    #
    # So: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T should equal 1 when the indices are consistent (i.e., this is the identity tensor for matmul).
    #
    # For Strassen on a 2x2 block:
    # A is 2x2, B is 2x2, C = A@B is 2x2.
    # n=2, m=2, p=2 for each block.
    #
    # U indices: i*m+j where i in {0,1}, j in {0,1} → 0,1,2,3
    # V indices: j*p+k where j in {0,1}, k in {0,1} → 0,1,2,3
    # W indices: k*n+i where k in {0,1}, i in {0,1} → 0,1,2,3
    #
    # Strassen's 7 terms:
    # M1 = (A00+A11)(B00+B11): 
    #   U: A00→1, A11→1; V: B00→1, B11→1; W: contributes to C00(+1), C11(+1)
    # M2 = (A10+A11)B00:
    #   U: A10→1, A11→1; V: B00→1; W: C10(+1), C11(-1)
    # M3 = A00(B01-B11):
    #   U: A00→1; V: B01→1, B11→-1; W: C00(-1), C01(+1)
    # M4 = A11(B10-B00):
    #   U: A11→1; V: B10→1, B00→-1; W: C00(+1), C10(+1)
    # M5 = (A00+A01)B11:
    #   U: A00→1, A01→1; V: B11→1; W: C00(-1), C01(+1)
    # M6 = (A10-A00)(B00+B01):
    #   U: A10→1, A00→-1; V: B00→1, B01→1; W: C11(+1)
    # M7 = (A01-A11)(B10+B11):
    #   U: A01→1, A11→-1; V: B10→1, B11→1; W: C00(+1), C11(+1)
    #
    # Let me verify C00 = M1 + M4 - M5 + M7:
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # M4: A11B10 - A11B00
    # -M5: -A00B11 - A01B11
    # M7: A01B10 + A01B11 - A11B10 - A11B11
    # Sum: A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A01B10 ✓ (that's C[0,0])
    #
    # C01 = M3 + M5:
    # M3: A00B01 - A00B11
    # M5: A00B11 + A01B11
    # Sum: A00B01 + A01B11 ✓ (that's C[0,1])
    #
    # C10 = M2 + M4:
    # M2: A10B00 + A11B00
    # M4: A11B10 - A11B00
    # Sum: A10B00 + A11B10 ✓ (that's C[1,0])
    #
    # C11 = M1 - M2 + M3 + M6:
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # -M2: -A10B00 - A11B00
    # M3: A00B01 - A00B11
    # M6: A10B00 + A10B01 - A00B00 - A00B01
    # Sum: A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A11B11 + A10B01 ✓ (that's C[1,1])
    #
    # Great, Strassen is verified. Now build the full 2x2x8 decomposition.
    #
    # For the full problem: n=2, m=2, p=8.
    # B is 2x8, so B[j,k] for j in {0,1}, k in {0,...,7}.
    # We split B into 4 blocks: B_t = B[:, 2t:2t+2] for t=0,1,2,3.
    # C = A @ B, so C[i, 2t+s] = sum_j A[i,j] B[j, 2t+s].
    #
    # U has shape (2*2, rank) = (4, rank)
    # V has shape (2*8, rank) = (16, rank)
    # W has shape (8*2, rank) = (16, rank)
    #
    # For each block t (0..3), we have 7 Strassen terms. Total rank = 4*7 = 28.
    #
    # U index: i*m+j = i*2+j, i in {0,1}, j in {0,1} → 0,1,2,3
    # V index: j*p+k = j*8+k, j in {0,1}, k in {0,...,7}
    # W index: k*n+i = k*2+i, k in {0,...,7}, i in {0,1}
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    num_blocks = p // 2  # 4 blocks of size 2x2
    
    # Strassen coefficients for 2x2 matmul
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) -> coeff
    # v_coeffs: dict mapping (j,k) -> coeff (for the 2x2 block)
    # w_coeffs: dict mapping (k,i) -> coeff (for the 2x2 block)
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for (u_coeff, v_coeff, w_coeff) in strassen_terms:
            # Fill U: index i*m+j
            for (i, j), c in u_coeff.items():
                U[i * m + j][col] += c
            
            # Fill V: index j*p+k, where k is local (0 or 1) mapped to global 2t+k
            for (j, k_local), c in v_coeff.items():
                k_global = 2 * t + k_local
                V[j * p + k_global][col] += c
            
            # Fill W: index k*n+i, where k is local (0 or 1) mapped to global 2t+k
            for (k_local, i), c in w_coeff.items():
                k_global = 2 * t + k_local
                W[k_global * n + i][col] += c
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
