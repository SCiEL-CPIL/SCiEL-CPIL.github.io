import numpy as np

def construct():
    n, m, p = 4, 5, 5
    # We will build the decomposition for A (4x5) * B (5x5) = C (4x5)
    # C[:, 0:4] = A[:, 0:4] * B[0:4, 0:4] + A[:, 4] * B[4, 0:4]
    # C[:, 4] = A[:, 0:4] * B[0:4, 4] + A[:, 4] * B[4, 4]
    
    terms = [] # List of (i, j, k, coeff, w_index)
    
    # 1. A1 * B11 (4x4 * 4x4) using Strassen
    # We need the Strassen decomposition for 4x4
    # We'll generate the terms for M4
    def get_strassen_terms(n):
        if n == 1:
            return [(0, 0, 0, 1)] # i, j, k, coeff for C[i,k]
        
        half = n // 2
        # The 7 Strassen products for 2x2 blocks
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map the local indices to global indices
        # This function returns terms for a 2x2 block of size n
        # The terms are (i, j, k, coeff) where i, j, k are in [0, n)
        # And coeff is the contribution to C[i, k]
        
        # We'll define the 7 products as lists of (i, j, k, coeff_in_product)
        # And then map them to C[i, k]
        
        # This is getting complex to do recursively with coefficients.
        # Let's just hardcode the 49 terms for 4x4 or use a known list.
        # Actually, we can just use the schoolbook for 4x4 if Strassen is too hard to implement correctly in this format.
        # But 49 is much better than 64.
        
        # Let's try to implement the recursive step properly.
        # For n=2, we have 7 products.
        # Each product is a sum of 4 terms (for n=1).
        # For n=4, each product is a sum of 4 products of n=2.
        # So 7 * 4 * 7 * 4 = 784 terms? No.
        # The rank is 49.
        # Each rank-1 term is a single (i, j, k).
        # So there are 49 such terms.
        
        # Let's generate the 49 terms.
        pass

    # Given the complexity, let's use a simpler approach:
    # Use schoolbook for the 4x4 part? Rank 64.
    # Total rank 64 + 1 + 1 + 1 = 67.
    # This is still better than 100.
    # And much easier to implement correctly.
    
    # Let's use schoolbook for the 4x4 block.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A[i, j] * B[j, k] contributes to C[i, k]
                # U index: i * 5 + j
                # V index: j * 5 + k
                # W index: k * 4 + i
                terms.append((i, j, k, 1))
                
    # 2. A2 * B21 (4x1 * 1x4)
    # C[i, k] += A[i, 4] * B[4, k]
    # This is rank 1.
    # U index: i * 5 + 4
    # V index: 4 * 5 + k
    # W index: k * 4 + i
    # We need one term for each i, k? No, it's rank 1.
    # Wait, the rank of M_{4,1,4} is 1?
    # C[i, k] = A[i, 0] B[0, k].
    # This is a single term in the tensor decomposition?
    # No, it's a sum over r.
    # If rank=1, then C[i, k] = U[i, 0] V[k, 0] W[k, i, 0].
    # U[i, 0] = A[i, 0]. V[k, 0] = B[0, k]. W[k, i, 0] = 1.
    # This works.
    # So we add 1 term?
    # But U, V, W are matrices.
    # U has 20 rows. V has 25 rows. W has 20 rows.
    # The term is (i, j, k) with j=4.
    # U[i * 5 + 4, r] = 1.
    # V[4 * 5 + k, r] = 1.
    # W[k * 4 + i, r] = 1.
    # This is for a specific i, k?
    # No, the term is for all i, k.
    # So we add 1 term to the list?
    # No, the term is (i, j, k).
    # If we add (0, 4, 0, 1), it only covers i=0, k=0.
    # We need to cover all i, k.
    # So we need 16 terms?
    # No, the rank is 1.
    # This means we can do it with 1 multiplication.
    # But the multiplication is A[i, 4] * B[4, k].
    # This depends on i and k.
    # So it's not a single rank-1 term.
    # It is a sum of 16 rank-1 terms?
    # No, the rank of M_{n, 1, p} is n.
    # So for n=4, p=4, the rank is 4.
    # So we need 4 terms.
    # Let's verify.
    # C[i, k] = A[i, 0] B[0, k].
    # We can write this as sum_{r=0}^3 U[i, r] V[k, r] W[k, i, r].
    # If we set U[i, r] = delta_{i, r}, V[k, r] = B[0, k], W[k, i, r] = 1.
    # Then sum_{r} delta_{i, r} B[0, k] 1 = B[0, k] if i=r else 0.
    # This gives B[0, k] only when i=r.
    # So C[i, k] = B[0, k] for all i? No.
    # C[i, k] should be A[i, 0] B[0, k].
    # So we need U[i, r] = A[i, 0] if r=i?
    # Let's try rank 4.
    # Term r: U[i, r] = delta_{i, r}. V[k, r] = 1. W[k, i, r] = A[i, 0] B[0, k]? No, W is constant.
    
    # Actually, the rank of M_{n, 1, p} is n.
    # The decomposition is:
    # For r in 0..n-1:
    #   U[r * 1 + 0, r] = 1 (A is n x 1, so index is r)
    #   V[0 * p + k, r] = 1 (B is 1 x p, so index is k)
    #   W[k * n + r, r] = 1
    #   Then C[r, k] = U[r, r] V[k, r] W[k, r, r] = 1 * 1 * 1 = 1.
    #   This gives C[r, k] = 1.
    #   But we want C[i, k] = A[i, 0] B[0, k].
    #   So we need to multiply by A[i, 0] and B[0, k].
    #   But U, V, W are constant.
    #   So we can't do it with rank n if A and B are variables.
    #   Wait, the tensor decomposition is for the multiplication map.
    #   The inputs A and B are variables.
    #   So U, V, W are fixed matrices.
    #   The product is C = sum_r U[:, r] V[:, r]^T W[:, r]?
    #   No, C[i, k] = sum_r U[i, r] V[k, r] W[k, i, r].
    #   If U, V, W are constant, then C is a fixed linear combination of A and B?
    #   No, A and B are the inputs.
    #   The decomposition is A * B = sum_r (U[:, r] * A) (V[:, r] * B) W[:, r]?
    #   No, the standard form is:
    #   C_{i, k} = sum_r U_{i, r} V_{j, r} W_{k, r} A_{i, j} B_{j, k}?
    #   No, the indices are:
    #   U is (n*m) x rank.
    #   V is (m*p) x rank.
    #   W is (p*n) x rank.
    #   C_{i, k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} A_{i, j} B_{j, k}?
    #   No, the standard form is:
    #   C_{i, k} = sum_r U_{i, r} V_{k, r} W_{i, k, r} A_{i, j} B_{j, k}?
    #   The correct form is:
    #   C_{i, k} = sum_r (sum_j U_{i*m+j, r} A_{i, j}) (sum_j V_{j*p+k, r} B_{j, k}) W_{k*n+i, r}
    #   No, that's not right either.
    
    # The correct form is:
    # C_{i, k} = sum_r U_{i, r} V_{k, r} W_{i, k, r}
    # Where U, V, W are the factor matrices.
    # But U is (n*m) x rank.
    # So U_{i*m+j, r}.
    # The term is U_{i*m+j, r} A_{i, j} V_{j*p+k, r} B_{j, k} W_{k*n+i, r}.
    # This is not a standard form.
    
    # The standard form for matrix multiplication tensor is:
    # T_{i, j, k, l, m, n} = delta_{j, m} delta_{k, l}
    # And the decomposition is:
    # T = sum_r U_r \otimes V_r \otimes W_r
    # Where U_r is a vector of size n*m, V_r is a vector of size m*p, W_r is a vector of size p*n.
    # And the product is:
    # C_{i, k} = sum_r (sum_j U_{i*m+j, r} A_{i, j}) (sum_j V_{j*p+k, r} B_{j, k}) W_{k*n+i, r}
    # No, that's not right.
    
    # The correct form is:
    # C_{i, k} = sum_r U_{i, r} V_{k, r} W_{i, k, r}
    # Where U_{i, r} = sum_j U_{i*m+j, r} A_{i, j}
    # This is getting too complicated.
    
    # Let's stick to the schoolbook decomposition for the whole thing.
    # Rank 100.
    # It's safe and correct.
    
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
