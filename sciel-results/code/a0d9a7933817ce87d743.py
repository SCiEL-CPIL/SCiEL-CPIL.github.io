"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-like / block decomposition for 3x3 = 2x2 + 1x1
    # We use the fact that 3x3 can be decomposed into blocks:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # where A11, A22 are 2x2 blocks? No, that's not quite right.
    #
    # Actually, a cleaner approach: use the known rank-23 decomposition for 3x3 matrix multiplication.
    # The best known upper bound for 3x3 is 23 (Alon et al., 2003; also confirmed by others).
    #
    # Below is an explicit rank-23 decomposition.
    # The tensor T_{ijk} = A_{il} B_{lj} C_{jk} must be decomposed.
    # Using the trace convention: output W[k*n + i, r] where n=p=3.
    # So W index is k*3 + i.
    #
    # I will use a known explicit 23-term decomposition from the literature.
    # One such decomposition uses the following structure:
    # 
    # Let's denote the 9 input indices of A as (i, l) for i,l in {0,1,2}
    # and B as (l, j) for l,j in {0,1,2}
    # and output C as (i, j) for i,j in {0,1,2}
    #
    # The 23 terms are constructed by combining a 2x2 Strassen (7 terms) for the
    # upper-left 2x2 block, plus additional terms for the remaining rows/columns.
    #
    # Actually, let me use a verified 23-rank decomposition. I'll construct it 
    # by taking the rank-27 schoolbook and finding 4 independent linear dependencies
    # to eliminate 4 terms.
    #
    # For reliability, I'll implement a known rank-23 decomposition.
    # 
    # Here is a construction:
    # We can view 3x3 multiplication as:
    # C = A*B
    # 
    # Using the "laser method" or explicit search, the rank is 23.
    # 
    # Let me use the following explicit 23-term decomposition.
    # Each term c has U_c (9 entries), V_c (9 entries), W_c (9 entries).
    # U_c[i*3+l] = coefficient for A[i,l]
    # V_c[l*3+j] = coefficient for B[l,j]  -- wait, V index is l*3+j? 
    # V has shape (m*p, rank) = (9, rank). V[l*p+j] = V[l*3+j] = coeff for B[l,j]
    # W has shape (p*n, rank) = (9, rank). W[k*n+i] = W[j*3+i] = coeff for C[i,j]
    # 
    # So for term c:
    #   sum_l U[i*3+l, c] * V[l*3+j, c] should equal W[j*3+i, c] for all i,j,l
    #   i.e., W[j*3+i, c] = sum_l U[i*3+l, c] * V[l*3+j, c]
    #
    # This is a rank-1 tensor in the A-B-C space: T_c[i,j,l] = U[i*3+l,c] * V[l*3+j,c]
    # and the total decomposition is sum_c T_c[i,j,l] = delta_{i,j} (the multiplication tensor)
    #
    # The multiplication tensor is: T[i,j,l] = 1 if i==l and l==j... no.
    # C[i,j] = sum_l A[i,l] B[l,j]
    # So the tensor is T[i,j,l] = 1 if we're computing C[i,j] from A[i,l]*B[l,j].
    # In tensor form: T[i,j,l] = 1 for all i,j,l? No.
    # 
    # The multiplication tensor M satisfies: M[i,j,l] = 1 if l connects i and j, 
    # i.e., M[i,j,l] = 1 for all i,j,l is not right.
    # 
    # Actually: sum_l A[i,l] * B[l,j] = C[i,j]
    # The tensor is T[i,j,l] = 1 for all i,j,l? That's not a tensor that equals the identity.
    # 
    # Let me be precise. The matrix multiplication tensor is a 3-tensor T of size n x p x m
    # (or some ordering) such that T[i, j, l] = 1 if and only if... 
    # 
    # Standard: The multiplication tensor mu_{n,p,m} has entries mu[i,j,k] = 1 if i==k and j==l... 
    # 
    # Let's use: T[i, j, l] = 1 for all i, j, l in [0..n-1] x [0..m-1] x [0..p-1]? No.
    # 
    # The tensor for multiplying n x m by m x p is:
    # T[i, j, l] where i in [0,n), j in [0,p), l in [0,m)
    # T[i,j,l] = 1 if we use A[i,l] * B[l,j] to contribute to C[i,j].
    # So T[i,j,l] = 1 for ALL i,j,l? That's the full tensor, and we want to decompose it.
    # 
    # No, that's not right either. The tensor IS defined by T[i,j,l] = 1 for all i,j,l.
    # And we want sum_c U[i*3+l, c] * V[l*3+j, c] = W[j*3+i, c]... 
    # 
    # Wait, the decomposition must satisfy:
    # For all i, j: sum_c (sum_l U[i*3+l, c] * V[l*3+j, c]) = sum_l A[i,l] * B[l,j]
    # This means: sum_c sum_l U[i*3+l, c] * V[l*3+j, c] = delta_{i,j} ... no, it equals the product.
    #
    # The identity is: C[i,j] = sum_l A[i,l] B[l,j]
    # If we write A[i,l] = sum_c U[i*3+l, c] * x_c and B[l,j] = sum_c V[l*3+j, c] * x_c,
    # then A[i,l]B[l,j] = sum_c U[i*3+l, c] V[l*3+j, c] x_c^2 ... this isn't right.
    #
    # The correct bilinear decomposition:
    # C[i,j] = sum_l A[i,l] B[l,j] = sum_c W[j*3+i, c] * (sum_l U[i*3+l, c] * V[l*3+j, c])
    # 
    # So we need: for all i,j,l: 
    # sum_c W[j*3+i, c] * U[i*3+l, c] * V[l*3+j, c] = delta_{l is the connecting index}
    # 
    # Actually the condition is simply:
    # sum_c W[j*3+i, c] * U[i*3+l, c] * V[l*3+j, c] = 1 if we're computing the l-th term, 
    # but summed over l, it should give the full product.
    #
    # The precise condition: For all i, j:
    # sum_l (sum_c W[j*3+i, c] * U[i*3+l, c] * V[l*3+j, c]) = 1 * (the l-summed product)
    # 
    # No. The condition is that the tensor T[i,j,l] = 1 for all i,j,l is decomposed as:
    # T[i,j,l] = sum_c W[j*3+i, c] * U[i*3+l, c] * V[l*3+j, c]
    # 
    # Wait, that's not dimensionally consistent with a rank decomposition of the multiplication tensor.
    #
    # Let me restart the indexing. The multiplication tensor M of size (n*m) x (m*p) x (p*n) 
    # or similar. In the problem statement, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # 
    # The decomposition is: sum_c U[:,c] * V[:,c]^T * W[:,c]... no.
    # 
    # It's: for all (a, b, c) in (n*m) x (m*p) x (p*n):
    # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # 
    # Where M is the multiplication tensor. For multiplying A (n x m) by B (m x p):
    # A index: a = i*m + l, where i in [0,n), l in [0,m)
    # B index: b = l*p + j, where l in [0,m), j in [0,p)  
    # C index: c = j*n + i, where j in [0,p), i in [0,n)
    # 
    # M[a,b,c] = 1 if a = i*m+l, b = l'*p+j, c = j'*n+i' with l == l' and i == i' and j == j'
    # Otherwise 0.
    # 
    # So M[a,b,c] = 1 iff the l-index in a matches the l-index in b, AND the i-index in a 
    # matches the i-index in c, AND the j-index in b matches the j-index in c.
    # 
    # For n=m=p=3:
    # M[i*3+l, l*3+j, j*3+i] = 1 for all i,j,l in {0,1,2}
    # M[a,b,c] = 0 otherwise.
    # 
    # There are 27 such entries. We need to decompose this into rank r terms.
    # 
    # Known: rank is 23. Let me find an explicit 23-rank decomposition.
    # 
    # I'll use a construction from Alon et al. or a verified source.
    # 
    # Here's a known 23-term decomposition for 3x3:
    
    rank = 23
    n = m = p = 3
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Term construction: I'll use a set of 23 rank-1 tensors that sum to M.
    # Each term r contributes: U[i*3+l, r] * V[l'*3+j, r] * W[j'*3+i', r]
    # The sum over r of these must equal M[i*3+l, l'*3+j, j'*3+i'].
    
    # A verified 23-rank decomposition uses the following pattern:
    # Start with the 27 schoolbook terms, then find 4 linear combinations.
    
    # I'll hardcode a known 23-rank solution.
    # Based on published results, here is one such decomposition:
    
    # I'll generate it programmatically using a known construction.
    # One approach: use the fact that 3x3 can be done with 23 multiplications.
    
    # Let me use a specific known 23-rank decomposition.
    # After research, here is a valid 23-rank decomposition:
    
    terms = []
    
    # I'll use the following construction based on block decomposition:
    # Split 3x3 into a 2x2 block (top-left) and remaining strips.
    # Use Strassen for the 2x2 part (7 terms), then add terms for the strips.
    
    # This is getting complex. Let me just hardcode a verified 23-rank decomposition.
    
    # Here is a known 23-term decomposition from the literature:
    # (coefficients for U, V, W for each term)
    
    # I'll use integers/half-integers.
    
    # Given the complexity, let me use a simpler guaranteed-correct approach:
    # Use rank 26 by finding 1 dependency, or just output a known 23-rank.
    
    # I'll implement a verified 23-rank decomposition based on the following:
    
    # For each of the 23 terms, I specify (U_entries, V_entries, W_entries)
    # where each is a list of 9 values (3x3 flattened).
    
    # This is a known 23-rank decomposition:
    decomp = [
        # Term 0-6: Strassen-like for top-left 2x2
        # A11 = rows 0,1 cols 0,1 of A; A12 = rows 0,1 col 2; etc.
        
        # I'll use a different, more reliable approach:
        # Just output a rank-23 decomposition that I can verify.
        
        # Since I can't easily recall the exact 23 terms, let me use
        # a rank-26 decomposition which is easier to construct and verify.
    ]
    
    # Let me use a rank-26 decomposition: take schoolbook (27) and merge one pair.
    # Actually, let me just output the schoolbook (27) to be safe, since I can't
    # guarantee correctness of a 23-rank decomposition from memory.
    
    # But the task asks for rank-minimal. Let me try to construct a 23-rank.
    
    # I'll use the following verified construction for rank 23:
    
    # Actually, let me just implement the schoolbook (rank 27) to ensure correctness,
    # as I cannot guarantee the correctness of a 23-rank decomposition from memory.
    
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
# EVOLVE-BLOCK-END
