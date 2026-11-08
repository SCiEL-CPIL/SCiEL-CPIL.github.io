"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll use a structured decomposition that exploits the rectangular nature.
    # For 4x4 times 4x7, we can use a decomposition based on splitting the middle dimension.
    
    # Strategy: Use the fact that we can share computations across rows of A and columns of B.
    # 
    # Key insight: For matrix multiplication C = A*B where A is n×m and B is m×p,
    # we can decompose using rank-1 terms. The schoolbook uses n*m*p terms.
    # 
    # For 4×4×7, schoolbook is 112. Let's try to do better.
    #
    # Approach: Use a bilinear decomposition. We can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can group terms. For example, for each pair (i, j), we have m terms in k.
    # But we can also think of it as: for each j, we compute A[:,j] * B[j,:],
    # which is an outer product of an n-vector and a p-vector, giving n*p rank-1 terms per j.
    # Total: m * n * p = 112.
    #
    # To reduce rank, we need to find linear dependencies or use identities.
    #
    # One known technique: For square matrices, Strassen's algorithm reduces 2×2×2 from 8 to 7.
    # For 4×4, we can use recursive Strassen-like decompositions.
    #
    # However, for 4×4×7 (rectangular), let's use a different approach.
    # We can pad B to be 4×8 and use a 4×4×8 decomposition, then drop the last column.
    # But that might not help directly.
    #
    # Let's try a simple optimization: For each i, define S_i = sum_j A[i,j].
    # Then C[i,k] = S_i * (sum_j B[j,k]) - sum_j A[i,j] * (sum_{l≠j} B[l,k])
    # This doesn't obviously reduce rank.
    #
    # Alternative: Use the identity for 2 elements:
    # a*b + c*d = (a+c)*(b+d) - a*d - c*b
    # This reduces 4 terms to 3 terms (net saving of 1).
    #
    # We can apply this to pairs of (j, j') for fixed i and k.
    # For each i, k, we have 4 terms in j. Pair them up: (j=0,1) and (j=2,3).
    # For pair (0,1): A[i,0]*B[0,k] + A[i,1]*B[1,k] = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - A[i,0]*B[1,k] - A[i,1]*B[0,k]
    # This still gives 3 terms per pair, so 6 terms per (i,k) instead of 4. That's worse!
    #
    # Wait, the identity is: a*b + c*d = (a+c)(b+d) - a*d - c*b
    # Original: 2 terms (a*b + c*d)
    # New: 3 terms ((a+c)(b+d) - a*d - c*b)
    # So it increases rank by 1 per pair. Not helpful for reducing rank.
    #
    # The correct Strassen identity for 2×2:
    # [a b; c d] * [e f; g h] = [ae+bg, af+bh; ce+dg, cf+dh]
    # Strassen uses 7 multiplications:
    # m1 = (a+b)(e+f)
    # m2 = (c+d)*e
    # m3 = a*(f-g)
    # m4 = d*(h-g)
    # m5 = (a+d)(e+h)
    # m6 = (b-c)(e+f)
    # m7 = (c-d)(g+h)
    # Then:
    # C11 = m1 + m4 - m5 + m6
    # C12 = m3 + m5
    # C21 = m2 + m5
    # C22 = m1 - m2 + m3 + m4
    #
    # For 4×4, we can use a recursive approach or a direct 4×4 decomposition.
    # Known: 4×4×4 can be done in rank 49 (using Strassen recursively).
    # For 4×4×7, we can compute 4×4×4 and 4×4×3 separately.
    # 4×4×4: rank 49
    # 4×4×3: Let's compute this. Schoolbook is 48. Can we do better?
    # For 4×4×3, we can use a similar approach.
    #
    # Actually, let's just use a known good rank for 4×4×7.
    # The minimum rank for 4×4×4 is 49. For 4×4×7, it should be at most 49 + 48 = 97.
    # But we can do better by sharing.
    #
    # Let's implement a 4×4×4 Strassen decomposition and extend it.
    #
    # For simplicity, let's use a rank-96 decomposition based on a known pattern.
    # 
    # Actually, let's just use a greedy approach to find a good decomposition.
    # We'll start with the schoolbook and try to merge terms using linear algebra.
    
    # For now, let's use a simple rank-100 decomposition.
    # We'll use the fact that we can share some computations.
    
    # Let's use a decomposition based on the following:
    # For each i (row of A), we compute A[i,:] * B, which is a 1×m times m×p matrix.
    # This is a rank-m operation for each i, total rank n*m*p.
    # 
    # To reduce rank, we can use the following trick:
    # Define for each i, a vector u_i of length m (the i-th row of A).
    # Define for each k, a vector v_k of length m (the k-th column of B, transposed).
    # Then C[i,k] = u_i · v_k.
    # This is a bilinear form, and the rank is the tensor rank of the multiplication tensor.
    
    # For now, let's just return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
