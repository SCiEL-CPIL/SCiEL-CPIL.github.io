"""Matrix multiplication 3x4x4 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # For 3x4x4 matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We use a rank-32 decomposition based on the following strategy:
    # Split the m=4 dimension into two groups of 2: {0,1} and {2,3}
    # For each group, use a 2x2 Strassen-like decomposition.
    #
    # For group {0,1}:
    #   C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   Using 4 products:
    #     P1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    #     P2 = (A[i,0]-A[i,1]) * (B[0,k]-B[1,k])
    #     But this still depends on k...
    #
    # Actually, for a 2-element dot product: a0*b0 + a1*b1
    # This has tensor rank 4 (schoolbook) since it's just 2 terms.
    # There's no Strassen trick for 2x2x2 that helps when one dimension is 1.
    #
    # Let me reconsider. The tensor for 1x4x4 has rank 4 (just the 4 products).
    # So 3x4x4 should have rank at most 3*4 = 12? No, that's not right.
    #
    # The 3x4x4 matrix multiplication tensor: T[i,j,k] = delta(i,k) in some sense
    # Actually T[i,j,k] = 1 if we multiply A[i,j] with B[j,k] to get C[i,k]
    # The tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    #
    # Rank lower bound: max(nm, mp, pn) = max(12,16,12) = 16
    # So rank >= 16.
    #
    # Can we achieve rank 16? That would be optimal.
    # For square n=m=p, rank = n^2 is achievable by schoolbook.
    # For 3x4x4, schoolbook gives 48.
    #
    # Let me try a different approach: use the fact that n=3 is small.
    # C is 3x4. A is 3x4. B is 4x4.
    #
    # Alternative: Think of it as computing 3 rows of C, each being a
    # vector (length 4) times a 4x4 matrix.
    #
    # For a single row: y = x * B where x is 1x4, B is 4x4, y is 1x4.
    # This is a linear map from (x, B) to y. The tensor is in R^4 (x) x R^16 (B) x R^4 (y)
    # Rank of this tensor: it's the 4x4 matrix multiplication "vector-matrix" tensor.
    #
    # Actually, y_k = sum_j x_j B[j,k]. This is just 4 independent products per k... no.
    # For fixed k: y_k = sum_j x_j B[j,k]. This is a bilinear form in x and the k-th column of B.
    # The tensor for one k is in R^4 (x) x R^4 (column k of B) x R^1 (y_k), which has rank 4.
    # For all 4 k's: rank <= 4*4 = 16 for one row of C.
    #
    # For 3 rows: rank <= 3*16 = 48. Still schoolbook.
    #
    # But we can share across rows! The B matrix is shared.
    #
    # Key insight: Let's define linear combinations of the rows of A and columns of B.
    #
    # For rank 16 construction (if possible):
    # We need 16 rank-1 terms. Each term has:
    #   U[i*m+j] = u_{i,j} (coefficient for A[i,j])
    #   V[j*p+k] = v_{j,k} (coefficient for B[j,k])
    #   W[k*n+i] = w_{k,i} (coefficient for C[i,k])
    #
    # And we need: sum_c u_{i,j}^{(c)} * v_{j,k}^{(c)} * w_{k,i}^{(c)} = 1 for all i,j,k
    # (i.e., the decomposition exactly reproduces matrix multiplication)
    #
    # This is equivalent to finding a rank-16 decomposition of the 3x4x4 matrix mult tensor.
    #
    # Known results: The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
    # For 4x4x4, the best known upper bound is 49 (Landsberg) or lower.
    #
    # For 3x4x4, let me try to construct rank 24 or rank 32.
    #
    # Here's a concrete approach for rank 32:
    # Split m=4 into two halves: j in {0,1} and j in {2,3}.
    #
    # Part 1: C[i,k] += sum_{j=0,1} A[i,j] * B[j,k]  (2x4x4 subproblem)
    # Part 2: C[i,k] += sum_{j=2,3} A[i,j] * B[j,k]  (2x4x4 subproblem)
    #
    # For each 2x4x4 part:
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]  (renumbering j=0,1)
    #
    # Can we do 2x4x4 in rank 16?
    #   For each i, k: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is 3*4 = 12 terms in schoolbook for this part.
    #   
    #   Using Strassen-like: for each i, define:
    #     P1_i = (A[i,0] + A[i,1])
    #     P2_i = (A[i,0] - A[i,1])
    #   
    #   For each k, define:
    #     Q1_k = (B[0,k] + B[1,k])
    #     Q2_k = (B[0,k] - B[1,k])
    #   
    #   Then: A[i,0]*B[0,k] + A[i,1]*B[1,k] = (P1_i * Q1_k + P2_i * Q2_k) / 2
    #   
    #   So for each (i,k): 2 products. Total: 3*4*2 = 24 products for this part.
    #   That's worse than 12!
    #
    # The issue is that Q1_k and Q2_k depend on k, so we get 2 products per (i,k).
    #
    # Let me try yet another approach. 
    #
    # Direct construction for rank 24:
    # For each i (3 values), use 8 products.
    # The 8 products for row i compute: C[i,:] = A[i,:] * B
    #
    # A[i,:] = (a0, a1, a2, a3), B is 4x4.
    # C[i,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] + a3*B[3,k]
    #
    # Define 8 linear combinations of a:
    #   s0 = a0 + a1
    #   s1 = a0 - a1
    #   s2 = a2 + a3
    #   s3 = a2 - a3
    #   s4 = a0 + a2
    #   s5 = a0 - a2
    #   s6 = a1 + a3
    #   s7 = a1 - a3
    #
    # And for each k, define corresponding combinations of B[:,k]:
    #   t0_k = B[0,k] + B[1,k]
    #   t1_k = B[0,k] - B[1,k]
    #   t2_k = B[2,k] + B[3,k]
    #   t3_k = B[2,k] - B[3,k]
    #   t4_k = B[0,k] + B[2,k]
    #   t5_k = B[0,k] - B[2,k]
    #   t6_k = B[1,k] + B[3,k]
    #   t7_k = B[1,k] - B[3,k]
    #
    # Now: s0*t0_k + s1*t1_k = (a0+a1)(B0k+B1k) + (a0-a1)(B0k-B1k) = 2*(a0*B0k + a1*B1k)
    # So: a0*B0k + a1*B1k = (s0*t0_k + s1*t1_k)/2
    # Similarly: a2*B2k + a3*B3k = (s2*t2_k + s3*t3_k)/2
    #
    # C[i,k] = (s0*t0_k + s1*t1_k + s2*t2_k + s3*t3_k)/2
    #
    # This uses 4 products per (i,k), total 3*4*4 = 48. No savings again!
    #
    # The fundamental issue: any decomposition where the B-side factors depend on k
    # will scale with p. We need to find rank-1 terms where the B-side factor is
    # a single value (not depending on k), but that's impossible for matrix multiplication.
    #
    # Unless... we use a different structure.
    #
    # Let me think about this differently. In a rank-1 term:
    #   u_{i,j} * v_{j,k} * w_{k,i}
    # 
    # v_{j,k} can depend on both j and k. So a single v factor can encode
    # a linear combination across both j and k dimensions.
    #
    # For rank 16 (the lower bound), we'd need:
    #   sum_{c=1}^{16} u^{(c)}_{i,j} * v^{(c)}_{j,k} * w^{(c)}_{k,i} = delta_{multiplication}
    #
    # This is a very constrained system. Let me just provide a correct
    # rank-48 decomposition for now, since I cannot verify a sub-48 construction
    # in this context.
    
    # Schoolbook rank 48
    rank = n * m * p  # 3*4*4 = 48
    
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 48
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 48
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 48, W[k*n + i]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
