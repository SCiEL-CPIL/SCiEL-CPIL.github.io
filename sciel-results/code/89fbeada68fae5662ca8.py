"""Matrix multiplication 3,3,5 - rank-minimal decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We'll build a decomposition with rank = 23 + 6 = 29
    # First 23 terms: rank-23 decomposition for 3x3 part (columns 0-2 of B)
    # Last 6 terms: schoolbook for 3x2 part (columns 3-4 of B)
    
    # Known rank-23 decomposition for 3x3 matrix multiplication
    # This uses the Baur-Strassen / Landsberg construction
    # U, V, W for the 3x3x3 part (each 9x23)
    
    # The 23 multiplications for 3x3 matrix multiplication:
    # Using the standard explicit rank-23 scheme
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a22-a11)(b11-b22)
    # M3 = (a11+a12)(b21+b22)
    # M4 = (a12-a11)(b21-b22)
    # M5 = (a11+a13)(b31+b32)
    # M6 = (a13-a11)(b31-b32)
    # M7 = (a22+a23)(b31+b33)
    # M8 = (a23-a22)(b31-b33)
    # M9 = (a12+a23)(b12+b23)
    # M10 = (a23-a12)(b12-b23)
    # M11 = (a13+a23)(b13+b21)  -- need to be careful
    # ... This is getting complex. Let me use a different approach.
    
    # Instead, let me use a well-documented rank-23 decomposition.
    # I'll use the construction from the literature.
    
    # Actually, let me use a simpler but still improved approach:
    # Use Strassen-like decomposition for 3x3 giving rank 22 or 23.
    
    # Let me use the explicit rank-23 decomposition coefficients.
    # Based on the standard 23-multiplication algorithm for 3x3:
    
    # The 23 rank-1 terms for 3x3x3:
    # Each term has (i,j,k) indices and coefficients for U[i,j], V[j,k], W[k,i]
    
    # I'll construct this carefully using the known scheme.
    
    # For now, let me use a reliable construction:
    # Combine rank-23 for 3x3 with rank-6 schoolbook for 3x2
    
    # Rank-23 3x3x3 decomposition (explicit):
    # Using the algorithm that achieves R(3,3,3) = 23
    
    # Coefficients for U (left matrix factors), V (right matrix factors), W (output factors)
    # Each of 23 terms
    
    # I'll use a verified rank-23 scheme. Here's one based on the standard construction:
    
    # Term structure: for each of 23 terms, we have:
    # U term: linear combination of a_ij (i in 0..2, j in 0..2)
    # V term: linear combination of b_jk (j in 0..2, k in 0..2) 
    # W term: linear combination contributing to c_ik
    
    # Let me use the following verified rank-23 decomposition:
    # This is the standard 23-multiplication algorithm
    
    # After careful consideration, let me use a direct construction.
    # The key insight: we can write the tensor as sum of 23 rank-1 tensors.
    
    # Here is an explicit rank-23 decomposition for 3x3x3:
    # (i,j,k) where i=row of C, j=row of B / col of A, k=col of C
    
    # I'll use numpy to construct and verify
    
    # For the 3x3 part (B columns 0,1,2), we use rank 23
    # For the 3x2 part (B columns 3,4), we use rank 6 (schoolbook)
    
    rank_3x3 = 23
    rank_3x2 = 6
    total_rank = rank_3x3 + rank_3x2  # 29
    
    U = np.zeros((9, total_rank))
    V = np.zeros((15, total_rank))
    W = np.zeros((15, total_rank))
    
    # Schoolbook part for 3x2 (columns 3,4 of B): 6 terms
    # These are terms where k=3 or k=4
    for i in range(3):
        for j in range(3):
            for k_offset, k in enumerate([3, 4]):
                col = rank_3x3 + i*3 + j*2 + k_offset
                # U[i*m+j] = U[i*3+j]
                U[i*3+j, col] = 1
                # V[j*p+k] = V[j*5+k]
                V[j*5+k, col] = 1
                # W[k*n+i] = W[k*3+i]
                W[k*3+i, col] = 1
    
    # Now for the 3x3 part (columns 0,1,2 of B): rank 23
    # I need to fill columns 0-22 of U, V, W
    
    # Using the known rank-23 decomposition for 3x3 matrix multiplication
    # This is based on the explicit scheme from the literature
    
    # The 23 multiplications and their contributions:
    # I'll use a parameterization that's been verified
    
    # After research, here's a valid rank-23 scheme for 3x3:
    # We express C = AB where A is 3x3, B is 3x3
    
    # The 23 rank-1 decompositions:
    # Each term t has: U_t (3x3 matrix for A's linear combo), V_t (3x3 for B's linear combo), W_t (3x3 for C's contribution)
    
    # Let me construct this using the standard 23-multiplication algorithm:
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a11-a00)(b00-b11)
    # M3 = (a00+a01)(b10+b11)
    # M4 = (a01-a00)(b10-b11)
    # M5 = (a00+a02)(b20+b21)
    # M6 = (a02-a00)(b20-b21)
    # M7 = (a11+a12)(b20+b22)
    # M8 = (a12-a11)(b20-b22)
    # M9 = (a01+a12)(b01+b12)
    # M10 = (a12-a01)(b01-b12)
    # M11 = (a02+a12)(b02+b10)
    # M12 = (a12-a02)(b02-b10)
    # M13 = (a00+a11+a02+a12)(b00+b11+b02+b12) - this is getting complicated
    
    # Let me use a different, cleaner approach. I'll use the fact that we can
    # verify the decomposition numerically and construct it from known coefficients.
    
    # Here's a verified rank-23 decomposition using explicit coefficients:
    
    # I'll define the 23 terms directly. Each term (u, v, w) where:
    # u is a 9-vector (coefficients for a_00..a_22)
    # v is a 9-vector (coefficients for b_00..b_22)  
    # w is a 9-vector (coefficients for c_00..c_22)
    
    # The constraint is: for all A, B: sum_t (u_t . A) * (v_t . B) = (w_t . C)
    
    # Here's a known valid set of 23 terms:
    
    terms = []
    
    # Term 1: (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
    terms.append(([1,0,0, 0,1,0, 0,0,0],
                 [1,0,0, 0,1,0, 0,0,0],
                 [1,0,0, 0,1,0, 0,0,0]))
    
    # Term 2: (a11-a00)(b00-b11)
    terms.append(([0,-1,0, 1,0,0, 0,0,0],
                 [1,0,0, 0,-1,0, 0,0,0],
                 [0,0,0, 0,0,0, 0,0,0]))  # This doesn't work directly
    
    # This manual construction is error-prone. Let me use a different strategy.
    
    # Strategy: Use the schoolbook for now but try to find a better one via search.
    # Given time constraints, I'll return a valid decomposition.
    
    # Let me just use schoolbook with rank 45 to guarantee correctness,
    # but I know we can do better.
    
    # Actually, let me use a known result: R(3,3,5) <= 41
    # I'll construct rank 41 using: rank-23 for 3x3x3 + rank-18 for 3x3x2
    
    # For 3x3x2, schoolbook gives rank 6, but we can do rank 18? No, schoolbook is 3*3*2=18.
    # Wait, 3*3*2 = 18 for schoolbook of 3x3 times 3x2.
    
    # So rank-23 + rank-18 = rank-41.
    
    # Let me construct this properly.
    
    # Reset
    rank_3x3 = 23
    rank_3x2 = 18  # schoolbook for 3x3x2
    total_rank = rank_3x3 + rank_3x2
    
    U = np.zeros((9, total_rank))
    V = np.zeros((15, total_rank))
    W = np.zeros((15, total_rank))
    
    # Schoolbook for 3x3x2 part (B columns 3,4): 18 terms
    for i in range(3):
        for j in range(3):
            for k_offset, k in enumerate([3, 4]):
                col = rank_3x3 + (i*3 + j)*2 + k_offset
                U[i*3+j, col] = 1
                V[j*5+k, col] = 1
                W[k*3+i, col] = 1
    
    # For the 3x3x3 part, I need rank 23.
    # Let me use a simple construction: schoolbook gives rank 9 for 3x3x3.
    # I'll use schoolbook for simplicity to guarantee correctness, giving total rank 27.
    # This is better than 45.
    
    # Actually, let me just do full schoolbook to be safe, then I know it works.
    # Schoolbook for 3x3x5: rank 45.
    
    # But I can do better with a partial optimization.
    # Let me use schoolbook for the 3x3x3 part (rank 9) and schoolbook for 3x3x2 (rank 6).
    # Total: rank 15. Wait, that's not right.
    
    # Schoolbook for 3x3x5: for each (i,j,k) in {0..2}x{0..2}x{0..4}, one term.
    # That's 3*3*5 = 45 terms.
    
    # If I partition B into B1 (cols 0-2) and B2 (cols 3-4):
    # C = A*B1 + A*B2
    # A*B1: 3x3 * 3x3, schoolbook rank 9
    # A*B2: 3x3 * 3x2, schoolbook rank 6
    # Total: rank 15? No, that's not how it works. The rank is the number of terms.
    # Schoolbook for 3x3x3 is 9 terms, for 3x3x2 is 6 terms. Total 15 terms.
    # But each term in the 3x3x3 part only uses B columns 0-2, and each term in 
    # 3x3x2 only uses B columns 3-4. So total rank is 9+6=15.
    
    # Wait, that can't be right because the minimum rank is at least 9 (dimension argument).
    # Actually, for 3x3x5, the minimum rank is at least max(9, 15, 15) = 15? No.
    # The lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15.
    # And schoolbook gives 45. So there's room for improvement.
    
    # But my partition approach with schoolbook for each part gives rank 9+6=15.
    # Is that valid? Let me check.
    
    # For A*B1 where B1 is 3x3: schoolbook uses 9 terms, each (i,j,k) for k in 0..2
    # For A*B2 where B2 is 3x2: schoolbook uses 6 terms, each (i,j,k) for k in 3..4
    # Total: 15 terms. Each term is rank-1 in the tensor sense.
    # The sum gives the full product. So yes, rank 15 is achievable!
    
    # But wait, that contradicts known results. Let me reconsider.
    # Oh, I see the issue. The rank of the tensor is not simply additive in this way
    # because the "rank" refers to the number of rank-1 tensors in the decomposition.
    # Schoolbook for 3x3x3 is indeed 9 rank-1 tensors.
    # Schoolbook for 3x3x2 is indeed 6 rank-1 tensors.
    # Combined, they form 15 rank-1 tensors for 3x3x5.
    # So R(3,3,5) <= 15.
    
    # But the lower bound is max(n*m, m*p, p*n) = max(9,15,15) = 15.
    # So R(3,3,5) = 15? That seems too low.
    
    # Let me reconsider. The tensor for multiplying 3x3 by 3x5 has:
    # A: 9 elements, B: 15 elements, C: 15 elements.
    # The tensor is in R^9 ⊗ R^15 ⊗ R^15.
    # The rank of this tensor...
    
    # Actually, I think I'm confusing myself. Let me just construct the schoolbook
    # decomposition which I know is correct, and see if I can reduce it.
    
    # Schoolbook: rank 45
    # For each i in 0..2, j in 0..2, k in 0..4:
    #   U[i*3+j, term] = 1
    #   V[j*5+k, term] = 1  
    #   W[k*3+i, term] = 1
    
    # This is definitely correct. Let me return this and verify.
    
    # But I want to do better. The key insight is that for fixed (i,j), 
    # the sum over k of a_ij * b_jk can be computed with fewer than 5 terms
    # if we use linear combinations.
    
    # For the 3x2 part (k=3,4): for each (i,j), we need a_ij*(b_j3, b_j4).
    # This is a scalar times a 2-vector. We can't reduce this below 2 terms per (i,j)
    # in general, so 3*3*2 = 18 terms for the 3x2 part.
    
    # Wait, no. For each (i,j), the contribution to C is a_ij * [b_j3, b_j4] added to rows i.
    # The output has C[i,3] += a_ij*b_j3 and C[i,4] += a_ij*b_j4.
    # So for each (i,j), we need 2 multiplications: a_ij*b_j3 and a_ij*b_j4.
    # Total: 3*3*2 = 18 terms for the 3x2 part.
    
    # Similarly for 3x3 part: 3*3*3 = 27 terms.
    # Total schoolbook: 27+18 = 45. Correct.
    
    # Now, can we reduce the 3x3 part below 27? Yes, using Strassen-like methods.
    # R(3,3,3) <= 23, so we can use 23 terms for the 3x3 part.
    # And R(3,3,2) <= 18 (schoolbook), so 18 terms for the 3x2 part.
    # Total: 23+18 = 41.
    
    # So I need a rank-23 decomposition for 3x3x3.
    
    # Let me hardcode a known rank-23 decomposition.
    
    # Here's an explicit rank-23 decomposition for 3x3 matrix multiplication:
    # Based on the standard construction.
    
    # I'll define the 23 terms as triples of 9-vectors (u, v, w) where:
    # u: coefficients for A (a00..a22)
    # v: coefficients for B (b00..b22)
    # w: coefficients for C (c00..c22)
    
    # The condition: for all A, B, sum_t (u_t @ A_flat) * (v_t @ B_flat) = w_t @ C_flat
    # where C = AB.
    
    # Here's a verified rank-23 scheme (from literature):
    
    # I'll use the following explicit coefficients:
    
    t23 = [
        # (u, v, w) where each is [a00,a01,a02,a10,a11,a12,a20,a21,a22]
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,-1,0, 0,0,0], [0,-1,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # ... I'm not confident in these coefficients.
    ]
    
    # This is getting too complex without verification. Let me use a safer approach.
    
    # I'll return the schoolbook decomposition (rank 45) which is guaranteed correct.
    # The score will be 45, which is the baseline.
    
    # But wait, I can at least do the simple partition:
    # Use schoolbook for 3x3x3 (27 terms) and schoolbook for 3x3x2 (18 terms).
    # Total: 45 terms. Same as full schoolbook.
    
    # To actually improve, I need a non-schoolbook decomposition for at least one part.
    
    # Let me try a different approach: use the fact that for the 3x2 output part,
    # I can use linear combinations.
    
    # Actually, let me just return schoolbook to ensure correctness.
    
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][column] = 1
                V_list[j * p + k][column] = 1
                W_list[k * n + i][column] = 1
                column += 1
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

