"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # We'll try to build a Strassen-based decomposition with rank 42
    # 6 blocks of 2x2x2, each using Strassen (rank 7) -> total rank 42
    
    # For multiplying a 2x4 matrix A by a 4x6 matrix B to get 2x6 C:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We partition the j dimension (4) into two 2-blocks, and k dimension (6) into three 2-blocks
    # This gives 1 * 3 = 3 blocks of size 2x2x2? No wait.
    # Actually: i has 2 values, j has 4, k has 6.
    # Partition j into 2 groups of 2, k into 3 groups of 2.
    # Block (J, K) where J in {0,1}, K in {0,1,2}:
    #   A block: rows 0..1, cols 2*J..2*J+1 (2x2)
    #   B block: rows 2*J..2*J+1, cols 2*K..2*K+1 (2x2)
    #   C block: rows 0..1, cols 2*K..2*K+1 (2x2)
    #
    # So we have 2*3 = 6 blocks, each a 2x2x2 tensor.
    # Apply Strassen to each block: 7 terms per block -> 42 total rank.
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    #
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b21 - b22)
    # M4 = a22 * (b12 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor T[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor)
    # We need to find U, V, W such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For each block (J, K) with J in {0,1}, K in {0,1,2}:
    #   Local indices: i_loc in {0,1}, j_loc in {0,1}, k_loc in {0,1}
    #   Global: i = i_loc, j = 2*J + j_loc, k = 2*K + k_loc
    #
    # Strassen decomposition for block (J,K):
    # For each of the 7 Strassen terms, we have a rank-1 tensor.
    # The local A entries are A[i_loc, j_loc], local B entries are B[j_loc, k_loc], local C entries are C[i_loc, k_loc].
    
    # Strassen terms as (U_coeff, V_coeff, W_coeff) for each local (i_loc, j_loc, k_loc):
    # We express C[i_loc, k_loc] = sum_t U_t[i_loc] * V_t[j_loc] * W_t[k_loc]
    # where t is the Strassen term index.
    
    # For term 1: M1 = (a11+a22)(b11+b22)
    #   Contribution to C: C11 += M1, C22 += M1
    #   So: C[0,0] += a11*b11 + a11*b22 + a22*b11 + a22*b22
    #         C[1,1] += a11*b11 + a11*b22 + a22*b11 + a22*b22
    #   This is not a rank-1 tensor in general... wait, Strassen works by precomputing products.
    #
    # Actually, in the tensor decomposition context, each Strassen term corresponds to a rank-1 tensor:
    # M1 = (a11+a22)(b11+b22) = sum_{i_loc,j_loc,k_loc} u1[i_loc] * v1[j_loc] * w1[k_loc] * a[i_loc,j_loc] * b[j_loc,k_loc]
    # 
    # Let me think more carefully. The matrix multiplication tensor T satisfies:
    # T[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k].
    #
    # A rank-r decomposition: T[i,j,k] = sum_{c=1}^r U_c[i,j] * V_c[j,k] * W_c[k,i]
    # where U_c, V_c, W_c are vectors/arrays.
    #
    # For Strassen, each term M_t corresponds to a specific combination.
    # Let me use the standard Strassen tensor decomposition.
    
    # For 2x2x2, Strassen gives rank 7. The 7 rank-1 tensors are:
    # t1: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1  -> contributes to C[0,0] and C[1,1]
    #     Actually W is indexed by (k,i), so W[0,0] and W[1,1].
    
    # Let me look up the exact Strassen tensor decomposition.
    # The 7 terms are:
    # M1 = (e1+e2)(f1+f2) where e_i are A rows... no.
    
    # Standard Strassen:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M_t, it's a product of a linear form in A and a linear form in B.
    # M1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    #   As a rank-1 tensor: sum over i,j,k of U1[i,j]*V1[j,k]*W1[k,i]
    #   We need: U1[i,j]*V1[j,k]*W1[k,i] to pick out the right terms.
    #   M1 contributes to C[0,0] with coefficient 1 and to C[1,1] with coefficient 1.
    #   C[0,0] gets: A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[1,0]*B[0,0] + A[1,0]*B[1,0]... 
    #   Wait, M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]). But in our indexing, A[i,j] and B[j,k].
    #   M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    #   This product expands to: A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    #   For this to contribute to C[i,k] = sum_j A[i,j]B[j,k], we need j to be consistent.
    #   A[0,0]B[0,0]: i=0, j=0, k=0 -> C[0,0]
    #   A[0,0]B[1,1]: i=0, j=0, k=1? No, B[j,k] with j=0, k=1 would be B[0,1], but we have B[1,1].
    #   Hmm, this doesn't work directly because the middle index must match.
    
    # I think I need to be more careful. Let me reconsider.
    # In the 2x2x2 tensor, the indices are (i, j, k) with i for A's row, j for A's col / B's row, k for B's col.
    # A[i,j], B[j,k], C[i,k].
    #
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # The terms are: A[0,0]B[0,0], A[0,0]B[1,1], A[1,1]B[0,0], A[1,1]B[1,1]
    # A[0,0]B[0,0]: i=0, j=0, k=0
    # A[0,0]B[1,1]: i=0, j=0, k=1 -- but B[1,1] has j=1, not j=0! Contradiction.
    
    # I see my error. In Strassen, the matrices are 2x2, and the middle index of B is the column of A.
    # Let me re-label: A has entries A[i,j] for i,j in {0,1}, B has entries B[j,k] for j,k in {0,1}.
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # For the product to be well-defined in the tensor sense, we need the j-index to be consistent.
    # A[0,0] * B[0,0]: j=0, OK.
    # A[0,0] * B[1,1]: j=0 for A, j=1 for B. MISMATCH!
    
    # So Strassen's M1 as written doesn't directly correspond to a rank-1 tensor in the (i,j,k) space.
    # The issue is that Strassen precomputes sums of A entries and sums of B entries, and the product
    # of these sums involves cross terms where the middle index doesn't match.
    
    # Actually, I think the standard way to express Strassen as a tensor decomposition is different.
    # Let me look at this more carefully.
    #
    # The matrix multiplication tensor for 2x2x2 has 8 entries (all 1s).
    # A rank-7 decomposition exists (Strassen). The 7 rank-1 tensors are:
    # 
    # Looking at the literature, the Strassen decomposition for 2x2x2 is:
    # T = sum_{t=1}^7 T_t where each T_t is rank-1.
    #
    # Let me use the explicit formula from the literature.
    # Define the 7 tensors as follows (using 0-indexed (i,j,k)):
    #
    # t=0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
    #   This gives: A[0,0]B[0,0]W[0,0] + A[0,0]B[1,1]W[1,0]? No...
    
    # I think I'm overcomplicating this. Let me just directly construct the rank-42 decomposition
    # by applying Strassen to each 2x2x2 block, using the known Strassen tensor decomposition.
    
    # From "Fast Matrix Multiplication" and related references, the Strassen decomposition
    # for 2x2x2 can be written as 7 rank-1 tensors. Let me use the following explicit form:
    #
    # For the 2x2x2 tensor T[i,j,k] = 1 for i,j,k in {0,1}:
    #
    # Rank-1 term 0: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    #   Contribution: sum_{i,j,k} U[i,j]V[j,k]W[k,i]
    #   = U[0,0]V[0,0]W[0,0] + U[0,0]V[0,1]W[1,0] + U[0,0]V[1,0]W[0,1] + ...
    #   This is getting complicated. Let me just try a direct approach.
    
    # Alternative: Let me try to find a known rank-42 decomposition for 2x4x6.
    # Or, let me try to use the fact that rank(2x4x6) <= rank(2x2x2)^3 / some factor? No.
    # Actually, rank(2x4x6) <= 6 * rank(2x2x2) = 6*7 = 42 if the blocks are independent.
    # And since the supports are disjoint, we can just sum the Strassen decompositions.
    
    # Let me try to implement this by explicitly writing out the Strassen terms.
    # For a 2x2x2 block with local indices (i,j,k) in {0,1}^3:
    # The Strassen decomposition has 7 terms. Let me find them explicitly.
    
    # After some research, the 7 rank-1 tensors for 2x2x2 (Strassen) are:
    # (Using 1-indexed for clarity, then convert to 0-indexed)
    #
    # Let me just try to construct it numerically.
    # Actually, let me try a simpler approach: use the known fact that the 2x2x2 tensor
    # has a rank-7 decomposition and try to find it by solving the system.
    
    # But that might be too slow. Let me just try the rank-48 schoolbook and see if I can
    # find a better decomposition within the time budget.
    
    # Actually, let me try to implement the Strassen-based decomposition directly.
    # Here's a known Strassen decomposition for 2x2x2:
    #
    # Term 1: U: A[0,0]=1, A[1,1]=1; V: B[0,0]=1, B[1,1]=1; W: C[0,0]=1, C[1,1]=1
    #   This corresponds to M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    #   But as noted, this has cross terms that don't match the middle index.
    
    # I think the correct way is:
    # M1 contributes to C[0,0] and C[1,1] with specific coefficients.
    # In the tensor decomposition, each term is a rank-1 tensor, meaning:
    # T_t[i,j,k] = u_t[i,j] * v_t[j,k] * w_t[k,i]
    #
    # For Strassen term M1:
    # The A-part is (A[0,0]+A[1,1]) = A[0,0] + A[1,1]
    # The B-part is (B[0,0]+B[1,1]) = B[0,0] + B[1,1]
    # The product is: A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # For this to be a valid contribution to C[i,k] = sum_j A[i,j]B[j,k]:
    # A[0,0]B[0,0]: i=0, j=0, k=0 -> C[0,0]
    # A[0,0]B[1,1]: i=0, j=0 for A, j=1 for B -> INVALID (j mismatch)
    #
    # So M1 as a product of sums doesn't directly give a rank-1 tensor.
    # The resolution is that Strassen's algorithm works by reorganizing the computation,
    # and the tensor decomposition is different from what I was thinking.
    
    # Let me try a different approach: I'll use the explicit Strassen tensor decomposition
    # from the literature. Here's one form:
    #
    # The 7 rank-1 tensors for 2x2x2 are given by the following (i,j,k) triples with coefficients:
    # Actually, each rank-1 tensor is a tensor T_t where T_t[i,j,k] = 1 for specific (i,j,k)
    # and 0 otherwise, but that would give rank 8 (the schoolbook).
    
    # For Strassen, the rank-1 tensors have multiple non-zero entries.
    # Let me look at this from the perspective of the factor matrices.
    
    # For 2x2x2, U is 4x7, V is 4x7, W is 4x7.
    # Let me try to find U, V, W such that T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    
    # I'll try to use a known solution. Here's one from the literature:
    # 
    # Actually, let me just try to verify if rank 42 is achievable and construct it.
    # Given the time pressure, let me fall back to the schoolbook rank-48 if I can't
    # quickly construct the Strassen-based decomposition.
    
    # Let me try one more approach: directly use the Strassen algorithm structure.
    # For the 2x4x6 multiplication, I'll partition into 6 blocks of 2x2x2.
    # For each block, I'll use 7 Strassen terms.
    
    # Here's my plan for the Strassen terms for a single 2x2x2 block:
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global: I = block_row_offset + i, J = block_col_offset_A + j, K = block_col_offset_B + k
    # Wait, the structure is: A is 2x4, B is 4x6, C is 2x6.
    # Block (J, K) where J in {0,1} (for A's columns / B's rows) and K in {0,1,2} (for B's columns / C's columns):
    #   A block: A[0:2, 2*J:2*J+2] (2x2)
    #   B block: B[2*J:2*J+2, 2*K:2*K+2] (2x2)
    #   C block: C[0:2, 2*K:2*K+2] (2x2)
    
    # For Strassen on this block, the 7 terms are:
    # Let a = A_block, b = B_block (both 2x2)
    # M1 = (a[0,0]+a[1,1]) * (b[0,0]+b[1,1])
    # M2 = (a[1,0]+a[1,1]) * b[0,0]
    # M3 = a[0,0] * (b[1,0]-b[1,1])
    # M4 = a[1,1] * (b[0,1]-b[0,0])
    # M5 = (a[0,0]+a[0,1]) * b[1,1]
    # M6 = (a[1,0]-a[0,0]) * (b[0,0]+b[0,1])
    # M7 = (a[0,1]-a[1,1]) * (b[1,0]+b[1,1])
    #
    # C_block[0,0] = M1 + M4 - M5 + M7
    # C_block[0,1] = M3 + M5
    # C_block[1,0] = M2 + M4
    # C_block[1,1] = M1 - M2 + M3 + M6
    
    # Each M_t is a product of a linear combination of A entries and a linear combination of B entries.
    # In the tensor decomposition, each M_t corresponds to a rank-1 tensor.
    # The rank-1 tensor for M_t has:
    #   U_part[i,j] = coefficient of A[i,j] in the A-linear-combination
    #   V_part[j,k] = coefficient of B[j,k] in the B-linear-combination
    #   W_part[k,i] = coefficient of C[i,k] in the output (but this is implicit)
    
    # Wait, I think the correct interpretation is:
    # M_t = (sum_{i,j} alpha_t[i,j] * A[i,j]) * (sum_{j,k} beta_t[j,k] * B[j,k])
    # This is NOT a rank-1 tensor in the (i,j,k) space because the sums are over different indices.
    #
    # The key insight is that in Strassen's algorithm, the products M_t are computed once and then
    # used in linear combinations to get C. The tensor decomposition captures this by having
    # each M_t correspond to a rank-1 tensor where the "middle" index j is summed over.
    
    # I think I need to accept that I can't easily derive the Strassen tensor decomposition from scratch
    # in this time. Let me just return the schoolbook rank-48 decomposition.
    
    rank = n * m * p  # 48
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
