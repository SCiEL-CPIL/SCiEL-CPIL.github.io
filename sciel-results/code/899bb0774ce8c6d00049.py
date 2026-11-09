"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Tensor decomposition for 4x4 * 4x7 matrix multiplication
    # T[i,j,k] = 1 means C[i,k] += A[i,j] * B[j,k]
    #
    # Strategy: Use Strassen for the 4x4x4 part (k=0..3) and schoolbook for 4x4x3 part (k=4..6)
    #
    # Block decomposition:
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B_left = [[B11, B12], [B21, B22]] each 2x2 (columns 0-3)
    # B_right = [[B13], [B23]] each 2x3 (columns 4-6)
    #
    # Strassen for 4x4x4: 7 block multiplications, each 2x2x2 -> 4 scalar terms each = 28 terms
    # Schoolbook for 4x4x3: 4*4*3 = 48 terms
    # Total: 76 terms
    
    # Each term: (i, j, k, u_coeff, v_coeff, w_coeff)
    # where the term contributes u_coeff * A[i,j] * v_coeff * B[j,k] * w_coeff to C[i,k]
    # In the factor matrices:
    #   U[i*m + j, r] = u_coeff
    #   V[j*p + k, r] = v_coeff  
    #   W[k*n + i, r] = w_coeff
    
    terms = []
    
    # ===== Part 1: Strassen for 4x4 * 4x4 (k in 0..3) =====
    # 
    # Strassen block multiplications (each is a 2x2 * 2x2 product):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # Results:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each block multiplication M_t = X * Y where X and Y are 2x2 matrices
    # (linear combinations of A and B blocks)
    #
    # For a block multiplication, the scalar expansion is:
    # M_t[di,dk] = sum_{dj=0}^{1} X[di,dj] * Y[dj,dk]
    #
    # X[di,dj] is a linear combination of A entries
    # Y[dj,dk] is a linear combination of B entries
    #
    # The key insight: each block multiplication M_t can be decomposed into
    # 4 scalar rank-1 terms (one for each (di,dj,dk) in the 2x2x2 sub-tensor),
    # but the coefficients in U, V, W depend on the specific Strassen formula.
    
    # Let's define the Strassen terms explicitly.
    # For each block multiplication, we need to express it as sum of rank-1 terms.
    #
    # M_t = X * Y where X is 2x2, Y is 2x2
    # M_t[di,dk] = sum_dj X[di,dj] * Y[dj,dk]
    #
    # This is a 2x2x2 tensor. Its rank-1 decomposition (schoolbook) has 4 terms:
    # Term (di,dj,dk): X[di,dj] * Y[dj,dk] -> M_t[di,dk]
    #
    # But X and Y are themselves linear combinations. So we need to expand.
    #
    # Let me define each Strassen multiplication explicitly.
    
    # Helper: add a term to the decomposition
    # The term represents: coeff_u * A[i,j] * coeff_v * B[j,k] -> coeff_w * C[i,k]
    # In the factorization: U[i*m+j, r] = coeff_u, V[j*p+k, r] = coeff_v, W[k*n+i, r] = coeff_w
    
    def add_term(i, j, k, cu, cv, cw):
        terms.append((i, j, k, cu, cv, cw))
    
    # M1 = (A11+A22)*(B11+B22)
    # X[di,dj] = A[di,dj] + A[di+2,dj+2]
    # Y[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    # M1[di,dk] = sum_dj (A[di,dj]+A[di+2,dj+2])*(B[dj,dk]+B[dj+2,dk+2])
    #
    # Expand: = sum_dj [A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2]]
    #
    # M1 contributes to C11 (i=di, k=dk, di,dk in 0..1) with coeff +1
    # and to C22 (i=di+2, k=dk+2) with coeff +1
    #
    # For each (di,dj,dk) in 0..1:
    #   A[di,dj]*B[dj,dk] -> C[di,dk] with coeff +1
    #   A[di,dj]*B[dj+2,dk+2] -> C[di,dk] with coeff +1
    #   A[di+2,dj+2]*B[dj,dk] -> C[di,dk] with coeff +1
    #   A[di+2,dj+2]*B[dj+2,dk+2] -> C[di,dk] with coeff +1
    #   A[di,dj]*B[dj,dk] -> C[di+2,dk+2] with coeff +1
    #   A[di,dj]*B[dj+2,dk+2] -> C[di+2,dk+2] with coeff +1
    #   A[di+2,dj+2]*B[dj,dk] -> C[di+2,dk+2] with coeff +1
    #   A[di+2,dj+2]*B[dj+2,dk+2] -> C[di+2,dk+2] with coeff +1
    
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # A[di,dj]*B[dj,dk]
                add_term(di, dj, dk, 1, 1, 1)       # -> C[di,dk]
                add_term(di, dj, dk, 1, 1, 1)       # -> C[di+2,dk+2]
                # A[di,dj]*B[dj+2,dk+2]
                add_term(di, dj, dk+2, 1, 1, 1)     # -> C[di,dk]
                add_term(di, dj, dk+2, 1, 1, 1)     # -> C[di+2,dk+2]
                # A[di+2,dj+2]*B[dj,dk]
                add_term(di+2, dj+2, dk, 1, 1, 1)   # -> C[di,dk]
                add_term(di+2, dj+2, dk, 1, 1, 1)   # -> C[di+2,dk+2]
                # A[di+2,dj+2]*B[dj+2,dk+2]
                add_term(di+2, dj+2, dk+2, 1, 1, 1) # -> C[di,dk]
                add_term(di+2, dj+2, dk+2, 1, 1, 1) # -> C[di+2,dk+2]
    
    # Wait, this is wrong. The issue is that each rank-1 term in the tensor decomposition
    # is of the form: u_i * v_j * w_k
    # where u, v, w are vectors of length n*m, m*p, p*n respectively.
    #
    # A single rank-1 term can only involve ONE (i,j,k) triple with specific coefficients.
    # The expansion above is incorrect because I'm trying to make one term contribute to
    # multiple C entries.
    #
    # Let me restart with a correct approach.
    #
    # In the tensor decomposition, each rank-1 term r has:
    #   U[:, r] = vector of length n*m
    #   V[:, r] = vector of length m*p
    #   W[:, r] = vector of length p*n
    #
    # The decomposition satisfies:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
    #
    # A rank-1 term with U[i*m+j, r] = a, V[j*p+k, r] = b, W[k*n+i, r] = c
    # contributes a*b*c to T[i,j,k].
    #
    # So each term is associated with a specific (i,j,k) triple.
    #
    # Now, Strassen's algorithm works by replacing 8 schoolbook terms with 7 "virtual" terms.
    # But at the scalar level, we still need to express the result as a sum of rank-1 terms.
    #
    # The correct way: Strassen reduces the number of independent multiplications from 8 to 7
    # at the block level. At the scalar level, each block multiplication (2x2*2x2) still
    # requires 4 scalar multiplications. So 7*4 = 28 scalar terms for the 4x4x4 part.
    #
    # But the 28 terms are NOT the same as 28 of the 64 schoolbook terms. They are
    # different combinations.
    #
    # Let me think about this more carefully using the Strassen algorithm.
    #
    # Actually, I think the correct approach is:
    # 1. The 4x4x4 tensor can be decomposed into 28 rank-1 terms using Strassen.
    # 2. Each of these 28 terms corresponds to a specific (i,j,k) triple with coefficients.
    #
    # Let me derive the 28 terms explicitly.
    
    # CLEAR the terms list and restart
    terms = []
    
    # For the 4x4x4 part, use Strassen.
    # The Strassen algorithm computes C = A*B where A, B, C are 4x4.
    # 
    # Define blocks:
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4]
    # A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4]
    # B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    #
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M_t is a 2x2 matrix. M_t[di,dk] = sum_{dj} X_t[di,dj] * Y_t[dj,dk]
    #
    # The key: each M_t is computed as a 2x2*2x2 product, which in tensor terms
    # is a 2x2x2 tensor. The rank-1 decomposition of a 2x2x2 multiplication tensor
    # (schoolbook) has 4 terms.
    #
    # So for each M_t, we get 4 scalar rank-1 terms.
    # Total: 7 * 4 = 28 terms for the 4x4x4 part.
    #
    # Now, let's enumerate these 28 terms.
    #
    # For M1 = (A11+A22)*(B11+B22):
    # X1[di,dj] = A[di,dj] + A[di+2,dj+2]
    # Y1[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    #
    # M1[di,dk] = sum_{dj} (A[di,dj]+A[di+2,dj+2])*(B[dj,dk]+B[dj+2,dk+2])
    #
    # This expands to 4 terms for each (di,dk), but the (dj) sum means:
    # M1[di,dk] = (A[di,0]+A[di+2,2])*(B[0,dk]+B[2,dk+2]) + (A[di,1]+A[di+2,3])*(B[1,dk]+B[3,dk+2])
    #
    # Each product is a scalar multiplication. So M1 has 4*2 = 8 scalar multiplications?
    # No wait, that's not right either.
    #
    # Let me be very explicit. M1 is a 2x2 matrix where:
    # M1[0,0] = (A[0,0]+A[2,2])*(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])*(B[1,0]+B[3,2])
    # M1[0,1] = (A[0,0]+A[2,2])*(B[0,1]+B[2,3]) + (A[0,1]+A[2,3])*(B[1,1]+B[3,3])
    # M1[1,0] = (A[1,0]+A[3,2])*(B[0,0]+B[2,2]) + (A[1,1]+A[3,3])*(B[1,0]+B[3,2])
    # M1[1,1] = (A[1,0]+A[3,2])*(B[0,1]+B[2,3]) + (A[1,1]+A[3,3])*(B[1,1]+B[3,3])
    #
    # Each M1[di,dk] is a sum of 2 products. Each product is of the form
    # (linear_comb_of_A) * (linear_comb_of_B).
    #
    # In the tensor decomposition, a rank-1 term is:
    # u_i * v_j * w_k
    # where u, v, w are fixed vectors.
    #
    # The product (A[a,b]+A[c,d])*(B[e,f]+B[g,h]) cannot be represented as a single
    # rank-1 term in the (i,j,k) space. It would need to be expanded.
    #
    # (A[a,b]+A[c,d])*(B[e,f]+B[g,h])
    # = A[a,b]*B[e,f] + A[a,b]*B[g,h] + A[c,d]*B[e,f] + A[c,d]*B[g,h]
    #
    # Each of these 4 terms is a rank-1 term in the tensor space!
    # A[a,b]*B[e,f] corresponds to the (a,b,f) triple (if we're computing C[a,f]).
    #
    # So M1[0,0] = (A[0,0]+A[2,2])*(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])*(B[1,0]+B[3,2])
    # = [A[0,0]*B[0,0] + A[0,0]*B[2,2] + A[2,2]*B[0,0] + A[2,2]*B[2,2]]
    #   + [A[0,1]*B[1,0] + A[0,1]*B[3,2] + A[2,3]*B[1,0] + A[2,3]*B[3,2]]
    #
    # This is 8 rank-1 terms for just M1[0,0]!
    #
    # So M1 (a 2x2 matrix) has 4 entries, each with 8 rank-1 terms = 32 terms?
    # That can't be right because Strassen is supposed to REDUCE the number of terms.
    #
    # I think I'm confusing myself. Let me step back.
    #
    # The tensor T for 4x4*4x4 has 4*4*4 = 64 entries, all equal to 1.
    # The rank of this tensor is at most 64 (schoolbook).
    # Strassen shows that the rank is at most 28 (7 block mults * 4 scalar mults each).
    #
    # Wait, no. Strassen shows that you can compute the product with 7 block multiplications
    # instead of 8. Each block multiplication is a 2x2*2x2 product, which requires 4 scalar
    # multiplications. So total scalar multiplications: 7*4 = 28.
    #
    # But the tensor rank is not the same as the number of scalar multiplications in an algorithm.
    # The tensor rank is the minimum number of rank-1 tensors that sum to T.
    #
    # For the 2x2*2x2 multiplication tensor, the rank is 7 (Strassen).
    # For the 4x4*4x4 multiplication tensor, the rank is at most 7*4 = 28 if we use
    # Strassen at the block level and schoolbook at the scalar level within each block.
    #
    # But actually, I don't think the rank is 28. The rank of the 4x4*4x4 tensor is
    # known to be at most 49 (using Strassen recursively) or 28 (using Strassen once).
    #
    # Let me just verify: if I use Strassen for 4x4*4x4, I get 7 block multiplications.
    # Each block multiplication is a 2x2*2x2 product. The 2x2*2x2 product tensor has
    # rank 7 (Strassen) or 8 (schoolbook). If I use schoolbook for each block, that's
    # 7*8 = 56 terms. If I use Strassen for each block, that's 7*7 = 49 terms.
    #
    # Hmm, but that would be recursive Strassen.
    #
    # I think the correct count is:
    # - Use Strassen once for 4x4*4x4: 7 block multiplications of 2x2*2x2
    # - Each 2x2*2x2 uses schoolbook: 4 scalar multiplications
    # - Total: 7*4 = 28 scalar rank-1 terms
    #
    # But I need to verify that these 28 terms actually sum to the 4x4*4x4 tensor.
    #
    # Let me try a different approach. Instead of trying to derive the 28 terms,
    # let me just use schoolbook for the entire 4x4*4x7 problem and get rank 112,
    # then see if I can improve it.
    #
    # Actually, let me just implement a correct Strassen-based decomposition.
    # I'll be very careful about the expansion.
    
    # Let me try yet another approach. I'll build the decomposition by
    # explicitly listing all rank-1 terms.
    #
    # For the 4x4x4 part, I'll use the following strategy:
    # - The 4x4x4 tensor T4[i,j,k] = 1 for all i,j,k in 0..3
    # - I want to decompose T4 into rank-1 terms
    # - Using Strassen, I can do this with 28 terms
    #
    # Here's the explicit list of 28 terms for the 4x4x4 part:
    #
    # M1 = (A11+A22)*(B11+B22)
    # The 4 scalar multiplications in M1 are:
    # For di in 0,1 and dk in 0,1:
    #   M1[di,dk] involves products of A and B entries
    #
    # I think the cleanest way is to just use schoolbook for now and ensure
    # correctness, then optimize.
    
    # Schoolbook for 4x4*4x7: rank = 112
    # This is correct but not optimal.
    
    # Let me try to implement Strassen correctly.
    # 
    # Key realization: In the tensor decomposition, each rank-1 term is:
    #   u * v * w
    # where u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    #
    # A rank-1 term u*v*w contributes to T[i,j,k] the value:
    #   u[i*m+j] * v[j*p+k] * w[k*n+i]
    #
    # For the matrix multiplication tensor, T[i,j,k] = 1.
    #
    # A simple rank-1 term is: e_{i,j} * e_{j,k} * e_{k,i}
    # where e_{i,j} is the unit vector with 1 at position i*m+j.
    # This contributes 1 to T[i,j,k] and 0 to all other entries.
    #
    # Strassen's algorithm replaces some of these simple terms with combinations.
    #
    # Let me define the 28 terms for the 4x4x4 part using Strassen.
    #
    # I'll use the following convention:
    # A block multiplication M = X*Y where X is 2x2, Y is 2x2
    # M[di,dk] = sum_{dj} X[di,dj]*Y[dj,dk]
    #
    # Each M[di,dk] is a scalar that is a sum of products of A and B entries.
    # In the tensor decomposition, each product A[a,b]*B[c,d] that contributes
    # to C[i,k] corresponds to a rank-1 term with:
    #   u[a*m+b] = 1, v[c*p+d] = 1, w[d*n+a] = 1
    # (and 0 elsewhere)
    #
    # But if X[di,dj] = sum of several A entries, and Y[dj,dk] = sum of several
    # B entries, then X[di,dj]*Y[dj,dk] is a sum of products, each of which
    # is a rank-1 term.
    #
    # So the total number of rank-1 terms is the sum over all (di,dj,dk) of
    # (number of A entries in X[di,dj]) * (number of B entries in Y[dj,dk]).
    #
    # For M1 = (A11+A22)*(B11+B22):
    # X[di,dj] = A[di,dj] + A[di+2,dj+2] (2 A entries)
    # Y[dj,dk] = B[dj,dk] + B[dj+2,dk+2] (2 B entries)
    # For each (di,dj,dk): 2*2 = 4 rank-1 terms
    # Total for M1: 2*2*2 * 4 = 32 terms? No, that's not right.
    #
    # Wait, M1 is a 2x2 matrix, so di,dk in 0,1. And dj in 0,1.
    # M1[di,dk] = sum_{dj} X[di,dj]*Y[dj,dk]
    # For fixed (di,dk), dj ranges over 0,1, so 2 terms in the sum.
    # Each term X[di,dj]*Y[dj,dk] is a product of (sum of 2 A's) and (sum of 2 B's)
    # = 4 scalar products.
    # So M1[di,dk] has 2*4 = 8 scalar products.
    # M1 has 4 entries (di,dk in 0,1), so 4*8 = 32 scalar products.
    #
    # That's way more than 4! I'm clearly misunderstanding something.
    #
    # OH WAIT. I think the issue is that in Strassen's algorithm, the "multiplications"
    # M1, M2, ..., M7 are not independent scalar multiplications. They are block
    # multiplications where the blocks are linear combinations. The number of scalar
    # multiplications is not simply 7*4.
    #
    # Let me count the scalar multiplications in Strassen for 4x4*4x4:
    # M1 = (A11+A22)*(B11+B22)
    # Computing A11+A22: 4 additions
    # Computing B11+B22: 4 additions
    # Multiplying two 2x2 matrices: 8 scalar multiplications
    # So M1 costs 8 scalar multiplications.
    #
    # Similarly for M2 through M7: each costs 8 scalar multiplications.
    # Total: 7*8 = 56 scalar multiplications.
    #
    # That's worse than schoolbook (64)! That can't be right.
    #
    # I think I'm confusing the number of scalar multiplications with the tensor rank.
    #
    # Let me look at this from a different angle. The tensor rank of the 2x2*2x2
    # multiplication tensor is 7 (Strassen). This means that the 2x2*2x2 tensor
    # (which has 8 entries, all 1) can be written as a sum of 7 rank-1 tensors.
    #
    # A rank-1 tensor in the 2x2x2 space is u*v*w where u,v,w are vectors in R^2, R^2, R^2.
    # The entry (i,j,k) is u[i]*v[j]*w[k].
    #
    # So the 2x2*2x2 tensor T2[i,j,k] = 1 = sum_{r=1}^{7} u_r[i]*v_r[j]*w_r[k]
    #
    # Now, for the 4x4*4x4 tensor, we can use the block structure.
    # The 4x4*4x4 tensor T4[i,j,k] = 1 can be viewed as a 2x2 block matrix
    # of 2x2x2 sub-tensors.
    #
    # T4 = [[T4_11, T4_12], [T4_21, T4_22]]
    # where T4_ab is a 2x2x2 tensor.
    #
    # Using Strassen, we can express the 8 block-level rank-1 terms as 7.
    # Each block-level rank-1 term corresponds to a 2x2x2 sub-tensor.
    #
    # So the total rank is 7 * rank(2x2x2 tensor) = 7 * 4 = 28 if we use
    # schoolbook for each 2x2x2 sub-tensor, or 7 * 7 = 49 if we use Strassen
    # for each sub-tensor.
    #
    # But wait, the 2x2x2 sub-tensors in the 4x4x4 tensor are not all the same.
    # Some of them are zero!
    #
    # T4[i,j,k] = 1 for all i,j,k in 0..3.
    # So all 8 sub-tensors are the full 2x2x2 tensor with all entries 1.
    #
    # So the rank of T4 is at most 7 * 4 = 28 (using Strassen at the block level
    # and schoolbook at the sub-block level).
    #
    # OK so I believe the rank is 28 for the 4x4x4 part.
    #
    # Now I need to actually construct the 28 rank-1 terms.
    #
    # Let me use the explicit Strassen decomposition for 2x2x2.
    #
    # The 2x2x2 tensor T2[i,j,k] = 1 for i,j,k in 0,1.
    # Strassen's 7 rank-1 terms:
    # r1: u=(1,1), v=(1,1), w=(1,1)  -> contributes 1 to all 8 entries
    # r2: u=(0,1), v=(1,1), w=(1,0)  -> contributes to (1,0,0), (1,1,0)
    # r3: u=(1,0), v=(0,1), w=(1,0)  -> contributes to (0,1,0)
    # r4: u=(1,0), v=(1,0), w=(0,1)  -> contributes to (0,0,1)
    # r5: u=(1,1), v=(0,1), w=(0,1)  -> contributes to (0,1,1), (1,1,1)
    # r6: u=(0,1), v=(1,0), w=(1,1)  -> contributes to (1,0,0), (1,0,1)
    # r7: u=(0,1), v=(0,1), w=(1,0)  -> contributes to (1,1,0)
    #
    # Let me verify: sum of all 7 terms should give 1 for each (i,j,k).
    #
    # (0,0,0): r1=1, others=0 -> 1 ✓
    # (0,0,1): r1=1, r4=1 -> 2 ✗
    #
    # That's not right. Let me look up the correct Strassen decomposition.
    #
    # The correct Strassen decomposition for 2x2*2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M_t is a scalar (for 2x2*2x2, the blocks are 1x1).
    #
    # For the 2x2*2x2 case:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Now, each M_t is a product of two linear combinations.
    # M1 = (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    #
    # In the tensor decomposition, each product A[i,j]*B[j,k] that contributes
    # to C[i',k'] corresponds to a rank-1 term.
    #
    # But M1 contributes to both C11 and C22.
    #
    # This is getting very complex. Let me just use schoolbook for the entire
    # problem and get rank 112, which is correct.
    
    # Build the tensor decomposition using Strassen for 4x4x4 part and schoolbook for 4x4x3 part.
    # 
    # Each rank-1 term is specified by:
    #   A_indices: list of (i,j) pairs with coefficients for U
    #   B_indices: list of (j,k) pairs with coefficients for V  
    #   C_indices: list of (k,i) pairs with coefficients for W
    #
    # The term contributes to T[i,j,k] the product of the U, V, W coefficients.
    #
    # For the decomposition to work, we need:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    terms = []  # Each term: list of (index, coeff) for U, V, W
    
    def add_term(U_entries, V_entries, W_entries):
        """Add a rank-1 term. U_entries, V_entries, W_entries are lists of (idx, coeff)."""
        terms.append((U_entries, V_entries, W_entries))
    
    # ===== 4x4x4 part using Strassen =====
    # 
    # Strassen for 2x2*2x2 (scalar blocks):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4*4x4, we apply Strassen at the block level where blocks are 2x2.
    # Each block multiplication is a 2x2*2x2 product.
    #
    # Instead of trying to derive all 28 terms, let me use a simpler approach:
    # Use Strassen for the outer 2x2 block structure, and for each of the 7 block
    # multiplications, use schoolbook (4 terms) for the inner 2x2*2x2 product.
    #
    # Block indices: A block (bi,bj) has entries A[2*bi+di, 2*bj+dj] for di,dj in 0,1
    #
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])  where A[bi,bj] is a 2x2 block
    #
    # (A[0,0]+A[1,1]) is a 2x2 matrix with entries A[2*0+di, 2*0+dj] + A[2*1+di, 2*1+dj]
    # = A[di,dj] + A[di+2,dj+2]
    #
    # (B[0,0]+B[1,1]) is a 2x2 matrix with entries B[2*0+dj, 2*0+dk] + B[2*1+dj, 2*1+dk]
    # = B[dj,dk] + B[dj+2,dk+2]
    #
    # M1[di,dk] = sum_{dj} (A[di,dj]+A[di+2,dj+2]) * (B[dj,dk]+B[dj+2,dk+2])
    #
    # This is a 2x2 matrix. M1 contributes to C[0,0] (top-left 2x2) and C[1,1] (bottom-right 2x2).
    #
    # The scalar expansion of M1:
    # M1[di,dk] = sum_{dj} (A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2])
    #
    # Each product A[a,b]*B[c,d] where c==b (for the product to be valid) contributes to C[a,d].
    #
    # Wait, in matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]. So the product A[a,b]*B[c,d]
    # only contributes to C[a,d] if b==c.
    #
    # In M1, the products are:
    # A[di,dj]*B[dj,dk] -> contributes to C[di,dk] (since the B index matches: dj==dj)
    # A[di,dj]*B[dj+2,dk+2] -> does NOT contribute to any C entry in the standard way because
    #   the A column index (dj) doesn't match the B row index (dj+2) unless we're in a different context.
    #
    # I think the issue is that in Strassen, the "multiplications" are not standard matrix
    # multiplications of the original A and B. They are multiplications of linear combinations.
    #
    # Let me try a completely different approach. I'll just use schoolbook and accept rank 112.
    
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k, 1, 1, 1))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (i, j, k, u, v, w) in enumerate(terms):
        U[i * m + j][r] = u
        V[j * p + k][r] = v
        W[k * n + i][r] = w
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
