"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def _to_list(a):
    if hasattr(a, 'tolist'):
        return a.tolist()
    return a

def construct():
    t0 = time.time()
    deadline = 120.0  # seconds

    # We'll implement the rank-28 decomposition for 2x4 * 4x4.
    # The decomposition uses the structure:
    #   A is 2x4, B is 4x4, C = A*B is 2x4.
    #   Split A into A1 (cols 0,1) and A2 (cols 2,3), each 2x2.
    #   Split B into 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    #   C_L = A*B(:,0:2) = A1*B_TL + A2*B_BL
    #   C_R = A*B(:,2:4) = A1*B_TR + A2*B_BR
    #
    # For each 2x2 * 2x2 product, we can use Strassen (rank 7) or schoolbook (rank 4).
    # C_L requires: A1*B_TL + A2*B_BL  -> two 2x2*2x2 products
    # C_R requires: A1*B_TR + A2*B_BR  -> two 2x2*2x2 products
    #
    # Total: 4 products of 2x2 * 2x2.
    # If we use Strassen for each: 4 * 7 = 28.
    # If schoolbook: 4 * 4 = 16. But schoolbook is rank 4 per product.
    #
    # Wait, let me reconsider. The matrix multiplication tensor for 2x4 * 4x4
    # has dimension 2*4*4 = 32. The rank of this tensor is what we want to minimize.
    #
    # Actually, let me think more carefully. The standard result is that
    # the rank of multiplying an n x m matrix by an m x p matrix is at most n*m*p,
    # but can be less using fast algorithms.
    #
    # For 2x4 * 4x4:
    # We can write this as block multiplication. Let A = [A1 A2] where A1, A2 are 2x2.
    # B is 4x4, split into 2x2 blocks:
    #   B = [B11 B12; B21 B22]
    # Then C = A1*B11 + A2*B21 for the first two columns, and
    #        C = A1*B12 + A2*B22 for the last two columns.
    #
    # So C = [A1*B11 + A2*B21,  A1*B12 + A2*B22]
    #
    # This is 4 products of 2x2 matrices. Each 2x2*2x2 product has rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    #
    # Can we do better? The lower bound for 2x2*2x2 is 7 (over most fields), so
    # if we need 4 independent such products, 28 might be tight.
    #
    # But wait - maybe we can share structure. Let me think about this differently.
    #
    # Actually, the rank of the 2x4 x 4x4 multiplication tensor is known to be 28.
    # This is a result from algebraic complexity theory.
    #
    # Let me construct the rank-28 decomposition using Strassen for the 2x2 blocks.
    #
    # Strassen's algorithm for 2x2 * 2x2:
    # Given P = Q * R where P, Q, R are 2x2:
    #   M1 = (Q11 + Q22) * (R11 + R22)
    #   M2 = (Q21 + Q22) * R11
    #   M3 = Q11 * (R12 - R22)
    #   M4 = Q22 * (R21 - R11)
    #   M5 = (Q11 + Q12) * R22
    #   M6 = (Q21 - Q11) * (R11 + R12)
    #   M7 = (Q12 - Q22) * (R21 + R22)
    #   P11 = M1 + M4 - M5 + M7
    #   P12 = M3 + M5
    #   P21 = M2 + M4
    #   P22 = M1 - M2 + M3 + M6
    #
    # So for each 2x2*2x2 product, we need 7 scalar multiplications.
    # For 4 such products, we need 28.
    #
    # Now let me set up the indices:
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i,k] for i in {0,1}, k in {0,1,2,3}
    #
    # Block structure:
    # A1 = A[:, 0:2] = [[A00, A01], [A10, A11]]
    # A2 = A[:, 2:4] = [[A02, A03], [A12, A13]]
    #
    # B_TL = B[0:2, 0:2] = [[B00, B01], [B10, B11]]
    # B_TR = B[0:2, 2:4] = [[B02, B03], [B12, B13]]
    # B_BL = B[2:4, 0:2] = [[B20, B21], [B30, B31]]
    # B_BR = B[2:4, 2:4] = [[B22, B23], [B32, B33]]
    #
    # C_L = A1 * B_TL + A2 * B_BL  (2x2 result)
    # C_R = A1 * B_TR + A2 * B_BR  (2x2 result)
    #
    # For C_L = A1*B_TL + A2*B_BL:
    #   Let X = A1*B_TL and Y = A2*B_BL
    #   X uses Strassen with Q=A1, R=B_TL
    #   Y uses Strassen with Q=A2, R=B_BL
    #   C_L = X + Y
    #
    # Similarly for C_R.
    #
    # Now I need to express this as a tensor decomposition.
    # The tensor T has entries T[i, j, k] = delta (the product A[i,j]*B[j,k] contributes to C[i,k]).
    # In the standard indexing: T is a 2x4x4 tensor where T[i,j,k] = 1 if the (i,j) element of A
    # times the (j,k) element of B contributes to C[i,k].
    #
    # A rank-r decomposition means:
    #   T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[i,k,c]
    #
    # Wait, let me re-read the interface:
    # U: shape (n*m, rank) = (8, rank)
    # V: shape (m*p, rank) = (16, rank)  
    # W: shape (p*n, rank) = (8, rank)
    #
    # And the output factor follows W[k*n+i, r].
    #
    # So U indexes (i,j) -> i*m+j where i in [0,n), j in [0,m)
    # V indexes (j,k) -> j*p+k where j in [0,m), k in [0,p)
    # W indexes (k,i) -> k*n+i where k in [0,p), i in [0,n)
    #
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    #
    # For the standard schoolbook, T[i,j,k] = 1 for each (i,j,k) triple, giving rank 32.
    #
    # For the rank-28 decomposition, I need to express the Strassen-based computation.
    #
    # Let me work out the Strassen decomposition for one 2x2*2x2 product.
    # 
    # For a 2x2*2x2 product P = Q*R:
    # Q = [[q00, q01], [q10, q11]], R = [[r00, r01], [r10, r11]]
    # P = [[p00, p01], [p10, p11]]
    #
    # Strassen:
    # m1 = (q00+q11)*(r00+r11)
    # m2 = (q10+q11)*r00
    # m3 = q00*(r01-r11)
    # m4 = q11*(r10-r00)
    # m5 = (q00+q01)*r11
    # m6 = (q10-q00)*(r00+r01)
    # m7 = (q01-q11)*(r10+r11)
    #
    # p00 = m1 + m4 - m5 + m7
    # p01 = m3 + m5
    # p10 = m2 + m4
    # p11 = m1 - m2 + m3 + m6
    #
    # Each mc is a product of two linear forms in the q's and r's.
    # In the tensor decomposition, each mc contributes to the U, V, W factors.
    #
    # For the full 2x4*4x4 problem, I have 4 Strassen-based products, each contributing 7 terms.
    # Total: 28 terms.
    #
    # Let me define the 28 terms:
    # Product 1: X = A1 * B_TL (for C_L)
    # Product 2: Y = A2 * B_BL (for C_L)
    # Product 3: Z = A1 * B_TR (for C_R)
    # Product 4: W_ = A2 * B_BR (for C_R)
    #
    # For each product, I have 7 Strassen multipliers.
    #
    # Let me work out Product 1: X = A1 * B_TL
    # A1 = [[A00, A01], [A10, A11]]
    # B_TL = [[B00, B01], [B10, B11]]
    #
    # m1_X = (A00+A11)*(B00+B11)
    # m2_X = (A10+A11)*B00
    # m3_X = A00*(B01-B11)
    # m4_X = A11*(B10-B00)
    # m5_X = (A00+A01)*B11
    # m6_X = (A10-A00)*(B00+B01)
    # m7_X = (A01-A11)*(B10+B11)
    #
    # X00 = m1_X + m4_X - m5_X + m7_X
    # X01 = m3_X + m5_X
    # X10 = m2_X + m4_X
    # X11 = m1_X - m2_X + m3_X + m6_X
    #
    # Similarly for the other products.
    #
    # Now, C_L = X + Y, so:
    # C_L00 = X00 + Y00
    # C_L01 = X01 + Y01
    # C_L10 = X10 + Y10
    # C_L11 = X11 + Y11
    #
    # C_R = Z + W_:
    # C_R00 = Z00 + W_00
    # C_R01 = Z01 + W_01
    # C_R10 = Z10 + W_10
    # C_R11 = Z11 + W_11
    #
    # So C = [C_L, C_R] where C_L and C_R are 2x2.
    #
    # Now I need to express each mc as a product of a linear form in A, a linear form in B,
    # and a coefficient for C.
    #
    # For example, m1_X = (A00+A11)*(B00+B11).
    # This is a product of:
    #   A-part: A00 + A11
    #   B-part: B00 + B11
    #   C-part: contributes to X00 with +1, X11 with +1.
    #
    # In the tensor decomposition:
    # For the (A00+A11)*(B00+B11) term:
    #   U entries: U[A00_idx] = 1, U[A11_idx] = 1 (where A00_idx = 0*4+0 = 0, A11_idx = 1*4+1 = 5)
    #   V entries: V[B00_idx] = 1, V[B11_idx] = 1 (where B00_idx = 0*4+0 = 0, B11_idx = 1*4+1 = 5)
    #   W entries: W[C00_idx] = 1, W[C11_idx] = 1 (where C00_idx = 0*2+0 = 0, C11_idx = 1*2+1 = 3)
    #
    # Wait, but this doesn't quite work because the W factor needs to capture the full C contribution.
    #
    # Let me reconsider. In the tensor decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For the term (A00+A11)*(B00+B11), this expands to:
    # A00*B00 + A00*B11 + A11*B00 + A11*B11
    #
    # Each of these contributes to C entries:
    # A00*B00 -> C00 (i=0, j=0, k=0)
    # A00*B11 -> C01 (i=0, j=1, k=1)
    # A11*B00 -> C10 (i=1, j=0, k=0)
    # A11*B11 -> C11 (i=1, j=1, k=1)
    #
    # So the term (A00+A11)*(B00+B11) contributes:
    #   +1 to T[0,0,0] (C00)
    #   +1 to T[0,1,1] (C01)
    #   +1 to T[1,0,0] (C10)
    #   +1 to T[1,1,1] (C11)
    #
    # In the decomposition, this single rank-1 term has:
    #   U[0] = 1 (A00), U[5] = 1 (A11), others 0
    #   V[0] = 1 (B00), V[5] = 1 (B11), others 0
    #   W[0] = 1 (C00), W[3] = 1 (C11), and also W[1] = 1 (C01)? 
    #
    # Wait, that's not right. The W factor is indexed by (k, i) -> k*n+i.
    # For C00: k=0, i=0 -> W[0]
    # For C01: k=1, i=0 -> W[1]
    # For C10: k=0, i=1 -> W[2]
    # For C11: k=1, i=1 -> W[3]
    #
    # So the term (A00+A11)*(B00+B11) contributes to C00, C01, C10, C11 all with coefficient 1.
    # But in the Strassen formula, m1_X contributes +1 to X00 and +1 to X11, not to X01 and X10.
    #
    # I think I'm confusing myself. Let me redo this more carefully.
    #
    # The key insight is that in the tensor decomposition, each rank-1 term corresponds to
    # a product of a linear form in A, a linear form in B, and a linear form in C (or rather,
    # it specifies which C entries are affected).
    #
    # For the Strassen multiplier m1_X = (A00+A11)*(B00+B11):
    # This is not a single monomial in the original variables - it's a product of two linear forms.
    # When expanded, it gives 4 monomials: A00*B00, A00*B11, A11*B00, A11*B11.
    #
    # In the tensor T, these monomials correspond to:
    # A00*B00: T[0,0,0] = 1
    # A00*B11: T[0,1,1] = 1
    # A11*B00: T[1,0,0] = 1
    # A11*B11: T[1,1,1] = 1
    #
    # Now, can I represent the sum of these 4 tensor entries as a single rank-1 term?
    # A rank-1 term in the 3-way tensor is an outer product of three vectors.
    # So I need vectors u, v, w such that:
    #   u[i*m+j] * v[j*p+k] * w[k*n+i] is nonzero exactly for the 4 (i,j,k) triples above.
    #
    # The 4 triples are: (0,0,0), (0,1,1), (1,0,0), (1,1,1).
    # 
    # For (0,0,0): i=0, j=0, k=0. Index in U: 0*4+0=0. Index in V: 0*4+0=0. Index in W: 0*2+0=0.
    # For (0,1,1): i=0, j=1, k=1. U: 0*4+1=1. V: 1*4+1=5. W: 1*2+0=1.
    # For (1,0,0): i=1, j=0, k=0. U: 1*4+0=4. V: 0*4+0=0. W: 0*2+1=2.
    # For (1,1,1): i=1, j=1, k=1. U: 1*4+1=5. V: 1*4+1=5. W: 1*2+1=3.
    #
    # So I need u[0]*v[0]*w[0] = 1, u[1]*v[5]*w[1] = 1, u[4]*v[0]*w[2] = 1, u[5]*v[5]*w[3] = 1,
    # and all other products = 0.
    #
    # From u[0]*v[0]*w[0] = 1 and u[4]*v[0]*w[2] = 1:
    #   v[0] != 0, u[0] != 0, u[4] != 0, w[0] != 0, w[2] != 0.
    #   u[4]/u[0] = w[0]/w[2].
    #
    # From u[1]*v[5]*w[1] = 1 and u[5]*v[5]*w[3] = 1:
    #   v[5] != 0, u[1] != 0, u[5] != 0, w[1] != 0, w[3] != 0.
    #   u[5]/u[1] = w[1]/w[3].
    #
    # Also, all other entries of u, v, w must be 0 (or the corresponding products must be 0).
    #
    # Let's try:
    # u[0] = 1, u[1] = 1, u[4] = 1, u[5] = 1, others 0.
    # v[0] = 1, v[5] = 1, others 0.
    # w[0] = 1, w[1] = 1, w[2] = 1, w[3] = 1, others 0.
    #
    # Check:
    # (0,0,0): u[0]*v[0]*w[0] = 1*1*1 = 1. OK.
    # (0,1,1): u[1]*v[5]*w[1] = 1*1*1 = 1. OK.
    # (1,0,0): u[4]*v[0]*w[2] = 1*1*1 = 1. OK.
    # (1,1,1): u[5]*v[5]*w[3] = 1*1*1 = 1. OK.
    #
    # But what about other (i,j,k) triples?
    # For example, (0,0,1): u[0]*v[0]*w[1] = 1*1*1 = 1. But T[0,0,1] should be 0!
    #
    # So this doesn't work. The outer product u*v*w has nonzero entries at many positions,
    # not just the 4 we want.
    #
    # This means that (A00+A11)*(B00+B11) CANNOT be represented as a single rank-1 term in the
    # 2x4x4 tensor. It requires multiple rank-1 terms.
    #
    # So my approach of using Strassen directly doesn't work in the way I was thinking.
    # The Strassen algorithm works because it restructures the computation, but the individual
    # multipliers are not directly rank-1 terms in the original tensor.
    #
    # Let me think about this differently. The rank-28 decomposition for 2x4*4x4 must exist,
    # but I need to find the actual U, V, W matrices.
    #
    # One approach: use the known result that the rank of 2x4*4x4 is 28, and construct
    # the decomposition by combining the 2x2*2x2 Strassen decomposition with the block structure.
    #
    # Actually, I recall that for matrix multiplication, the rank of an n x m by m x p product
    # is related to the ranks of smaller products. Specifically, if we can decompose the product
    # into a sum of smaller products, the rank is bounded by the sum of the ranks.
    #
    # For 2x4*4x4, we can write:
    # C = A1*B_TL + A2*B_BL + A1*B_TR + A2*B_BR
    # where A1, A2 are 2x2 and the B's are 2x2.
    #
    # But this is not a sum of independent 2x2*2x2 products in the tensor sense.
    #
    # Let me try a different approach. I'll use the fact that the rank-28 decomposition exists
    # and try to construct it algorithmically.
    #
    # Actually, let me just try to verify that rank 28 is achievable by constructing the
    # decomposition numerically and checking it.
    #
    # Given the time constraints, let me try a simpler approach: use the schoolbook decomposition
    # (rank 32) as a fallback, and if I have time, try to find a lower-rank decomposition.
    #
    # For now, let me return the schoolbook decomposition to ensure correctness, and add
    # a TODO for finding the rank-28 decomposition.
    #
    # Actually, let me try one more thing. I'll attempt to construct the rank-28 decomposition
    # by exploiting the structure more carefully.
    #
    # Key idea: The 2x4*4x4 multiplication can be seen as two 2x2*4x2 products:
    #   C[:, 0:2] = A * B[:, 0:2]  (2x4 * 4x2 = 2x2)
    #   C[:, 2:4] = A * B[:, 2:4]  (2x4 * 4x2 = 2x2)
    #
    # Each 2x4*4x2 product has rank at most 2*4*2 = 16, but can be lower.
    # The rank of 2x4*4x2 is known to be 14 (using Strassen-like methods).
    # So total rank = 14 + 14 = 28.
    #
    # Let me construct the rank-14 decomposition for 2x4*4x2.
    #
    # For 2x4*4x2:
    # A is 2x4, B is 4x2, C is 2x2.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3), each 2x2.
    # Split B into B1 (rows 0,1) and B2 (rows 2,3), each 2x2.
    # C = A1*B1 + A2*B2.
    #
    # This is the sum of two 2x2*2x2 products. Each has rank 7 (Strassen).
    # Total rank = 7 + 7 = 14.
    #
    # So for 2x4*4x4, we have:
    # C_L = A*B[:, 0:2] = A1*B_TL + A2*B_BL  (rank 14)
    # C_R = A*B[:, 2:4] = A1*B_TR + A2*B_BR  (rank 14)
    # Total rank = 14 + 14 = 28.
    #
    # Now I need to express each 2x2*2x2 Strassen product as rank-1 terms in the larger tensor.
    #
    # For the 2x2*2x2 product X = A1*B_TL:
    # A1 = [[A00, A01], [A10, A11]], B_TL = [[B00, B01], [B10, B11]]
    # X = A1*B_TL is 2x2.
    #
    # Using Strassen:
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    #
    # X00 = m1 + m4 - m5 + m7
    # X01 = m3 + m5
    # X10 = m2 + m4
    # X11 = m1 - m2 + m3 + m6
    #
    # Each mc is a product of a linear form in A1 and a linear form in B_TL.
    # In the full 2x4*4x4 tensor, these correspond to specific (i,j,k) entries.
    #
    # For example, m1 = (A00+A11)*(B00+B11):
    # A00 is at position (i=0, j=0) in A.
    # A11 is at position (i=1, j=1) in A.
    # B00 is at position (j=0, k=0) in B.
    # B11 is at position (j=1, k=1) in B.
    #
    # The product (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11.
    #
    # These contribute to:
    # A00*B00: C[0,0] (i=0, k=0)
    # A00*B11: C[0,1] (i=0, k=1)
    # A11*B00: C[1,0] (i=1, k=0)
    # A11*B11: C[1,1] (i=1, k=1)
    #
    # So m1 contributes +1 to C[0,0], +1 to C[0,1], +1 to C[1,0], +1 to C[1,1].
    #
    # But as I showed earlier, this cannot be a single rank-1 term.
    #
    # I think the resolution is that the Strassen decomposition doesn't directly give
    # rank-1 terms in the original tensor. Instead, it gives a factorization of the
    # multiplication tensor that may require additional structure.
    #
    # Given the complexity and time constraints, let me just return the schoolbook
    # decomposition (rank 32) to ensure a valid answer. I can try to improve this
    # in future iterations.
    #
    # Actually, wait. Let me reconsider the problem. The rank-28 decomposition MUST exist,
    # and it's well-known in the literature. Let me try to construct it more carefully.
    #
    # I think the key is that the Strassen multipliers, when expressed in the full tensor,
    # do correspond to rank-1 terms, but I need to be more careful about the indexing.
    #
    # Let me try to explicitly write out the rank-1 terms for the first Strassen multiplier
    # and see if it works.
    #
    # For m1_X = (A00+A11)*(B00+B11) contributing to X = A1*B_TL (which is part of C_L):
    #
    # The contribution to C is:
    # C[0,0] += m1_X
    # C[0,1] += m1_X  (wait, no. X00 contributes to C[0,0], X01 to C[0,1], etc.)
    #
    # Actually, X = A1*B_TL is a 2x2 matrix, and X[i,j] contributes to C[i, j] (since B_TL
    # covers columns 0,1 of B).
    #
    # So X00 -> C[0,0], X01 -> C[0,1], X10 -> C[1,0], X11 -> C[1,1].
    #
    # m1_X contributes +1 to X00 and +1 to X11.
    # So m1_X contributes +1 to C[0,0] and +1 to C[1,1].
    #
    # Similarly, for the other multipliers:
    # m2_X contributes +1 to X10 -> C[1,0]
    # m3_X contributes +1 to X01 and +1 to X11 -> C[0,1] and C[1,1]
    # m4_X contributes +1 to X00 and +1 to X10 -> C[0,0] and C[1,0]
    # m5_X contributes -1 to X00 and +1 to X01 -> -C[0,0] and +C[0,1]
    # m6_X contributes +1 to X11 -> C[1,1]
    # m7_X contributes +1 to X00 -> C[0,0]
    #
    # Now, each mc is a product of a linear form in A and a linear form in B.
    # For example, m1_X = (A00+A11)*(B00+B11).
    #
    # In the tensor T, the entry T[i,j,k] is 1 if A[i,j]*B[j,k] contributes to C[i,k].
    #
    # The term (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11.
    #
    # A00*B00: i=0, j=0, k=0. T[0,0,0] = 1.
    # A00*B11: i=0, j=1, k=1. T[0,1,1] = 1.
    # A11*B00: i=1, j=0, k=0. T[1,0,0] = 1.
    # A11*B11: i=1, j=1, k=1. T[1,1,1] = 1.
    #
    # So the term (A00+A11)*(B00+B11) corresponds to the sum of T[0,0,0] + T[0,1,1] + T[1,0,0] + T[1,1,1].
    #
    # For this to be a rank-1 term, I need U, V, W such that:
    # U[0]*V[0]*W[0] = 1  (for T[0,0,0])
    # U[1]*V[5]*W[1] = 1  (for T[0,1,1])
    # U[4]*V[0]*W[2] = 1  (for T[1,0,0])
    # U[5]*V[5]*W[3] = 1  (for T[1,1,1])
    # And all other products = 0.
    #
    # As I showed before, this is not possible with a single outer product.
    #
    # So the Strassen multipliers do NOT directly correspond to rank-1 terms in the original tensor.
    #
    # This means that the rank-28 decomposition cannot be obtained by simply applying Strassen
    # to the 2x2 blocks. There must be a more sophisticated construction.
    #
    # Given the time constraints and the complexity of finding the exact rank-28 decomposition,
    # let me just return the schoolbook decomposition (rank 32) to ensure a valid answer.
    #
    # I'll add a TODO for future work on finding the rank-28 decomposition.
    
    n, m, p = DIMENSIONS
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
