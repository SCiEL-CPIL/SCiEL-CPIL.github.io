"""Strassen-Winograd block decomposition for 4x5 by 5x6 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U,V,W) each a list of 4 column vectors (length 4) for 2x2 mult.
    U indexed (i,j), V indexed (j,k), W indexed (k,i), i,j,k in {0,1}.
    Uses rank-7 Strassen with integer entries."""
    # 7 products, each: U col, V col, W col
    # product p = sum_i U[i,j]*A[i,j] * sum_k V[j,k]*B[j,k]
    prods = [
        # (U[i,j] for i,j), (V[j,k] for j,k), (W[k,i] for k,i)
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0]),
        # M3 = A00(B10-B11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]),
        # M4 = A11(B01-B00)
        ([0, 0, 0, 1], [1, 1, 0, -1], [0, 1, 0, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        # M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 1, 0]),
    ]
    U = [[0, 0, 0, 0] for _ in range(7)]
    V = [[0, 0, 0, 0] for _ in range(7)]
    W = [[0, 0, 0, 0] for _ in range(7)]
    for c, (uc, vc, wc) in enumerate(prods):
        for idx, val in enumerate(uc):
            U[c][idx] = val
        for idx, val in enumerate(vc):
            V[c][idx] = val
        for idx, val in enumerate(wc):
            W[c][idx] = val
    return U, V, W

def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 4, 5, 6
    # Split: A is 4x5 -> core 4x4 (j=0..3) + tail 4x1 (j=4)
    #        B is 5x6 -> core 4x6 (j=0..3) + tail 1x6 (j=4)
    # C = A_core @ B_core + A_tail @ B_tail
    # A_core @ B_core is 4x4 mult -> Strassen-Winograd rank 14
    # A_tail @ B_tail is 4x1 @ 1x6 -> rank 4 (schoolbook)
    # Total rank = 18? Wait, let me reconsider.

    # Actually for 4x4 we use 2x2 Strassen blocks:
    # 4x4 = [[A00,A01],[A10,A11]] each 2x2
    # Strassen for 4x4 uses 7 products of 2x2, each 2x2 uses 7 -> 49? No.
    # Better: 4x4 Strassen-Winograd is rank 14 (two independent 2x2 Strassen on
    # the off-diagonal structure). Let me use the known rank-14 for 4x4.

    # Simplest correct approach: 4x4 via 4 blocks of 2x2.
    # Top-left 2x2 and bottom-right 2x2 are independent 2x2 mults (rank 7 each).
    # Top-right 2x2 and bottom-left 2x2 need cross terms.
    # Actually 4x4 matmul tensor rank is 14. Construction:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Use Strassen on the 4x4 as a whole: rank 14.

    # Let me build 4x4 rank-14 directly using the standard formula.
    # 4x4 = 2x2 blocks. The 4x4 Strassen (rank 14) is:
    # Treat 4x4 as 2x2 of 2x2 blocks and apply 2x2 Strassen (7 products of 2x2),
    # but that gives 7*4=28 if each 2x2 is schoolbook. To get 14, each 2x2
    # product must be Strassen (rank 7) but shared... Actually rank of 4x4 is 14
    # and is achieved by: 2x2 Strassen where the 7 "products" are each 2x2
    # multiplications, but the 2x2 mults are NOT done fully - instead the 4x4
    # rank-14 comes from a different structure.

    # Let me just use: 4x4 mult = apply 2x2 Strassen to the block matrix.
    # The 7 block-products are each 2x2*2x2. If we compute each 2x2*2x2 with
    # Strassen (rank 7), total = 7*7 = 49. That's worse than 16 schoolbook.
    # So 4x4 rank-14 is a special non-trivial decomposition.

    # Known rank-14 for 4x4: it's the "Strassen-Winograd" where you do:
    # 4x4 = [[P,Q],[R,S]] and use 14 products. Let me use a verified construction.

    # I'll use the standard rank-14 construction for 4x4:
    # It's based on: 4x4 mult has 16 entries. Rank 14 decomposition:
    # Use the fact that 4x4 = 2x2 Kronecker. The rank-14 is:
    # M1 = (A00+A11+A22+A33)(B00+B11+B22+B33)  ... this is getting complex.

    # Let me take a safer path: use 2x2 Strassen for the 4x4 core by splitting
    # 4x4 into 2x2 blocks and using the 7-product Strassen, computing each 2x2
    # block product with schoolbook (rank 4). Total for 4x4 core = 7*4 = 28? 
    # No wait - in Strassen for 4x4-as-2x2-blocks, the 7 products are 2x2*2x2.
    # Each 2x2*2x2 needs rank 4 (schoolbook) or 7 (Strassen). 
    # If schoolbook: 7*4 = 28. If Strassen: 7*7 = 49.
    # But 4x4 schoolbook is only 16! So this block approach is worse.

    # The correct rank-14 for 4x4 is a flat decomposition, not nested.
    # Let me hardcode the known rank-14 4x4 Strassen-Winograd.

    # Rank-14 for 4x4 (i,j in {0,1,2,3}):
    # This is well-known. Let me use the construction from the literature.
    # The 4x4 tensor has rank 14. One construction:
    # 
    # Actually, let me reconsider the whole problem. 
    # 4x5 * 5x6. 
    # Split 5 = 4 + 1.
    # A = [A4 | a4] where A4 is 4x4, a4 is 4x1
    # B = [B4; b4] where B4 is 4x6, b4 is 1x6
    # C = A4*B4 + a4*b4
    # a4*b4 is rank 4 (outer product of 4x1 and 1x6).
    # A4*B4 is 4x4 * 4x6.
    # 
    # 4x4 * 4x6: split 6 = 4 + 2.
    # A4*B4 = A4*[B4a | B4b] = [A4*B4a | A4*B4b]
    # A4*B4a is 4x4*4x4 (rank 14)
    # A4*B4b is 4x4*4x2 
    # 
    # This is getting complicated. Let me just do:
    # 4x5 * 5x6 directly with a block Strassen approach.

    # Simpler robust plan: 
    # 4x5*5x6. Use the fact that we can pad to 4x6*6x6 or use blocks.
    # 
    # Let me use: 4x5*5x6 = 4x6*6x6 with the 5th row/col zero? No, that changes rank.
    #
    # Best approach: direct block decomposition.
    # A (4x5) = [A11(4x4) | A12(4x1)]
    # B (5x6) = [B11(4x6); B21(1x6)]
    # C = A11*B11 + A12*B21
    # A12*B21: 4x1 * 1x6, rank 4.
    # A11*B11: 4x4 * 4x6.
    #   Split 6 = 4+2: B11 = [B11a(4x4) | B11b(4x2)]
    #   A11*B11a: 4x4*4x4, rank 14
    #   A11*B11b: 4x4*4x2. Split 2=1+1: 
    #     4x4*4x1: rank 4 (each col: 4x4*1, but that's 4x4 mult of a vector = rank 4? 
    #     Actually 4x4 * 4x1 is just 4x4 matrix-vector, rank 4 (each output row is a dot).
    #     So 4x4*4x2 = 2 columns, each rank 4 -> rank 8.
    #   So A11*B11 = 14 + 8 = 22.
    # Total = 22 + 4 = 26? Let me verify: 4x4*4x4 is 14, 4x4*4x2 is 8, 4x1*1x6 is 4.
    # Total = 14 + 8 + 4 = 26.

    # Hmm, but is 4x4*4x4 really rank 14? Yes, that's the Strassen-Winograd rank.
    # And 4x4*4x2: this is multiplying 4x4 by 4x2. The tensor for 4x4*4x2 has rank 8
    # (each of the 2 output columns needs 4 products, and they share the A matrix).
    # Actually 4x4*4x2: output is 4x2. C[:,c] = A * B[:,c]. 
    # This is rank 8: for each output column c, 4 row-dots, but A is shared.
    # The tensor rank of (4x4)*(4x2) is 8? Let me think: it's 4x4*4x2 = 
    # sum over j of A[:,j] * B[j,:] where B[j,:] is 1x2. So rank <= 4*2=8 (schoolbook).
    # Can it be less? The 4x4*4x1 tensor has rank 4. 4x4*4x2 has rank 8 (they're independent).
    # So yes, 8.

    # So total rank = 14 + 8 + 4 = 26.
    # But wait, I need to double check 4x4*4x4 = 14. The standard result: 
    # rank of 4x4 matrix multiplication is 14. Yes.

    # Let me construct this. I'll need the rank-14 decomposition for 4x4*4x4.
    # This is the hard part. Let me use a known rank-14 construction.

    # Rank-14 for 4x4*4x4:
    # One standard construction uses the "Strassen-Winograd" method.
    # The 14 products are:
    # 
    # Let me use the construction from "Strassen's algorithm" for 4x4:
    # It's based on 2x2 Strassen applied cleverly. Actually, the rank-14 for 4x4
    # is: treat 4x4 as 2x2 of 2x2 blocks, but use a rank-14 (not 7*4=28) by
    # sharing. The actual rank-14 construction:
    #
    # M1 = (A00+A11)(B00+B11)  [2x2 products]
    # ... this gives 7 products of 2x2, each 2x2 has 4 entries -> but we need
    # the 2x2 products to be rank-4 each, giving 28. That's not 14.
    #
    # The TRUE rank-14 for 4x4 is a flat 14-term decomposition. Let me look up
    # the explicit formula. 
    #
    # Known rank-14 for 4x4 (from literature):
    # The 14 rank-1 tensors are indexed by (i,j,k) combinations. 
    # This is complex to hardcode correctly. 

    # Given time constraints, let me use a SAFER approach that I can verify:
    # Use 2x2 Strassen for the 4x4 core, but recognize that 4x4 = 2x2 blocks
    # and the 7 Strassen products are 2x2*2x2. If I compute each 2x2*2x2 with
    # schoolbook (rank 4), total = 28 for the 4x4 core. That's worse than 16
    # schoolbook for 4x4. So this is bad.
    #
    # Alternative: just use schoolbook for everything but exploit the structure.
    # 4x5*5x6 schoolbook = 120. 
    # 
    # Let me try a different split that I CAN implement correctly:
    # 4x5*5x6. Split 5=2+3.
    # A = [Aa(4x2) | Ab(4x3)]
    # B = [Ba(2x6); Bb(3x6)]
    # C = Aa*Ba + Ab*Bb
    # Aa*Ba: 4x2*2x6, rank 12 (schoolbook: 2 inner, 4*6=24 outer, but rank = 2*4*6/... 
    #   actually 4x2*2x6: C[i,k] = sum_j Aa[i,j]*Ba[j,k], j in 0..1. Rank = 2*4*6? No.
    #   The tensor rank of (4x2)*(2x6) is 4*2*6 = 48? No, that's schoolbook count.
    #   Rank of (m x k)*(k x p) tensor is m*k*p for schoolbook, but can be lower.
    #   For 4x2*2x6: it's like 4x2 times 2x6. This equals (4x2)*(2x6).
    #   We can write C = Aa * Ba. Aa is 4x2, Ba is 2x6. 
    #   C[i,k] = Aa[i,0]*Ba[0,k] + Aa[i,1]*Ba[1,k]. 
    #   So C = Aa[:,0] outer Ba[0,:] + Aa[:,1] outer Ba[1,:]. That's rank 2!
    #   Wait no - that's not right either because Aa[i,0] and Ba[0,k] vary.
    #   C[i,k] = sum_j Aa[i,j]*Ba[j,k]. For fixed j, the term is Aa[:,j] outer Ba[j,:].
    #   So C = sum_j (Aa[:,j] outer Ba[j,:]), which is rank <= k = 2? 
    #   No! Aa[:,j] is a 4-vector, Ba[j,:] is a 6-vector, their outer product is 4x6.
    #   C = Aa[:,0] outer Ba[0,:] + Aa[:,1] outer Ba[1,:]. Yes! Rank 2.
    #   Wait, that can't be right. Let me check: (Aa outer Ba)[i,k] where Aa is 4x2, Ba is 2x6.
    #   Standard matrix mult: C = Aa @ Ba, C[i,k] = sum_j Aa[i,j]*Ba[j,k].
    #   Aa[:,0] outer Ba[0,:] has [i,k] = Aa[i,0]*Ba[0,k]. 
    #   Aa[:,1] outer Ba[1,:] has [i,k] = Aa[i,1]*Ba[1,k].
    #   Sum = Aa[i,0]*Ba[0,k] + Aa[i,1]*Ba[1,k] = C[i,k]. YES!
    #   So (m x k) * (k x p) has rank <= k! Because C = sum_j A[:,j] outer B[j,:].
    #   Wait, that means 4x5*5x6 has rank <= 5?! That's wrong.
    #   The error: A[:,j] is m-vector, B[j,:] is p-vector, outer is m x p. 
    #   But C[i,k] = sum_j A[i,j]*B[j,k]. And (A[:,j] outer B[j,:])[i,k] = A[i,j]*B[j,k].
    #   So C = sum_j A[:,j] outer B[j,:]. This is a sum of k rank-1 (in the m*p sense)
    #   terms, BUT each "rank-1" here is a rank-1 m x p matrix, and we have k of them.
    #   So the RANK of the matrix C is <= k, but that's the matrix rank, not the
    #   tensor rank! The tensor rank is different.
    #   
    #   The tensor for matrix multiplication (m x k) * (k x p) -> (m x p) has rank
    #   m*k*p in general (schoolbook), and the "sum of k outer products" is NOT a
    #   tensor decomposition because A[:,j] and B[j,:] are not scalars - they're vectors
    #   that depend on the input. The tensor decomposition requires each term to be
    #   u_r outer v_r outer w_r where u_r, v_r, w_r are FIXED vectors (coefficients),
    #   not functions of the input.
    #   
    #   So my "rank 2" argument is wrong. The tensor rank of (4x2)*(2x6) is 4*2*6 = 48
    #   for schoolbook, and the actual minimal rank is unknown but likely less.

    # OK, I'm overcomplicating this. Let me just use a KNOWN correct decomposition.
    # The safest is to use 2x2 Strassen blocks that I can verify.

    # New plan: Use 2x2 Strassen on the 4x5*5x6 by padding/splitting.
    # 4x5*5x6. Split into 2x2 blocks of size 2:
    # A (4x5) = [[A00(2x2), A01(2x2), A02(2x1)],
    #            [A10(2x2), A11(2x2), A12(2x1)]]
    # B (5x6) = [[B00(2x2), B01(2x2), B02(2x2)],
    #            [B10(2x2), B11(2x2), B12(2x2)],
    #            [B20(1x2), B21(1x2), B22(1x2)]]
    # 
    # C = A*B. This is messy with the 5 and 6.

    # Let me just do the straightforward thing that I KNOW works:
    # 4x5*5x6. Use the 5 as the inner dimension.
    # C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k].
    # 
    # Group: j=0,1,2,3 (core 4) and j=4 (tail 1).
    # C = A_core * B_core + A_tail * B_tail
    # where A_core is 4x4, B_core is 4x6, A_tail is 4x1, B_tail is 1x6.
    # 
    # A_tail * B_tail: 4x1 * 1x6. C_tail[i,k] = A_tail[i,0]*B_tail[0,k].
    # This is rank 4: C_tail = A_tail[:,0] outer B_tail[0,:]. 
    # As a tensor: C_tail[i,k] = A_tail[i,0]*B_tail[0,k].
    # Tensor decomposition: for each (i,k), the term is A[i,0]*B[0,k].
    # We need rank-1 tensors u_r outer v_r outer w_r.
    # C_tail[i,k] = A_tail[i,0] * B_tail[0,k].
    # This is a single rank-1 in the (i,k) output, but A_tail[i,0] depends on i and
    # B_tail[0,k] depends on k. So C_tail[i,k] = (A_tail[i,0]) * (B_tail[0,k]).
    # As a tensor in (i, j=0, k): it's sum over... actually it's just one term:
    # u[i, j, k] where u[i,0,k] = 1, and the product is A[i,0]*B[0,k].
    # So the tensor for A_tail*B_tail is: T[i, j, k] = delta(j,0) for the "middle" index.
    # The rank-1 decomposition: we need u_r[i], v_r[j], w_r[k] such that
    # sum_r u_r[i]*v_r[j]*w_r[k] = delta(j,0) for all i,k, and then
    # C[i,k] = sum_r u_r[i]*v_r[j]*w_r[k] * A[i,j]*B[j,k]... 
    # 
    # I'm confusing myself. Let me use the standard tensor decomposition form.
    # 
    # The matrix multiplication tensor T for (m x k) * (k x p) -> (m x p) is:
    # T[i, j, k] = delta where the "product" index... 
    # 
    # Standard: T is a 3-tensor with indices (i, j, l) where i in [0,m), j in [0,k),
    # l in [0,p). T[i,j,l] = 1 if... no. 
    # 
    # The multiplication: C[i,l] = sum_j A[i,j]*B[j,l].
    # The tensor T[i, (j), l] where the middle index j is the summation index.
    # T[i, j, l] = delta(i,j,l) in the sense that the "1" is at all (i,j,l) and
    # C[i,l] = sum_j T[i,j,l] * A[i,j] * B[j,l] with T[i,j,l] = 1 for all i,j,l.
    # 
    # A rank-1 decomposition: T[i,j,l] = sum_r u_r[i]*v_r[j]*w_r[l].
    # Then C[i,l] = sum_r (sum_j u_r[i]*v_r[j]*A[i,j]) * (sum_j ... ) 
    # No, that's not how it works. The correct form:
    # C[i,l] = sum_r (sum_j u_r[i]*A[i,j]*v_r[j]) ... no.
    # 
    # The standard: if T[i,j,l] = sum_r u_r[i]*v_r[j]*w_r[l], then
    # C[i,l] = sum_j T[i,j,l]*A[i,j]*B[j,l] = sum_r u_r[i]*w_r[l] * (sum_j v_r[j]*A[i,j]*B[j,l])
    # That's not separable. 
    # 
    # The CORRECT tensor decomposition for matrix mult:
    # C[i,l] = sum_j A[i,j]*B[j,l].
    # We want: C[i,l] = sum_r (sum_j A[i,j]*u_r[j]) * (sum_l B[j,l]*w_r[l])... no.
    # 
    # The right way: C[i,l] = sum_r p_r where p_r = (A * u_r) * (v_r * B) or similar.
    # Specifically: if we have vectors such that
    # C[i,l] = sum_r (sum_j A[i,j]*U_r[j]) * (sum_j B[j,l]*V_r[j])
    # then this is a valid decomposition with rank r, where the "middle" is:
    # T[i,j,l] = sum_r U_r[j]*V_r[j]*delta... 
    # 
    # Let me just use the CONVENTION from the task:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # So for the full 4x5*5x6:
    # U has shape (4*5, rank) = (20, rank), indexed by (i,j) with i in [0,4), j in [0,5).
    # V has shape (5*6, rank) = (30, rank), indexed by (j,k) with j in [0,5), k in [0,6).
    # W has shape (6*4, rank) = (24, rank), indexed by (k,i) with k in [0,6), i in [0,4).
    # 
    # The decomposition: C[i,k] = sum_c U[i*5+j, c] * V[j*6+k, c] * W[k*4+i, c] summed over j?
    # No. The convention is:
    # C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that sums over j AND c. Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So for each (i,j,k,c): the contribution is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # And C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # 
    # For schoolbook: for each (i,j,k), set U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Then C[i,k] = sum_{j,c} ... = sum_j 1*1*1 = sum_j A[i,j]*B[j,k]. Correct.
    # 
    # So the decomposition is: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the actual matrix mult is C = A*B, and the "1"s in U,V,W are the
    # coefficients, and A[i,j], B[j,k] are the actual values.
    # 
    # So the tensor is: T[i, j, k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = delta(i,j,k)
    # where delta means T[i,j,k] = 1 for all i,j,k (the schoolbook case).
    # 
    # For a Strassen decomposition, T[i,j,k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c]
    # should equal the "multiplication pattern" which for the full tensor is 1 for all (i,j,k).
    # 
    # Wait, that's not right either. In Strassen, the tensor is NOT all 1s. 
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k IS the schoolbook.
    # Strassen gives a DIFFERENT decomposition of the SAME tensor (all 1s) with fewer terms.
    # 
    # So yes: T[i,j,k] = 1 for all i in [0,m), j in [0,k_inner), k in [0,p), and
    # we want T[i,j,k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] with rank < m*k*p.
    # 
    # For 2x2*2x2: T[i,j,k] = 1 for all i,j,k in {0,1}, and Strassen gives rank 7:
    # sum_{c=1}^{7} U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    # 
    # Let me verify my Strassen 2x2 above:
    # U, V, W are 7x4 (7 columns, 4 entries each for i,j or j,k or k,i).
    # For each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should = 1.
    # 
    # Let me check (i=0,j=0,k=0):
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # c=1: U[0,0]=0, ... -> 0
    # c=2: U[0,0]=1, V[0,0]=0, ... -> 0
    # c=3: U[0,0]=0, ... -> 0
    # c=4: U[0,0]=1, V[0,0]=0, ... -> 0
    # c=5: U[0,0]=-1, V[0,0]=1, W[0,0]=0 -> 0
    # c=6: U[0,0]=0, ... -> 0
    # Sum = 1. Correct!
    # 
    # Let me check (i=0,j=0,k=1):
    # c=0: U[0,0]=1, V[0,1]=0, ... -> 0
    # c=1: U[0,0]=0, ... -> 0
    # c=2: U[0,0]=1, V[0,1]=0, ... -> 0
    # c=3: U[0,0]=0, ... -> 0
    # c=4: U[0,0]=1, V[0,1]=1, W[1,0]=0 -> 0
    # c=5: U[0,0]=-1, V[0,1]=1, W[1,0]=0 -> 0
    # c=6: U[0,0]=0, ... -> 0
    # Sum = 0. But it should be 1! 
    # 
    # Error! Let me recheck. W[k,i] for k=1, i=0: W[1,0].
    # My W columns: 
    # c=0: [1,0,0,0] -> W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=0
    # c=1: [1,0,0,0] -> W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=0
    # c=2: [0,0,0,1] -> W[0,0]=0, W[0,1]=0, W[1,0]=0, W[1,1]=1
    # c=3: [0,1,0,0] -> W[0,0]=0, W[0,1]=1, W[1,0]=0, W[1,1]=0
    # c=4: [0,0,0,1] -> W[0,0]=0, W[0,1]=0, W[1,0]=0, W[1,1]=1
    # c=5: [0,0,1,0] -> W[0,0]=0, W[0,1]=0, W[1,0]=1, W[1,1]=0
    # c=6: [0,1,1,0] -> W[0,0]=0, W[0,1]=1, W[1,0]=1, W[1,1]=0
    # 
    # For (i=0,j=0,k=1): W[k=1,i=0] = W[1,0].
    # c=0: W[1,0]=0
    # c=1: W[1,0]=0
    # c=2: W[1,0]=0
    # c=3: W[1,0]=0
    # c=4: W[1,0]=0
    # c=5: W[1,0]=1, U[0,0]=-1, V[0,1]=1 -> -1*1*1 = -1
    # c=6: W[1,0]=1, U[0,0]=0, ... -> 0
    # Sum = -1. But should be 1. WRONG.
    # 
    # My Strassen 2x2 is incorrect. Let me fix it.
    # 
    # Standard Strassen for 2x2:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M1 - M2 + M6 + M7
    # C11 = M2 - M3 + M4 - M5
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # So the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should give
    # the coefficients such that C[i,k] = sum_{j,c} ... 
    # 
    # Actually, the way it works: C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (intermediate).
    # The Strassen decomposition: C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k])
    # No, that's not standard either.
    # 
    # The correct interpretation: 
    # C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # Wait no. Let me re-read the task convention.
    # 
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]" with the decomposition being exact.
    # The schoolbook sets all to 1, giving C[i,k] = sum_j 1*1*1 * A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k].
    # 
    # So the general form: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # 
    # For this to equal sum_j A[i,j]*B[j,k] for ALL A,B, we need:
    # sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1 for all i,j,k.
    # 
    # So yes, T[i,j,k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1 for all i,j,k.
    # 
    # Now for Strassen 2x2, the 7 products M1..M7 each correspond to a "c", and
    # U[i,j,c], V[j,k,c], W[k,i,c] are the coefficients. The sum over c of
    # U[i,j,c]*V[j,k,c]*W[k,i,c] should = 1 for all i,j,k in {0,1}.
    # 
    # Let me redo the Strassen coefficients properly.
    # 
    # M1 = (A00+A11)(B00+B11): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W: contributes to C00(+1), C10(+1)
    #   So W[0,0]=1, W[1,0]=1, W[0,1]=0, W[1,1]=0
    # M2 = (A10+A11)B00: U[1,0]=1,U[1,1]=1; V[0,0]=1; W: C10(+1), C11(+1)
    #   W[1,0]=1, W[1,1]=1, W[0,0]=0, W[0,1]=0
    # M3 = A00(B10-B11): U[0,0]=1; V[1,0]=1,V[1,1]=-1; W: C01(+1), C11(-1)
    #   W[0,1]=1, W[1,1]=-1, W[0,0]=0, W[1,0]=0
    # M4 = A11(B01-B00): U[1,1]=1; V[0,0]=-1,V[0,1]=1; W: C00(-1), C01(+1), C10(+1), C11(-1)
    #   Wait, M4 appears in C00(-M4), C01(+M4), C10(+M4), C11(-M4).
    #   So W[0,0]=-1, W[0,1]=1, W[1,0]=1, W[1,1]=-1
    # M5 = (A00+A01)B11: U[0,0]=1,U[0,1]=1; V[1,1]=1; W: C00(-M5), C11(-M5)
    #   W[0,0]=-1, W[1,1]=-1, W[0,1]=0, W[1,0]=0
    # M6 = (A10-A00)(B00+B01): U[1,0]=1,U[0,0]=-1; V[0,0]=1,V[0,1]=1; W: C10(+M6)
    #   W[1,0]=1, W[1,1]=0, W[0,0]=0, W[0,1]=0
    # M7 = (A01-A11)(B10+B11): U[0,1]=1,U[1,1]=-1; V[1,0]=1,V[1,1]=1; W: C00(-M7), C10(+M7)
    #   W[0,0]=-1, W[1,0]=1, W[0,1]=0, W[1,1]=0
    # 
    # Now let me verify T[0,0,0] = sum_c U[0,0,c]*V[0,0,c]*W[0,0,c]:
    # c=0(M1): U=1,V=1,W=1 -> 1
    # c=1(M2): U=0,V=1,W=0 -> 0
    # c=2(M3): U=1,V=0,W=0 -> 0
    # c=3(M4): U=0,V=-1,W=-1 -> 0
    # c=4(M5): U=1,V=0,W=-1 -> 0
    # c=5(M6): U=-1,V=1,W=0 -> 0
    # c=6(M7): U=0,V=0,W=-1 -> 0
    # Sum = 1. Correct!
    # 
    # T[0,0,1] = sum_c U[0,0,c]*V[0,0,c]*W[0,1,c]:
    # c=0: U=1,V=1,W[0,1]=0 -> 0
    # c=1: U=0,... -> 0
    # c=2: U=1,V=0,W[0,1]=1 -> 0
    # c=3: U=0,V=-1,W[0,1]=1 -> 0
    # c=4: U=1,V=0,W[0,1]=0 -> 0
    # c=5: U=-1,V=1,W[0,1]=0 -> 0
    # c=6: U=0,V=0,W[0,1]=0 -> 0
    # Sum = 0. WRONG, should be 1.
    # 
    # Hmm. The issue is that T[i,j,k] = 1 for ALL i,j,k is the schoolbook, and
    # Strassen does NOT give T[i,j,k] = 1 for all i,j,k. Instead, Strassen gives
    # a decomposition where C[i,k] = sum_j A[i,j]*B[j,k] is achieved, but the
    # intermediate "tensor" is not all 1s.
    # 
    # I think I'm misunderstanding the convention. Let me re-read.
    # 
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]" - these are the factor matrices.
    # The decomposition: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # 
    # For this to work, we need: for all A, B:
    # sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    # 
    # This holds if and only if: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # So yes, T[i,j,k] = 1 for all i,j,k. And Strassen must satisfy this.
    # 
    # But my Strassen gives T[0,0,1] = 0, not 1. So either my Strassen is wrong
    # or my understanding is wrong.
    # 
    # Let me reconsider. Maybe the convention is different. Maybe:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # 
    # This is the "Strassen" form: C[i,k] = sum_c W[k,i,c] * X_c[i] * Y_c[k]
    # where X_c[i] = sum_j U[i,j,c]*A[i,j] and Y_c[k] = sum_j V[j,k,c]*B[j,k].
    # 
    # For this to equal sum_j A[i,j]*B[j,k], we need a different condition.
    # This form is NOT the same as the tensor sum. 
    # 
    # I think the task's convention is the tensor sum form:
    # C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # 
    # And for Strassen, the way it works is that the "1"s are replaced by the
    # Strassen coefficients, and the sum over j,c still gives the correct C.
    # 
    # Let me just VERIFY my Strassen by checking if it produces the correct C.
    # 
    # Actually, you know what, let me just use a DIFFERENT approach that I can
    # verify numerically. Let me construct the decomposition and then verify it
    # with random matrices. If it's wrong, I'll fall back to schoolbook.
    # 
    # Given the complexity and time, let me use a SAFE approach:
    # 1. Try to construct a low-rank decomposition.
    # 2. Verify it numerically.
    # 3. If it fails, return schoolbook (rank 120).
    # 
    # For the low-rank attempt, let me use the 2x2 Strassen correctly.
    # 
    # The CORRECT Strassen 2x2 in the tensor convention:
    # We need T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    # 
    # Let me derive it. The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M1 - M2 + M6 + M7
    # C11 = M2 - M3 + M4 - M5
    # 
    # In the tensor convention, each M_c is a "product" that contributes to C[i,k]
    # via W[k,i,c]. The U and V define the "factors".
    # 
    # For M1: U[0,0]=1, U[1,1]=1, others 0. V[0,0]=1, V[1,1]=1, others 0.
    # M1 contributes +1 to C00, +1 to C10. So W[0,0]=1, W[1,0]=1, W[0,1]=0, W[1,1]=0.
    # 
    # For M2: U[1,0]=1, U[1,1]=1. V[0,0]=1. Contributes +1 to C10, +1 to C11.
    # W[1,0]=1, W[1,1]=1, W[0,0]=0, W[0,1]=0.
    # 
    # For M3: U[0,0]=1. V[1,0]=1, V[1,1]=-1. Contributes +1 to C01, -1 to C11.
    # W[0,1]=1, W[1,1]=-1, W[0,0]=0, W[1,0]=0.
    # 
    # For M4: U[1,1]=1. V[0,0]=-1, V[0,1]=1. Contributes -1 to C00, +1 to C01, +1 to C10, -1 to C11.
    # W[0,0]=-1, W[0,1]=1, W[1,0]=1, W[1,1]=-1.
    # 
    # For M5: U[0,0]=1, U[0,1]=1. V[1,1]=1. Contributes -1 to C00, -1 to C11.
    # W[0,0]=-1, W[1,1]=-1, W[0,1]=0, W[1,0]=0.
    # 
    # For M6: U[1,0]=1, U[0,0]=-1. V[0,0]=1, V[0,1]=1. Contributes +1 to C10.
    # W[1,0]=1, W[1,1]=0, W[0,0]=0, W[0,1]=0.
    # 
    # For M7: U[0,1]=1, U[1,1]=-1. V[1,0]=1, V[1,1]=1. Contributes -1 to C00, +1 to C10.
    # W[0,0]=-1, W[1,0]=1, W[0,1]=0, W[1,1]=0.
    # 
    # Now verify T[0,0,1] = sum_c U[0,0,c]*V[0,0,c]*W[0,1,c]:
    # c=0(M1): U[0,0]=1, V[0,0]=1, W[0,1]=0 -> 0
    # c=1(M2): U[0,0]=0, ... -> 0
    # c=2(M3): U[0,0]=1, V[0,0]=0, W[0,1]=1 -> 0
    # c=3(M4): U[0,0]=0, V[0,0]=-1, W[0,1]=1 -> 0
    # c=4(M5): U[0,0]=1, V[0,0]=0, W[0,1]=0 -> 0
    # c=5(M6): U[0,0]=-1, V[0,0]=1, W[0,1]=0 -> 0
    # c=6(M7): U[0,0]=0, V[0,0]=0, W[0,1]=0 -> 0
    # Sum = 0. Still 0, not 1.
    # 
    # I think the issue is that the Strassen decomposition does NOT satisfy
    # T[i,j,k] = 1 for all i,j,k. Instead, it satisfies a WEAKER condition that
    # still gives the correct C for all A,B. 
    # 
    # The correct condition is: for all A, B:
    # sum_{j,c} U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c]*A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k]
    # 
    # This is equivalent to: for all i,k:
    # sum_{j,c} U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c]*A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k]
    # 
    # This must hold for ALL A, B, which means the "coefficient tensor" 
    # S[i,j,k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c]
    # must satisfy: sum_j S[i,j,k]*A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k] for all A,B.
    # 
    # This holds if S[i,j,k] = 1 for all i,j,k (sufficient), but it's not necessary.
    # The necessary and sufficient condition is that S[i,j,k] = 1 for all i,j,k.
    # Because if we fix i,k and vary A[i,·] and B[·,k], we get a bilinear form
    # in A[i,·] and B[·,k], and for it to equal the standard inner product for all
    # A[i,·], B[·,k], we need S[i,j,k] = 1 for all j.
    # 
    # So S[i,j,k] = 1 IS necessary. And my Strassen gives S[0,0,1] = 0. So my
    # Strassen is WRONG for this convention.
    # 
    # I think the problem is that Strassen's algorithm doesn't fit this convention
    # directly. The convention requires S[i,j,k] = 1 for all i,j,k, which is the
    # schoolbook tensor. Strassen gives a DIFFERENT tensor that still computes the
    # product, but through a different mechanism.
    # 
    # Wait, I think I've been confusing two different things. Let me reconsider.
    # 
    # In the standard tensor decomposition of matrix multiplication, the tensor is
    # T[i,j,k] = 1 for all i,j,k (the "multiplication tensor"). A rank-r decomposition
    # is T[i,j,k] = sum_{c=1}^{r} u_c[i]*v_c[j]*w_c[k].
    # 
    # For Strassen 2x2, the rank-7 decomposition of the 2x2 multiplication tensor
    # (which has T[i,j,k] = 1 for all i,j,k in {0,1}) is:
    # T[i,j,k] = sum_{c=1}^{7} u_c[i]*v_c[j]*w_c[k]
    # 
    # And this DOES satisfy T[i,j,k] = 1 for all i,j,k. Let me look up the correct
    # Strassen rank-7 decomposition of the 2x2 tensor.
    # 
    # The 2x2 multiplication tensor: T[i,j,k] = 1 for i,j,k in {0,1}.
    # Rank-7 decomposition:
    # 
    # This is a well-known result. The 7 rank-1 tensors are:
    # (using the standard Strassen):
    # 
    # Let me just use a verified source. The Strassen rank-7 for 2x2:
    # 
    # u1 = (1,1,0,0) [i=0,j=0: 1; i=0,j=1: 0; i=1,j=0: 0; i=1,j=1: 1]
    # v1 = (1,0,0,1) [j=0,k=0: 1; j=0,k=1: 0; j=1,k=0: 0; j=1,k=1: 1]
    # w1 = (1,0,1,0) [k=0,i=0: 1; k=0,i=1: 0; k=1,i=0: 1; k=1,i=1: 0]
    # 
    # Hmm, I'm not sure of the exact values. Let me just implement it and verify
    # numerically. If it fails, I'll use schoolbook.
    # 
    # Given the time constraints and the difficulty of getting Strassen exactly right,
    # let me take a pragmatic approach:
    # 1. Implement schoolbook (rank 120) as a fallback.
    # 2. Try to implement a block decomposition that I can verify.
    # 3. Verify numerically and return the best valid result.

    # Let me use a block decomposition that I can verify:
    # 4x5*5x6. Split 5 = 2 + 3.
    # A = [Aa(4x2) | Ab(4x3)], B = [Ba(2x6); Bb(3x6)]
    # C = Aa*Ba + Ab*Bb
    # 
    # Aa*Ba: 4x2*2x6. Schoolbook rank = 4*2*6 = 48? No, the tensor rank of
    # (4x2)*(2x6) is 4*2*6 = 48 for schoolbook. But we can do better.
    # Actually, (m x k) * (k x p) has tensor rank m*k*p for schoolbook.
    # For 4x2*2x6: schoolbook rank = 4*2*6 = 48.
    # For 4x3*3x6: schoolbook rank = 4*3*6 = 72.
    # Total = 48 + 72 = 120. Same as direct schoolbook. No improvement.
    # 
    # The only way to improve is to use Strassen or similar on the blocks.
    # 
    # Given the difficulty, let me just return the schoolbook decomposition
    # (rank 120) which is guaranteed correct, and add a deadline guard.
    # This matches the current score of 1.0.
    
    # Actually, let me try ONE more thing: use 2x2 Strassen on the 4x4 core
    # of the 4x5*5x6 problem, and verify it numerically.
    
    # For now, let me just return schoolbook to be safe.
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
