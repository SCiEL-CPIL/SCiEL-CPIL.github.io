"""Matrix multiplication 3x4 x 4x4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 118.0

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p  # 48
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def build_strassen_2x2():
        """7 Strassen terms: (u_dict, v_dict, w_dict).
        Keys: u=(i,j), v=(j,k), w=(k,i)."""
        T = []
        # P1 = (a00+a11)(b00+b11)
        T.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # P2 = (a10+a11)b00
        T.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # P3 = a00(b01-b11)
        T.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # P4 = a11(b10-b00)
        T.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # P5 = (a00+a01)b11
        T.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # P6 = (a10-a00)(b00+b01)
        T.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # P7 = (a01-a11)(b10+b11)
        T.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return T

    strassen = build_strassen_2x2()

    def verify(U, V, W, rank):
        """Verify the decomposition is exact. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        Cc = np.zeros((n, p))
        for r in range(rank):
            ar = Ua @ (A.flatten()).reshape(n*m) if False else \
                 sum(Ua[i*m+j, r] * A[i, j] for i in range(n) for j in range(m))
            br = sum(Va[j*p+k, r] * B[j, k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    Cc[i, k] += Wa[k*n + i, r] * ar * br
        return float(np.max(np.abs(Cc - C)))

    # ============================
    # Strategy A: current 44-rank decomposition
    # ============================
    rank_a = 44
    U_A = [[0] * rank_a for _ in range(n * m)]
    V_A = [[0] * rank_a for _ in range(m * p)]
    W_A = [[0] * rank_a for _ in range(p * n)]

    def add_s(a_off, b_co, j_off, c_co, c_ro, cs, block=U_A, V=V_A, W=W_A):
        for r in range(7):
            ud, vd, wd = strassen[r]
            c = cs + r
            for (ai, aj), v in ud.items():
                block[(a_off+ai)*m + (j_off+aj)][c] += v
            for (bj, bk), v in vd.items():
                V[(j_off+bj)*p + (b_co+bk)][c] += v
            for (ck, ci), v in wd.items():
                W[(c_co+ck)*n + (c_ro+ci)][c] += v
        return cs + 7

    def add_4(a_row, b_co, j_off, c_co, cs, block=U_A, V=V_A, W=W_A):
        for r in range(4):
            c = cs + r
            jj = j_off + r % 2
            kk = b_co + r // 2
            block[a_row*m + jj][c] = 1
            V[jj*p + kk][c] = 1
            W[kk*n + a_row][c] = 1
        return cs + 4

    c = 0
    c = add_s(0, 0, 0, 0, 0, c)
    c = add_s(0, 0, 2, 0, 0, c)
    c = add_s(0, 2, 0, 0, 2, c)
    c = add_s(0, 2, 2, 0, 2, c)
    c = add_4(2, 0, 0, 0, c)
    c = add_4(2, 0, 2, 0, c)
    c = add_4(2, 2, 0, 2, c)
    c = add_4(2, 2, 2, 2, c)
    assert c == rank_a

    err_a = verify(U_A, V_A, W_A, rank_a)

    # ============================
    # Strategy B: 2+1 split with j-interleaving
    # A[0:2,0:2]·B[0:2,:] → 7,  A[0:2,2:4]·B[2:4,:] → 7
    # A[2:3,:]·B[:,:2] → 4,    A[2:3,:]·B[:,2:4] → 4  (total 22)
    # ============================
    # But can we do better? The 2×4×4 block: C_2×4 = A_2×4 · B_4×4
    # Split j: C[:2,0:2] += A[:2,0:2]·B[0:2,0:2] (rank 7)
    #          C[:2,2:4] += A[:2,2:4]·B[2:4,2:4] (rank 7)
    # The 1×4×2 block: C[2:3,0:2] = A[2:3,0:2]·B[0:2,0:2] (rank 4)
    # The 1×4×2 block: C[2:3,2:4] = A[2:3,2:4]·B[2:4,2:4] (rank 4)
    # Wait, that's wrong. Let me redo.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0:2, 0:2] = sum_j A[0:2,j] * B[j, 0:2]
    #   = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # This is what the current code does.
    #
    # Alternative split:
    # C[0:2, :] = A[0:2,:] * B[:, 0:2] (for k=0,1) + A[0:2,:] * B[:, 2:4] (for k=2,3)
    # No, let me think differently.
    #
    # Split C by output rows:
    # C[0:2, 0:4] = A[0:2,0:4] * B[0:4,0:4]  (2x4 x 4x4)
    #   Use Strassen on j-pairs and k-pairs: 4 blocks of 2x2*2x2, each rank 7 → 28
    # C[2:3, 0:4] = A[2:3,0:4] * B[0:4,0:4]  (1x4 x 4x4)
    #   4 blocks of 1x2*2x2, each rank 4 → 16
    # Total: 44. Same as current.
    #
    # What about a smarter split?
    # C[0:2, 0:4] can be done with:
    #   A[0:2,0:2]*B[0:2,0:4] + A[0:2,2:4]*B[2:4,0:4]
    #   (2x2*2x4) + (2x2*2x4)
    #   Each 2x2*2x4: split k into 2x2: 2x2*2x2 + 2x2*2x2 = 7+7=14
    #   Total for 2x2*2x4: 14. Two of these: 28.
    #   C[2:3,:]: 1x4*4x4: split j and k: 4 blocks of 1x2*2x2, each rank 4 → 16
    # Total: 44 again.
    #
    # What if we try 2+1 on j for the top part but interleave k?
    # C[0:2, 0:2] gets contributions from A[0:2,j]*B[j,0:2] for j=0..3
    # If we group j=0,1 and j=2,3 separately with Strassen: 14 total for C[0:2,0:2]
    # Similarly C[0:2,2:4]: 14. C[2:3,0:2]: 4. C[2:3,2:4]: 4. Total: 36.
    #
    # Wait! That's 36, not 44! Let me recheck the current code.
    # Current: 4 rank-7 + 4 rank-4 = 28+16 = 44.
    # My new calculation: C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    #   = 7 + 7 = 14
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    #   = 7 + 7 = 14
    # C[2:3,0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    #   = 4 + 4 = 8  (NOT 4! Each 1x2*2x2 is rank 4)
    # C[2:3,2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
    #   = 4 + 4 = 8
    # Total: 14+14+8+8 = 44.
    #
    # So the current code is correct. The key question is:
    # Can we do better than rank 4 for 1x2*2x2 -> 1x2?
    #
    # The tensor is T_{i,j,k} = delta_{j,k} for i=0 only (1x2*2x2).
    # Actually no: C[0,k] = sum_j A[j] * B[j,k]
    # So it's the standard matrix multiplication tensor for 1x2 * 2x2.
    #
    # R(1,2,2): The minimum rank. Since C has 2 elements and each A·B pair gives one,
    # we need at least 2 rank-1 terms (for the two k values). But we also need to
    # sum over j. Each rank-1 term fixes one (j,k) pair.
    # Actually, the tensor is T(j,k) = delta_{j,k}... no.
    # T_{j,k} is the coefficient of A[j]*B[j,k] in C[k].
    # So T_{j,k} = 1 if we're looking at C[k], 0 otherwise. But the tensor is
    # defined as T_{i,j,k} where i indexes C. For i=0 (only one row):
    # T_{0,j,k} = 1 for all j,k? No...
    #
    # C[k] = sum_j A[j] B[j,k]
    # The tensor T_{i,j,k} has i=0 only, and T_{0,j,k} = 1 for all j,k.
    # Wait no. The decomposition is C[k] = sum_r U[j,r] A[j] V[j,k,r] B[j,k] W[k,r].
    # For the 1x2*2x2 block: C[k] = sum_j A[j] B[j,k], j=0,1, k=0,1.
    # T_{j,k} = 1 for all (j,k) in {(0,0),(0,1),(1,0),(1,1)}.
    #
    # This is just the 4-element tensor that equals the outer product of ones.
    # Rank of T = 1 (it's rank-1: T_{j,k} = 1·1).
    # Wait, that can't be right. Let me reconsider.
    #
    # In the matrix multiplication tensor M_{i,j,k}, T_{i,j,k} = 1 if the
    # decomposition C[i,k] = sum_r a_r b_r w[i,k,r] works.
    # For 1x2*2x2: C[k] = sum_j A[j] B[j,k].
    # The tensor M has M[j,k] = 1 for all j,k (since C[k] = sum_j A[j]B[j,k]).
    # But the rank of M as a tensor... M is a 2x2 tensor (not a 3-tensor since
    # i is fixed). The rank of the 2x2 all-ones matrix is 1.
    #
    # But wait, we need to decompose this as a sum of rank-1 terms where each
    # term is U[j,r] · V[j,k,r] · W[k,r]. With only one i index (i=0):
    # sum_r U[j,r] V[j,k,r] W[k,r] = 1 for all j,k.
    #
    # If rank 1: U[j] V[j,k] W[k] = 1 for all j,k.
    # V[j,k] = 1/(U[j] W[k]). We need V[j,k] to be the "B coefficient" for
    # the same (j,k) in all terms.
    #
    # Actually, the standard result: R(1,n,1) = n (multiply row vector by column
    # vector). R(n,1,m) = m. R(1,1,m) = m.
    #
    # R(1,2,2) is less clear. Let me think again.
    # The bilinear map: (A in F^2, B in F^{2x2}) -> C in F^2, where C = A·B.
    # This is: C[k] = A[0]B[0,k] + A[1]B[1,k].
    # The multiplication tensor has entries T_{j,k} = 1 (4 entries, all 1).
    # As a bilinear form on (F^2, F^4) -> F^2.
    #
    # Rank: can we do it in 2 terms?
    # C[0] = (a·u1)(b·v1) + (a·u2)(b·v2)
    # C[1] = (a·u1)(b·v1) w1 + (a·u2)(b·v2) w2
    # where b·vr means sum of b[0,k]*vr[0,k] + b[1,k]*vr[1,k]
    #
    # We need:
    # C[0] = A[0]B[0,0] + A[1]B[1,0]
    # C[1] = A[0]B[0,1] + A[1]B[1,1]
    #
    # With 2 terms:
    # (A·u1)(B·v1) gives: u1[0]v1[0,0] A[0]B[0,0] + u1[0]v1[1,0] A[0]B[1,0]
    #                     + u1[1]v1[0,0] A[1]B[0,0] + u1[1]v1[1,0] A[1]B[1,0]
    #
    # This is getting complicated. The known bound is R(1,n,m) >= max(n,m)
    # and R(1,n,m) <= nm. For n=m=2: R >= 2, R <= 4.
    #
    # Can R(1,2,2) = 2? Try:
    # Term 1: U=[1,0], V=[1,0;0,0] (i.e., V[0,0]=1, others 0), W=[w10, w11]
    #   Contribution to C[k]: w1k * A[0] * B[0,0]
    # Term 2: U=[0,1], V=[0,0;0,1], W=[w20, w21]
    #   Contribution to C[k]: w2k * A[1] * B[1,1]
    # Total: C[0] = w10*A[0]*B[0,0] + w20*A[1]*B[1,1]
    #         C[1] = w11*A[0]*B[0,0] + w21*A[1]*B[1,1]
    # But we need C[0] = A[0]B[0,0] + A[1]B[1,0] (note B[1,0], not B[1,1]!)
    # So this doesn't work.
    #
    # Try rank 3:
    # We need to produce A[0]B[0,0], A[1]B[1,0] in C[0] and
    # A[0]B[0,1], A[1]B[1,1] in C[1].
    # 4 monomials, 2 outputs. With 3 terms, each term contributes to both outputs.
    # 3 terms → at most 3*1=3 distinct (U·A, V·B) products.
    # But we need 4 distinct monomials. So rank 3 seems insufficient.
    #
    # Actually, a term (U·A)(V·B) can produce at most 1*1=1 "type" of monomial
    # if U and V each pick one element. But U and V can be multi-component.
    #
    # I think R(1,2,2) = 4. So rank 4 for 1x2*2x2 is optimal.
    #
    # Similarly, what about R(2,2,2) = 7 (Strassen, optimal).
    #
    # So the question becomes: can we do the full 3x4*4x4 in fewer than 44?
    #
    # Lower bound: R(n,m,p) >= n*m*p / ... various bounds.
    # Simple: R >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16.
    # Bilander bound: R >= (n*m*p)^{2/3} / (something) ... not very tight.
    #
    # For 3x4*4x4, let's try to find a decomposition with rank < 44.
    #
    # What if we use a different split? Instead of 2+1 for rows, use
    # all 3 rows together with a different structure?
    #
    # C[0:3, 0:4] = A[0:3, 0:4] * B[0:4, 0:4]
    #
    # Split j into groups: {0,1} and {2,3}
    # C = A[:,0:2]*B[0:2,:] + A[:,2:4]*B[2:4,:]
    # Each term is 3x2 * 2x4 -> 3x4.
    #
    # For 3x2*2x4: split k into {0,1} and {2,3}
    # C[:,0:2] += A[:,0:2]*B[0:2,0:2] (3x2*2x2)
    # C[:,2:4] += A[:,0:2]*B[0:2,2:4] (3x2*2x2)
    # etc.
    #
    # R(3,2,2): multiply 3x2 by 2x2.
    # C[i,k] = sum_{j=0,1} A[i,j] B[j,k]
    # 3 rows, 2 cols each, 2 intermediate.
    #
    # Can we do R(3,2,2) < 6?
    # Each output C[i,k] needs 2 terms. 3x2 = 6 outputs.
    # But terms can be shared.
    #
    # Split into row 0 (1x2*2x2, rank 4) and rows 1-2 (2x2*2x2, rank 7): total 11.
    # Or row 0-1 (2x2*2x2, rank 7) and row 2 (1x2*2x2, rank 4): total 11.
    # Or all 3 rows at once?
    #
    # Actually for 3x2*2x2:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is: for each i, C[i,:] = A[i,:] * B (row vector times matrix).
    # The tensor has 3*2*2 = 12 entries, all 1.
    # R >= max(3*2, 2*2, 2*3) = 6.
    # Can we achieve 6? Each of the 6 outputs needs one term.
    # But terms share structure...
    #
    # Try: for each (i,k), one term: U_{i,j} = delta_{j,?}, V_{j,k} = ..., W_k = ...
    # With 6 terms, each term contributes to one output.
    # C[i,k] = sum_r U[i,j_r] A[i,j_r] V[j_r,k_r] B[j_r,k_r] W[k_r]
    #
    # This is getting complex. Let me just try rank 4 for 1x2*2x2 (confirmed)
    # and see if there's a way to save on the cross terms.
    #
    # The key insight: the current decomposition treats 4 independent sub-blocks
    # of 2x2*2x2 (for the top 2 rows) and 4 independent sub-blocks of 1x2*2x2
    # (for the bottom row).
    #
    # Can we share Strassen terms between the top-2-row blocks and the bottom-row
    # blocks? For example, the top-2-row block C[0:2,0:2] uses A[0:2,0:2]*B[0:2,0:2]
    # and A[0:2,2:4]*B[2:4,0:2]. The bottom-row block C[2:3,0:2] uses
    # A[2:3,0:2]*B[0:2,0:2] and A[2:3,2:4]*B[2:4,0:2].
    #
    # The B[0:2,0:2] factor is shared! If we could combine the A[0:2,0:2]*B[0:2,0:2]
    # and A[2:3,0:2]*B[0:2,0:2] into a single 3x2*2x2 block, we might save terms.
    #
    # 3x2*2x2: R(3,2,2). If R(3,2,2) = 6, then we save 7+4-6 = 5 per k-block,
    # giving 4*5 = 20 savings, total rank 24. That seems too good.
    #
    # Actually R(3,2,2): let's think about it.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k], i=0,1,2; k=0,1.
    # 6 outputs, each a sum of 2 products.
    #
    # Rank-1 term: u_i * v_{j,k} * w_k. This gives contributions to all 6 outputs
    # but structured.
    #
    # I think R(3,2,2) = 6 might not be achievable. Let's try:
    # 6 terms, one per output:
    # Term for C[i,k]: U[i,j]=delta, V[j,k]=1, W[k]=1
    # C[i,k] = A[i,j]*B[j,k] for specific j.
    # But C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] needs TWO terms per output.
    # So we'd need 12 terms with this approach.
    #
    # Shared terms: a term (U, V, W) contributes to C[i,k] the value
    # U[i,j] V[j,k] W[k] * A[i,j] B[j,k].
    # If U[i,j] is nonzero for multiple i and j, it contributes to multiple outputs.
    #
    # Try: Term 1: U=[1,0,0], V=I_2, W=[1,1]
    #   C[0,0] += A[0,0]*B[0,0]*1 + A[0,0]*B[1,0]*0...
    #   Actually V[j,k] with V=I: V[0,0]=1,V[0,1]=0,V[1,0]=0,V[1,1]=1
    #   C[0,k] += U[0,0]*V[0,k]*W[k]*A[0,0]*B[0,k] + U[0,1]*V[1,k]*W[k]*A[0,1]*B[1,k]
    #   = 1*V[0,k]*W[k]*A[0,0]*B[0,k] + 0*...
    #   For k=0: A[0,0]*B[0,0]. For k=1: 0.
    #
    # This is getting very tedious. Let me just verify the rank-44 solution is
    # correct and try to find any rank reduction through a more global approach.
    #
    # Actually, I just realized: the current code's Strategy A is verified.
    # Let me check if the 3x4*4x4 tensor has known rank.
    #
    # R(3,4,4) >= R(2,2,2) * (3/2)*(4/2)*(4/2) = 7*2 = 14 (very weak).
    #
    # Let me try a completely different approach: use the fact that
    # 3x4*4x4 can be viewed as (2+1)x4 * 4x(2+2) or other factorizations.
    #
    # What about: treat it as 4x4*4x4 with the first row zero? No, that's not
    # a tensor identity.
    #
    # Actually, here's an idea: 3x4*4x4 = (4x4*4x4) with one row of A zero.
    # R(4,4,4) >= R(3,4,4). The best known for 4x4*4x4 is 49 (Stoß 1983).
    # But we want 3x4*4x4 which could be less.
    #
    # Let me try yet another split:
    # C = A * B where A is 3x4, B is 4x4.
    # Write A = [A1; A2] where A1 is 2x4 and A2 is 1x4.
    # C = A1*B + A2*B
    # A1*B: 2x4 * 4x4. Split j: A1[0:2,0:2]*B[0:2,:] + A1[0:2,2:4]*B[2:4,:]
    #   Each 2x2*2x4: split k: 2x2*2x2 + 2x2*2x2 = 7+7=14. Total: 28.
    # A2*B: 1x4 * 4x4. Split j: A2[0:2]*B[0:2,:] + A2[2:4]*B[2:4,:]
    #   Each 1x2*2x4: split k: 1x2*2x2 + 1x2*2x2 = 4+4=8. Total: 16.
    # Grand total: 44. Same as before.
    #
    # The fundamental issue is R(1,2,2)=4 and R(2,2,2)=7, giving:
    # 4 * (7 + 4) = 4 * 11 = 44.
    #
    # Unless we can find a better decomposition that doesn't factor through
    # these sub-blocks.
    #
    # Let me try to search for a rank-43 or lower solution using a different
    # structure. One idea: use the fact that the 3rd row of A can be partially
    # absorbed into the 2x2 blocks using linear combinations.
    #
    # For example, instead of C[2:3,0:2] = A[2,0]*B[0,0:2] + A[2,1]*B[1,0:2]
    #                    + A[2,2]*B[2,0:2] + A[2,3]*B[3,0:2]  (rank 8)
    # Can we express some of this using the same products as the 2x2 blocks?
    #
    # In the Strassen decomposition of A[0:2,0:2]*B[0:2,0:2], we compute:
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # P2 = (A[1,0]+A[1,1])*B[0,0]
    # etc.
    #
    # These involve specific linear combinations of A[0:2,0:2] and B[0:2,0:2].
    # The 3rd row A[2,:] involves different A elements, so we can't directly
    # reuse these products.
    #
    # However, if we redefine the Strassen terms to use A[0:3,:] instead of
    # A[0:2,:], we might be able to share some products. But that changes the
    # 2x2 blocks and they might no longer be valid Strassen decompositions.
    #
    # I think rank 44 is likely optimal for this simple block structure.
    # Let me verify and submit.

    if err_a < 1e-9:
        return {"U": U_A, "V": V_A, "W": W_A}
    else:
        print(f"Strategy A failed: {err_a}")
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
