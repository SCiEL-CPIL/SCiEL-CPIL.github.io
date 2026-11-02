import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strategy: Decompose 2x4 by 4x8 multiplication.
    # Split the intermediate dimension m=4 into two channels of size 2.
    # A = [A0 | A1] where A0, A1 are 2x2.
    # B = [B0; B1] where B0, B1 are 2x8.
    # C = A0 B0 + A1 B1.
    #
    # Each term is a 2x2 by 2x8 multiplication.
    # We can further decompose 2x2 by 2x8.
    # Split B into 4 blocks of 2x2: B = [B00, B01, B02, B03].
    # Then A0 B0 = A0 B00 + A0 B01 + A0 B02 + A0 B03.
    # This results in 8 separate 2x2 by 2x2 multiplications.
    #
    # Using Strassen for each 2x2x2 gives rank 7.
    # Total rank = 8 * 7 = 56.
    #
    # Can we do better?
    # The rank of 2x2x2 is 7.
    # The rank of 2x4x8 is at least max(n*m, m*p, p*n) = 32? No, that's a loose bound.
    # Actually, the rank of matrix multiplication M_{n,m,p} is known to be between
    # n*m*p / max(n,m,p) and n*m*p.
    # For 2x4x8, lower bound is 64/8 = 8. Upper bound 64.
    # Known ranks:
    # 2x2x2: 7
    # 2x2x4: 13? (Actually, rank of 2x2x4 is 13? No, 2x2x4 is 13? Let's check.
    # Rank of 2x2x4 is 13? No, it's 13 for 2x2x4?
    # Wait, rank of 2x2x4 is 13? No, 2x2x4 is 13?
    # Actually, rank of 2x2x4 is 13? No, 2x2x4 is 13?
    # Let's stick to the known decomposition.
    #
    # Is there a decomposition with rank < 56?
    # Yes. We can share some of the Strassen terms if we group them carefully?
    # No, the 8 multiplications are independent in the tensor sense because
    # the indices (i, j, k) are disjoint for different blocks?
    # Let's check the indices.
    # Block (s, j) where s in {0,1} (channel) and j in {0,1,2,3} (col block).
    # A indices: i in {0,1}, j_g in {2s, 2s+1}.
    # B indices: j_g in {2s, 2s+1}, k_g in {2j, 2j+1}.
    # C indices: i in {0,1}, k_g in {2j, 2j+1}.
    #
    # The tensor T_{i, j_g, k_g} is non-zero only if j_g is in the range of the block.
    # Since the j_g ranges are disjoint for different s, and k_g ranges are disjoint for different j,
    # the supports of the rank-1 tensors from different blocks are disjoint in the (j, k) plane?
    # No, the rank-1 tensor is u ⊗ v ⊗ w.
    # u has support on specific i, j_g.
    # v has support on specific j_g, k_g.
    # w has support on specific k_g, i.
    #
    # If we look at the "middle" index j_g, the blocks for s=0 use j_g=0,1 and s=1 use j_g=2,3.
    # So the v-vectors for s=0 have support on rows 0,1 of B.
    # The v-vectors for s=1 have support on rows 2,3 of B.
    # Thus, the v-vectors from different s are orthogonal in support? No, they are disjoint in support.
    # This means the rank-1 terms for s=0 and s=1 cannot cancel each other in the sum
    # unless we combine them. But they operate on disjoint parts of B.
    # However, the rank is the sum of the ranks of the independent sub-tensors?
    # Yes, if the supports are disjoint in a way that prevents linear dependence.
    # Specifically, if we view the tensor as a matrix M_{(i,k), j}, then
    # for s=0, the non-zero entries are in rows j=0,1.
    # for s=1, the non-zero entries are in rows j=2,3.
    # So the matrix is block diagonal with respect to j.
    # The rank of a block diagonal matrix is the sum of the ranks of the blocks.
    # So Rank(2x4x8) = Rank(Block s=0) + Rank(Block s=1).
    # Block s=0 is 2x2x8 (A is 2x2, B is 2x8, C is 2x8).
    # Block s=1 is 2x2x8.
    # So Rank(2x4x8) = 2 * Rank(2x2x8).
    #
    # Now, what is Rank(2x2x8)?
    # 2x2x8 can be seen as 4 blocks of 2x2x2.
    # B is 2x8 = [B0, B1, B2, B3] (each 2x2).
    # A is 2x2.
    # C is 2x8 = [C0, C1, C2, C3].
    # C_j = A B_j.
    # These are 4 independent 2x2x2 multiplications.
    # Are they independent?
    # The j-index in the middle is shared? No, the middle index is the column of A / row of B.
    # In 2x2x8, the middle dimension is 2.
    # So it's just 4 separate 2x2x2 multiplications that share the same A?
    # No, A is the same 2x2 matrix.
    # C_0 = A B_0
    # C_1 = A B_1
    # etc.
    # The tensor is T_{i, l, k} where l is 0,1 (middle dim).
    # k is 0..7.
    # For a fixed l, k determines which B block.
    # The rank-1 terms for B_0 involve v-vectors with support on k=0,1.
    # The rank-1 terms for B_1 involve v-vectors with support on k=2,3.
    # Since the k-supports are disjoint, the rank-1 terms from different B blocks are linearly independent.
    # So Rank(2x2x8) = 4 * Rank(2x2x2) = 4 * 7 = 28.
    #
    # Therefore, Rank(2x4x8) = 2 * 28 = 56.
    #
    # So 56 is likely the minimal rank for this specific structure if we don't find a cleverer decomposition
    # that mixes the blocks.
    #
    # Is it possible to mix?
    # The tensor is T_{i, j, k}.
    # i in 0,1. j in 0..3. k in 0..7.
    # T_{i, j, k} = 1.
    #
    # We established that the tensor is block diagonal in j (splitting j into 0,1 and 2,3).
    # And each block is block diagonal in k (splitting k into pairs).
    # So the tensor is a direct sum of 8 copies of the 2x2x2 tensor.
    # The rank of a direct sum of tensors is the sum of the ranks.
    # So Rank = 8 * 7 = 56.
    #
    # Unless... there is a decomposition that doesn't respect this block structure.
    # But the block structure is inherent to the indices.
    # The rank of the tensor T is the minimal r such that T = sum_{r'} u_r' ⊗ v_r' ⊗ w_r'.
    # If the tensor is a direct sum T = T1 ⊕ T2 ⊕ ... ⊕ T8, then Rank(T) >= Rank(T1) + ... + Rank(T8).
    # This is a known result: the rank of a direct sum is at least the sum of the ranks.
    # And we have a construction that achieves exactly the sum.
    # So the rank is exactly 56.
    #
    # Wait, is it a direct sum?
    # A direct sum of tensors T1 ⊕ T2 means that T1 and T2 have disjoint supports in at least one mode.
    # Here, T_{s,j} (for s in 0,1 and j in 0..3) has support:
    # i: all
    # j: {2s, 2s+1}
    # k: {2j, 2j+1}
    #
    # Consider T_{0,0} and T_{0,1}.
    # T_{0,0} support: j in {0,1}, k in {0,1}.
    # T_{0,1} support: j in {0,1}, k in {2,3}.
    # They share the j-support {0,1}. They have disjoint k-supports.
    # So they are "disjoint" in the k-mode.
    # This implies linear independence of the rank-1 terms from different k-blocks.
    # Similarly, T_{0,0} and T_{1,0}.
    # T_{0,0} support: j in {0,1}, k in {0,1}.
    # T_{1,0} support: j in {2,3}, k in {0,1}.
    # They share the k-support {0,1}. They have disjoint j-supports.
    #
    # In general, if two tensors have disjoint support in at least one mode, their rank-1 decompositions
    # cannot be mixed to reduce the total rank.
    # Proof sketch: Suppose T1 and T2 have disjoint support in mode 2.
    # Let T1 = sum_{i=1}^{r1} u_i ⊗ v_i ⊗ w_i.
    # Let T2 = sum_{j=1}^{r2} u'_j ⊗ v'_j ⊗ w'_j.
    # Suppose T1 + T2 = sum_{k=1}^{r} a_k ⊗ b_k ⊗ c_k with r < r1 + r2.
    # This is not immediately obvious that r >= r1 + r2.
    # However, for matrix multiplication tensors, it is known that the rank is additive
    # for block diagonal structures where the blocks are "independent" in the sense above.
    # Specifically, if the supports are disjoint in one mode, the rank is additive.
    #
    # So 56 is the minimal rank.
    #
    # The current program constructs rank 56.
    # The score is 1.1429.
    # The seed program likely had rank 64 (schoolbook).
    # 64/56 = 1.142857...
    # So the current program is optimal within this family (and likely globally).
    #
    # Can we improve the score?
    # The score is seed/value.
    # If seed is 64, and value is 56, score is 64/56 = 1.1428.
    # To improve, we need rank < 56.
    # But we argued rank is 56.
    #
    # Is there any possibility that the rank is lower?
    # Some sources suggest Rank(2x2x2)=7, Rank(2x2x4)=13?
    # If Rank(2x2x4)=13, then Rank(2x2x8) might be less than 4*7=28?
    # No, 2x2x8 is 4 blocks of 2x2x2.
    # If Rank(2x2x4)=13, then 2x2x4 is two blocks of 2x2x2.
    # 2*7=14. If Rank is 13, it means we can save 1 rank by mixing the two 2x2x2 blocks.
    # Is that possible?
    # T_{2x2x4} = T_{2x2x2} ⊕ T_{2x2x2} (disjoint in k).
    # If Rank is 13 < 14, then the rank is NOT additive for disjoint k-support?
    # Let's check literature.
    # "The rank of the 2x2x4 matrix multiplication tensor is 13."
    # Yes, this is a known result.
    # So we CAN save rank by mixing.
    #
    # How?
    # The 2x2x4 tensor can be decomposed with rank 13.
    # This implies that the 8 blocks of 2x2x2 in the 2x4x8 case might not be fully independent.
    #
    # Let's look at the structure again.
    # 2x4x8 = 2 * (2x2x8).
    # 2x2x8 = 2 * (2x2x4).
    # So 2x4x8 = 4 * (2x2x4).
    # If Rank(2x2x4) = 13, then Rank(2x4x8) <= 4 * 13 = 52.
    #
    # Can we go lower?
    # Maybe the 4 blocks of 2x2x4 can be mixed further?
    #
    # Let's try to construct a rank 52 decomposition.
    # We need a rank 13 decomposition for 2x2x4.
    #
    # Strassen for 2x2x2 uses 7 multiplications.
    # For 2x2x4, we have two 2x2x2 multiplications: C1 = A B1, C2 = A B2.
    # Standard Strassen would use 7+7=14.
    # The rank 13 decomposition saves 1 multiplication.
    #
    # How to find the rank 13 decomposition?
    # It involves combining the Strassen terms of the two blocks.
    #
    # Let's look up the explicit rank 13 decomposition for 2x2x4.
    # Or derive it.
    #
    # Actually, there is a known decomposition for 2x2x4 with rank 13.
    # It is not trivial to write down from scratch.
    #
    # However, we can try a heuristic search for rank 52 or 51.
    # But the search space is huge.
    #
    # Alternative:
    # Is there a simpler improvement?
    # The current rank is 56.
    # If we can find a rank 55 decomposition, score = 64/55 = 1.1636.
    # If rank 52, score = 64/52 = 1.2307.
    #
    # Let's check if the rank of 2x2x4 is indeed 13.
    # Yes, Bini et al. showed Rank(2x2x4)=13.
    #
    # So the target is rank 52.
    #
    # How to construct it?
    # We need to mix the 4 pairs of 2x2x2 blocks.
    #
    # Let's try to implement a rank 13 decomposition for 2x2x4.
    #
    # The 2x2x4 tensor T_{i,j,k} with i in 0,1; j in 0,1; k in 0..3.
    # Split k into k1 (0,1) and k2 (2,3).
    # T = T1 + T2 where T1 is 2x2x2 (k=0,1) and T2 is 2x2x2 (k=2,3).
    #
    # Strassen for T1:
    # M1 = (a00+a11)(b00+b11) -> C00, C11
    # M2 = (a10+a11)b00 -> C10, C11
    # M3 = a00(b01-b11) -> C01, C11
    # M4 = a11(b10-b00) -> C00, C10
    # M5 = (a00+a01)b11 -> C00, C01
    # M6 = (a10-a00)(b00+b01) -> C11
    # M7 = (a01-a11)(b10+b11) -> C00
    #
    # Strassen for T2 (using B2 = [b20 b21; b22 b23]):
    # N1 = (a00+a11)(b20+b23)
    # N2 = (a10+a11)b20
    # N3 = a00(b21-b23)
    # N4 = a11(b22-b20)
    # N5 = (a00+a01)b23
    # N6 = (a10-a00)(b20+b21)
    # N7 = (a01-a11)(b22+b23)
    #
    # Can we combine M and N terms?
    # Notice that some terms share the same A factors.
    # For example, M1 and N1 both use (a00+a11).
    # M1 = (a00+a11)(b00+b11)
    # N1 = (a00+a11)(b20+b23)
    # We can write (a00+a11)(b00+b11+b20+b23) = M1 + N1.
    # This gives one term: P1 = (a00+a11)(b00+b11+b20+b23).
    # This term contributes to C00, C11 (from M1) and C20, C23 (from N1, mapped to C indices).
    # Wait, C indices for T2 are k=2,3.
    # So P1 contributes to C00, C11 (k=0,1) and C02, C13 (k=2,3)?
    # No, M1 contributes to C00 (k=0) and C11 (k=1).
    # N1 contributes to C02 (k=2) and C13 (k=3).
    # So P1 contributes to C00, C11, C02, C13.
    #
    # This saves 1 rank (2 terms -> 1 term).
    #
    # Let's check other combinations.
    # M2 = (a10+a11)b00. N2 = (a10+a11)b20.
    # P2 = (a10+a11)(b00+b20).
    # M2 contributes to C10, C11.
    # N2 contributes to C12, C13.
    # So P2 contributes to C10, C11, C12, C13.
    #
    # M6 = (a10-a00)(b00+b01). N6 = (a10-a00)(b20+b21).
    # P6 = (a10-a00)(b00+b01+b20+b21).
    # M6 contributes to C11.
    # N6 contributes to C13.
    # So P6 contributes to C11, C13.
    #
    # M3 = a00(b01-b11). N3 = a00(b21-b23).
    # Can we combine? Not directly, the B factors are different.
    # But we can write a00(b01-b11+b21-b23) = M3 + N3.
    # P3 = a00(b01-b11+b21-b23).
    # M3 contributes to C01, C11.
    # N3 contributes to C03, C13.
    # So P3 contributes to C01, C11, C03, C13.
    #
    # M4 = a11(b10-b00). N4 = a11(b22-b20).
    # P4 = a11(b10-b00+b22-b20).
    # M4 contributes to C00, C10.
    # N4 contributes to C02, C12.
    # So P4 contributes to C00, C10, C02, C12.
    #
    # M5 = (a00+a01)b11. N5 = (a00+a01)b23.
    # P5 = (a00+a01)(b11+b23).
    # M5 contributes to C00, C01.
    # N5 contributes to C02, C03.
    # So P5 contributes to C00, C01, C02, C03.
    #
    # M7 = (a01-a11)(b10+b11). N7 = (a01-a11)(b22+b23).
    # P7 = (a01-a11)(b10+b11+b22+b23).
    # M7 contributes to C00.
    # N7 contributes to C02.
    # So P7 contributes to C00, C02.
    #
    # We have P1..P7. That's 7 terms.
    # Plus the remaining terms?
    # We combined M1,N1 -> P1.
    # M2,N2 -> P2.
    # M3,N3 -> P3.
    # M4,N4 -> P4.
    # M5,N5 -> P5.
    # M6,N6 -> P6.
    # M7,N7 -> P7.
    #
    # This uses 7 terms. But we started with 14.
    # So we saved 7 terms? That would give rank 7 for 2x2x4?
    # That's impossible. Rank(2x2x4) >= Rank(2x2x2) = 7.
    # But 2x2x4 is larger.
    # The issue is that the P terms are not rank-1 in the full 2x2x4 tensor?
    # P1 = (a00+a11)(b00+b11+b20+b23).
    # This is a product of a linear form in A and a linear form in B.
    # It is rank-1.
    # But does it correctly decompose the tensor?
    # We need to check if the sum of P1..P7 equals the tensor.
    #
    # Let's check the contributions to C00.
    # Original: M1 + M4 - M5 + M7 (from T1) + N1 + N4 - N5 + N7 (from T2, mapped to C02? No, C00 is k=0).
    # Wait, T2 contributes to C with k=2,3.
    # So T2 does NOT contribute to C00 (k=0).
    # So C00 = M1 + M4 - M5 + M7.
    #
    # In our P decomposition:
    # P1 contributes to C00? P1 = (a00+a11)(b00+b11+b20+b23).
    # The B part has b20, b23.
    # When we compute C = sum P_r * W_r, the W vector for P1 must select the correct C entries.
    # For P1, the B part is (b00+b11+b20+b23).
    # This B part is a sum of 4 terms.
    # The product (a00+a11)(b00+b11+b20+b23) is a single scalar.
    # This scalar multiplies the W vector.
    # The W vector must have non-zero entries at C00, C11, C02, C13.
    # But the contribution to C00 should be (a00+a11)(b00+b11).
    # The contribution to C02 should be (a00+a11)(b20+b23).
    #
    # If we use a single rank-1 term P1 = u1 ⊗ v1 ⊗ w1,
    # where u1 = (a00+a11), v1 = (b00+b11+b20+b23),
    # then the contribution to C00 is u1 * v1 * w1[C00].
    # This equals (a00+a11)(b00+b11+b20+b23) * w1[C00].
    # We want this to be (a00+a11)(b00+b11).
    # This is only possible if w1[C00] = 1 and the terms (a00+a11)(b20+b23) cancel out or are zero.
    # But they are not zero.
    #
    # So we CANNOT simply combine M1 and N1 into one rank-1 term because the B factors are different
    # and contribute to different C entries.
    # The "combining" trick only works if the B factors are the same or if the C entries are the same.
    # Here, the C entries are different (k=0 vs k=2).
    #
    # So the rank is NOT reduced by this naive combination.
    #
    # The rank 13 decomposition for 2x2x4 is more subtle.
    # It involves finding a different set of 13 rank-1 terms.
    #
    # Given the complexity and the time limit, and the fact that the current rank 56 is a valid
    # and likely near-optimal solution (given the difficulty of finding the rank 52 decomposition),
    # I will stick with the rank 56 solution.
    #
    # The score is 1.1429.
    # The seed is 64.
    # 64/56 = 1.142857.
    #
    # I will output the current program, which is correct and efficient.

    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]

    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11

    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11

    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11

    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10

    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01

    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11

    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56

    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56

    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7

            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]

            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]

            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val

    if time.time() - start_time > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
