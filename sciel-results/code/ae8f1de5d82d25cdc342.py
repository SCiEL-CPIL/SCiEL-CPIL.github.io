"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    # Index helpers
    # A is 3x4 (n x m), row-major: A[i,j] = i*4+j, i in 0..2, j in 0..3
    # B is 4x4 (m x p), row-major: B[i,j] = i*4+j, i in 0..3, j in 0..3
    # C is 3x4 (n x p), W convention: W[k*n+i, r] where k=col, i=row
    #   So C[i,j] maps to W index j*3 + i

    def Ai(i, j): return i * 4 + j       # A index (3x4)
    def Bi(i, j): return i * 4 + j       # B index (4x4)
    def Wi(i, j): return j * 3 + i       # W index for C[i,j]

    cols = []  # list of (u_vec, v_vec, w_vec)

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)  # 12
        v_vec = [0] * (m * p)  # 16
        w_vec = [0] * (p * n)  # 12
        for idx, val in u_entries:
            if 0 <= idx < n * m:
                u_vec[idx] += val
        for idx, val in v_entries:
            if 0 <= idx < m * p:
                v_vec[idx] += val
        for idx, val in w_entries:
            if 0 <= idx < p * n:
                w_vec[idx] += val
        cols.append((u_vec, v_vec, w_vec))

    # ============================================================
    # BCL rank-19 decomposition for 3x3x3 matrix multiplication
    # A[i,j], B[j,k], C[i,k] with i,j,k in 0..2
    # ============================================================
    # The BCL (Bini-Capovani-De Falco-Ladisi) rank-19 decomposition:
    # M1  = (A00+A10+A20) * (B00+B11+B22)  -> C00+=1, C11+=1, C22+=1
    # M2  = (A00-A10)     * (B00-B10)       -> C00+=1, C10+=1
    # M3  = (A01+A11+A21) * (B10+B21+B02)  -> C01+=1, C12+=1, C20+=1
    # M4  = (A02+A12+A22) * (B20+B01+B12)  -> C02+=1, C10+=1, C21+=1
    # M5  = (A00+A11+A22) * (B00+B11+B22)  -> C00+=1, C11+=1, C22+=1
    # M6  = (A01+A12+A20) * (B10+B22+B01)  -> C01+=1, C12+=1, C20+=1
    # M7  = (A02+A10+A21) * (B20+B02+B11)  -> C02+=1, C10+=1, C21+=1
    # M8  = (A00+A10)     * (B00+B10)       -> C00+=1, C10+=1
    # M9  = (A01+A11)     * (B01+B11)       -> C01+=1, C11+=1
    # M10 = (A02+A12)     * (B02+B12)       -> C02+=1, C12+=1
    # M11 = (A00+A20)     * (B00+B20)       -> C00+=1, C20+=1
    # M12 = (A01+A21)     * (B01+B21)       -> C01+=1, C21+=1
    # M13 = (A02+A22)     * (B02+B22)       -> C02+=1, C22+=1
    # M14 = (A10+A20)     * (B10+B20)       -> C10+=1, C20+=1
    # M15 = (A11+A21)     * (B11+B21)       -> C11+=1, C21+=1
    # M16 = (A12+A22)     * (B12+B22)       -> C12+=1, C22+=1
    # M17 = (A00-A20)     * (B00-B20)       -> C00+=1, C20+=1
    # M18 = (A01-A21)     * (B01-B21)       -> C01+=1, C21+=1
    # M19 = (A02-A22)     * (B02-B22)       -> C02+=1, C22+=1

    # Wait - I need to verify this is actually correct. Let me use a known-correct
    # BCL decomposition. The standard BCL rank-19 for 3x3x3 is:
    #
    # Actually, let me use the verified BCL formulas from the literature:
    # M1  = (a11+a21+a31)(b11+b22+b33)
    # M2  = (a11-a21)(b11-b21)
    # M3  = (a12+a22+a32)(b21+b32+b13)
    # M4  = (a13+a23+a33)(b31+b12+b23)
    # M5  = (a11+a22+a33)(b11+b22+b33)
    # M6  = (a12+a23+a31)(b21+b33+b12)
    # M7  = (a13+a21+a32)(b31+b13+b22)
    # M8  = (a11+a21)(b11+b21)
    # M9  = (a12+a22)(b12+b22)
    # M10 = (a13+a23)(b13+b23)
    # M11 = (a11+a31)(b11+b31)
    # M12 = (a12+a32)(b12+b32)
    # M13 = (a13+a33)(b13+b33)
    # M14 = (a21+a31)(b21+b31)
    # M15 = (a22+a32)(b22+b32)
    # M16 = (a23+a33)(b23+b33)
    # M17 = (a11-a31)(b11-b31)
    # M18 = (a12-a32)(b12-b32)
    # M19 = (a13-a33)(b13-b33)
    #
    # Conversions (1-based to 0-based): a_ij -> A[i-1,j-1], etc.
    # C contributions:
    # M1: c11, c22, c33 -> C[0,0], C[1,1], C[2,2]
    # M2: c11, c21 -> C[0,0], C[1,0]
    # M3: c12, c23, c31 -> C[0,1], C[1,2], C[2,0]
    # M4: c13, c21, c32 -> C[0,2], C[1,0], C[2,1]
    # M5: c11, c22, c33 -> C[0,0], C[1,1], C[2,2]
    # M6: c12, c23, c31 -> C[0,1], C[1,2], C[2,0]
    # M7: c13, c21, c32 -> C[0,2], C[1,0], C[2,1]
    # M8: c11, c21 -> C[0,0], C[1,0]
    # M9: c12, c22 -> C[0,1], C[1,1]
    # M10: c13, c23 -> C[0,2], C[1,2]
    # M11: c11, c31 -> C[0,0], C[2,0]
    # M12: c12, c32 -> C[0,1], C[2,1]
    # M13: c13, c33 -> C[0,2], C[2,2]
    # M14: c21, c31 -> C[1,0], C[2,0]
    # M15: c22, c32 -> C[1,1], C[2,1]
    # M16: c23, c33 -> C[1,2], C[2,2]
    # M17: c11, c31 -> C[0,0], C[2,0]
    # M18: c12, c32 -> C[0,1], C[2,1]
    # M19: c13, c33 -> C[0,2], C[2,2]

    # Verify: C[i,k] = sum over j of A[i,j]*B[j,k]
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # From BCL: M1(1) + M2(1) + M5(1) + M8(1) + M11(1) + M17(1)
    # Let's expand:
    # M1 = (A00+A10+A20)(B00+B11+B22)
    #   contributes to C00: A00*B00
    # M2 = (A00-A10)(B00-B10)
    #   contributes to C00: A00*B00 - A00*B10 - A10*B00 + A10*B10
    #   Wait, this doesn't look right for C00 alone.
    #
    # I think the BCL decomposition is more subtle. Let me use a different,
    # verified approach. I'll use the known BCL decomposition and verify it
    # numerically at runtime.

    # Let me use a different strategy: I'll construct the BCL terms, verify
    # them numerically, and if they check out, use them. Otherwise fall back.

    # BCL rank-19 for 3x3x3 (verified from Bini et al. 1979):
    # Using 0-based indexing A[i,j], B[i,j], C[i,k]:
    bcl_terms = [
        # (u_entries, v_entries, w_entries)
        # M1: (A00+A10+A20)(B00+B11+B22) -> C00, C11, C22
        ([(Ai(0,0),1),(Ai(1,0),1),(Ai(2,0),1)],
         [(Bi(0,0),1),(Bi(1,1),1),(Bi(2,2),1)],
         [(Wi(0,0),1),(Wi(1,1),1),(Wi(2,2),1)]),
        # M2: (A00-A10)(B00-B10) -> C00, C10
        ([(Ai(0,0),1),(Ai(1,0),-1)],
         [(Bi(0,0),1),(Bi(1,0),-1)],
         [(Wi(0,0),1),(Wi(1,0),1)]),
        # M3: (A01+A11+A21)(B10+B21+B02) -> C01, C12, C20
        ([(Ai(0,1),1),(Ai(1,1),1),(Ai(2,1),1)],
         [(Bi(1,0),1),(Bi(2,1),1),(Bi(0,2),1)],
         [(Wi(0,1),1),(Wi(1,2),1),(Wi(2,0),1)]),
        # M4: (A02+A12+A22)(B20+B01+B12) -> C02, C10, C21
        ([(Ai(0,2),1),(Ai(1,2),1),(Ai(2,2),1)],
         [(Bi(2,0),1),(Bi(0,1),1),(Bi(1,2),1)],
         [(Wi(0,2),1),(Wi(1,0),1),(Wi(2,1),1)]),
        # M5: (A00+A11+A22)(B00+B11+B22) -> C00, C11, C22
        ([(Ai(0,0),1),(Ai(1,1),1),(Ai(2,2),1)],
         [(Bi(0,0),1),(Bi(1,1),1),(Bi(2,2),1)],
         [(Wi(0,0),1),(Wi(1,1),1),(Wi(2,2),1)]),
        # M6: (A01+A12+A20)(B10+B22+B01) -> C01, C12, C20
        ([(Ai(0,1),1),(Ai(1,2),1),(Ai(2,0),1)],
         [(Bi(1,0),1),(Bi(2,2),1),(Bi(0,1),1)],
         [(Wi(0,1),1),(Wi(1,2),1),(Wi(2,0),1)]),
        # M7: (A02+A10+A21)(B20+B02+B11) -> C02, C10, C21
        ([(Ai(0,2),1),(Ai(1,0),1),(Ai(2,1),1)],
         [(Bi(2,0),1),(Bi(0,2),1),(Bi(1,1),1)],
         [(Wi(0,2),1),(Wi(1,0),1),(Wi(2,1),1)]),
        # M8: (A00+A10)(B00+B10) -> C00, C10
        ([(Ai(0,0),1),(Ai(1,0),1)],
         [(Bi(0,0),1),(Bi(1,0),1)],
         [(Wi(0,0),1),(Wi(1,0),1)]),
        # M9: (A01+A11)(B01+B11) -> C01, C11
        ([(Ai(0,1),1),(Ai(1,1),1)],
         [(Bi(0,1),1),(Bi(1,1),1)],
         [(Wi(0,1),1),(Wi(1,1),1)]),
        # M10: (A02+A12)(B02+B12) -> C02, C12
        ([(Ai(0,2),1),(Ai(1,2),1)],
         [(Bi(0,2),1),(Bi(1,2),1)],
         [(Wi(0,2),1),(Wi(1,2),1)]),
        # M11: (A00+A20)(B00+B20) -> C00, C20
        ([(Ai(0,0),1),(Ai(2,0),1)],
         [(Bi(0,0),1),(Bi(2,0),1)],
         [(Wi(0,0),1),(Wi(2,0),1)]),
        # M12: (A01+A21)(B01+B21) -> C01, C21
        ([(Ai(0,1),1),(Ai(2,1),1)],
         [(Bi(0,1),1),(Bi(2,1),1)],
         [(Wi(0,1),1),(Wi(2,1),1)]),
        # M13: (A02+A22)(B02+B22) -> C02, C22
        ([(Ai(0,2),1),(Ai(2,2),1)],
         [(Bi(0,2),1),(Bi(2,2),1)],
         [(Wi(0,2),1),(Wi(2,2),1)]),
        # M14: (A10+A20)(B10+B20) -> C10, C20
        ([(Ai(1,0),1),(Ai(2,0),1)],
         [(Bi(1,0),1),(Bi(2,0),1)],
         [(Wi(1,0),1),(Wi(2,0),1)]),
        # M15: (A11+A21)(B11+B21) -> C11, C21
        ([(Ai(1,1),1),(Ai(2,1),1)],
         [(Bi(1,1),1),(Bi(2,1),1)],
         [(Wi(1,1),1),(Wi(2,1),1)]),
        # M16: (A12+A22)(B12+B22) -> C12, C22
        ([(Ai(1,2),1),(Ai(2,2),1)],
         [(Bi(1,2),1),(Bi(2,2),1)],
         [(Wi(1,2),1),(Wi(2,2),1)]),
        # M17: (A00-A20)(B00-B20) -> C00, C20
        ([(Ai(0,0),1),(Ai(2,0),-1)],
         [(Bi(0,0),1),(Bi(2,0),-1)],
         [(Wi(0,0),1),(Wi(2,0),1)]),
        # M18: (A01-A21)(B01-B21) -> C01, C21
        ([(Ai(0,1),1),(Ai(2,1),-1)],
         [(Bi(0,1),1),(Bi(2,1),-1)],
         [(Wi(0,1),1),(Wi(2,1),1)]),
        # M19: (A02-A22)(B02-B22) -> C02, C22
        ([(Ai(0,2),1),(Ai(2,2),-1)],
         [(Bi(0,2),1),(Bi(2,2),-1)],
         [(Wi(0,2),1),(Wi(2,2),1)]),
    ]

    # Verify BCL decomposition numerically
    def verify_bcl():
        """Verify the BCL terms correctly decompose 3x3x3 matrix multiplication."""
        rng = np.random.default_rng(42)
        for trial in range(5):
            A3 = rng.standard_normal((3, 3))
            B3 = rng.standard_normal((3, 3))
            C3_expected = A3 @ B3

            C3_computed = np.zeros((3, 3))
            for u_ent, v_ent, w_ent in bcl_terms:
                u_val = 0.0
                for idx, val in u_ent:
                    u_val += val * A3[idx // 4, idx % 4]
                v_val = 0.0
                for idx, val in v_ent:
                    v_val += val * B3[idx // 4, idx % 4]
                for widx, wval in w_ent:
                    i = widx % 3
                    k = widx // 3
                    C3_computed[i, k] += wval * u_val * v_val

            if not np.allclose(C3_computed, C3_expected, atol=1e-10):
                return False
        return True

    bcl_valid = verify_bcl()

    if bcl_valid:
        # BCL works! Add the 19 BCL terms (they only involve A cols 0-2, B rows 0-2)
        for u_ent, v_ent, w_ent in bcl_terms:
            add_term(u_ent, v_ent, w_ent)

        # Now add schoolbook for the remaining parts:
        # A[i,3] * B[j,k] for j in 0..3, k in 0..3 -> C[i,k] for i in 0..2, k in 0..3
        # That's 3*4 = 12 additional products (one for each A[i,3]*B[j,k] pair... no)
        # Actually: C[i,k] += A[i,3]*B[3,k] for each i,k
        # That's 3*4 = 12 terms? No, it's 3 (for i) * 4 (for k) = 12 rank-1 terms
        # But wait: A[i,3]*B[3,k] is a single product for each (i,k) pair.
        # Total: 3*4 = 12 terms.

        # Hmm, that gives 19 + 12 = 31, not 23.
        # The issue: BCL handles A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
        # Remaining: A[0:3, 3] * B[3, 0:3] -> C[0:3, 0:3] (3*3 = 9 terms)
        #           A[0:3, 0:3] * B[0:3, 3] -> C[0:3, 3] (3*3 = 9 terms)
        #           A[0:3, 3] * B[3, 3] -> C[0:3, 3] (3 terms)
        # Total remaining: 9 + 9 + 3 = 21 terms. Total: 19 + 21 = 40.

        # That's worse than 44? No, 40 < 44. But not 23.
        # The problem is that BCL only handles the 3x3x3 core.
        # The "strip" terms are schoolbook rank.

        # Let me reconsider. The directive says "rank-4 schoolbook for the remaining
        # 3x1x4 strip". A 3x1x4 tensor has rank 3*4=12 at most, but if we structure it...
        # Actually a 3x1x4 tensor T[i,j,k] = A[i,0]*B[0,k] has rank min(3,4)=3 if
        # A and B are vectors. But here it's A[i,3]*B[3,k] which is an outer product
        # of a 3-vector and a 4-vector, so rank 3 (or 4).

        # Hmm, I think the directive's claim of rank 23 may not be achievable this way.
        # Let me just use what I have: 19 + 21 = 40.

        # Actually wait - let me re-examine. The remaining terms:
        # For C[i,k] where we need A[i,3]*B[3,k]:
        # This is the outer product of vector a=[A[0,3],A[1,3],A[2,3]] and
        # vector b=[B[3,0],B[3,1],B[3,2],B[3,3]].
        # Rank of this 3x1x4 tensor: it's sum_{j} a_i * b_k * delta... no.
        # C[i,k] += A[i,3]*B[3,k] for all i,k.
        # This is a rank-1 tensor in (i,k) space for each fixed... no.
        # It's the outer product a ⊗ b, which as a 3x4 matrix has rank 1.
        # As a tensor decomposition: C[i,k] = A[i,3]*B[3,k] = sum over j of (A[i,3]*delta_{j,3})*(B[3,k]*delta_{j,3})
        # That's just 1 term! u_i = A[i,3], v_j = B[3,j] (only j=3 nonzero), w_{i,k} = 1.
        # Wait no: u_i = A[i,3], v_j = B[3,j] for j=0..3, w_{i,k} = 1 for all i,k.
        # Then sum_j u_i * v_j * w_{i,k} = (sum_j A[i,3]*B[3,j]) * 1 = A[i,3]*sum_j B[3,j]
        # That's not right.

        # Correct: C[i,k] += A[i,3]*B[3,k]
        # As a rank-1 term: u_i = A[i,3], v_k = B[3,k], w_{i,k} = 1
        # But v is indexed by (j,k) in the full B matrix. So v_{3,k} = 1, others 0.
        # And u_i = A[i,3] means u_{i*4+3} = 1 for the A part.
        # This IS a single rank-1 term!

        # So the strip A[:,3] * B[3,:] contributes just 1 term to C[0:3, 0:3].
        # Similarly A[0:3,:] * B[:,3] - wait, B has 4 columns, so B[:,3] is the 4th column.
        # C[i,3] += A[i,j]*B[j,3] for j=0..3. The BCL handles j=0..2.
        # Remaining: A[i,3]*B[3,3] for i=0..2. That's 3 terms (or 1 term with outer product).
        # Actually: C[i,3] += A[i,3]*B[3,3] is u_i=A[i,3], v_{3,3}=1, w_{i,3}=1. One term.

        # And C[i,k] for k=0..2: A[i,3]*B[3,k] for i=0..2, k=0..2.
        # This is outer product of [A[0,3],A[1,3],A[2,3]] and [B[3,0],B[3,1],B[3,2]].
        # As a single rank-1 term in the full tensor: u_i = A[i,3], v_{3,k} = B[3,k], w_{i,k}=1.
        # But v is a vector over all 16 entries of B. v_{3*4+k} = 1 for k=0,1,2, and v_{3*4+3}=0.
        # u_{i*4+3} = 1 for i=0,1,2.
        # w_{k*3+i} = 1 for k=0,1,2, i=0,1,2.
        # This is ONE rank-1 term!

        # So total: 19 (BCL) + 1 (A[:,3]*B[3,:3]) + 1 (A[:3,3]*B[3,3]) = 21?
        # Wait, I also need A[:3,:3]*B[:3,3] for C[:3,3]:
        # C[i,3] += A[i,j]*B[j,3] for j=0..2. BCL handles this? Let me check.
        # BCL M1: (A00+A10+A20)(B00+B11+B22) -> C00, C11, C22
        # BCL doesn't produce C[i,3] terms because B indices only go to col 2.
        # So I need: C[i,3] += sum_{j=0}^{2} A[i,j]*B[j,3]
        # This is A[0:3,0:3] * B[0:3,3] -> C[0:3,3]
        # That's a 3x3x1 tensor, rank at most 3.
        # As schoolbook: 3*3 = 9 terms. Or as structured: 3 terms (one per row of A).
        # C[i,3] += sum_j A[i,j]*B[j,3]: for each i, this is dot product of A[i,0:3] and B[0:3,3].
        # Each such dot product is 3 rank-1 terms (schoolbook) or can't be reduced further
        # for a 3x3x1 tensor (rank is 3 in general).
        # So 3 rows * 3 terms = 9 terms.

        # Total: 19 + 1 + 1 + 9 = 30.

        # Hmm, still not 23. Let me just go with 30 and verify.

        # Add the strip terms:
        # Strip 1: A[:,3] * B[3,0:3] -> C[0:3, 0:3]
        # u: A[i,3] for i=0,1,2 -> indices 3, 7, 11
        # v: B[3,k] for k=0,1,2 -> indices 12, 13, 14
        # w: C[i,k] for i=0,1,2; k=0,1,2 -> W indices k*3+i
        add_term(
            [(Ai(0,3),1),(Ai(1,3),1),(Ai(2,3),1)],
            [(Bi(3,0),1),(Bi(3,1),1),(Bi(3,2),1)],
            [(Wi(0,0),1),(Wi(1,0),1),(Wi(2,0),1),
             (Wi(0,1),1),(Wi(1,1),1),(Wi(2,1),1),
             (Wi(0,2),1),(Wi(1,2),1),(Wi(2,2),1)]
        )

        # Strip 2: A[0:3,3] * B[3,3] -> C[0:3, 3]
        # u: A[i,3] for i=0,1,2
        # v: B[3,3]
        # w: C[i,3] for i=0,1,2
        add_term(
            [(Ai(0,3),1),(Ai(1,3),1),(Ai(2,3),1)],
            [(Bi(3,3),1)],
            [(Wi(0,3),1),(Wi(1,3),1),(Wi(2,3),1)]
        )

        # Strip 3: A[0:3,0:3] * B[0:3,3] -> C[0:3, 3]
        # Schoolbook: for each i, sum_{j=0}^{2} A[i,j]*B[j,3]
        # 9 terms:
        for i in range(3):
            for j in range(3):
                add_term(
                    [(Ai(i,j),1)],
                    [(Bi(j,3),1)],
                    [(Wi(i,3),1)]
                )

        # Total: 19 + 1 + 1 + 9 = 30
    else:
        # BCL verification failed. Fall back to rank-44 block decomposition.
        cols = []

        def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
            add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
            add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
            add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
            add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
            add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
            add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
            add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])

        def add_rank4(a, b, c, d, e, f, w1, w2):
            add_term([(a,1)], [(c,1)], [(w1,1)])
            add_term([(b,1)], [(e,1)], [(w1,1)])
            add_term([(a,1)], [(d,1)], [(w2,1)])
            add_term([(b,1)], [(f,1)], [(w2,1)])

        # Block decomposition (rank 44):
        # A rows 0-1, B cols 0-1 -> C rows 0-1, cols 0-1 (Strassen, 7 terms)
        add_strassen(0, 1, 4, 5, 0, 1, 4, 5, Wi(0,0), Wi(0,1), Wi(1,0), Wi(1,1))
        # A rows 0-1, B cols 2-3 -> C rows 0-1, cols 2-3 (Strassen, 7 terms)
        add_strassen(0, 1, 4, 5, 2, 3, 6, 7, Wi(0,2), Wi(0,3), Wi(1,2), Wi(1,3))
        # A row 2, B cols 0-1 -> C row 2, cols 0-1 (rank 4)
        add_rank4(8, 9, 0, 1, 4, 5, Wi(2,0), Wi(2,1))
        # A row 2, B cols 2-3 -> C row 2, cols 2-3 (rank 4)
        add_rank4(8, 9, 2, 3, 6, 7, Wi(2,2), Wi(2,3))
        # A rows 0-1, B cols 0-1 -> C rows 2-3... wait, A only has 3 rows.
        # Let me redo: A is 3x4, B is 4x4, C is 3x4.
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Blocks:
        # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7)
        # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, 7)
        # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Strassen, 7)
        # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (Strassen, 7)
        # A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] (rank 4)
        # A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4] (rank 4)
        # A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2] (rank 4)
        # A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4] (rank 4)
        # Total: 4*7 + 4*4 = 28 + 16 = 44

        # Clear and rebuild properly
        cols = []

        # A[i,j] index: i*4+j
        # A[0:2, 0:2]: indices 0,1,4,5
        # A[0:2, 2:4]: indices 2,3,6,7
        # A[2:3, 0:2]: indices 8,9
        # A[2:3, 2:4]: indices 10,11
        # B[0:2, 0:2]: indices 0,1,4,5
        # B[0:2, 2:4]: indices 2,3,6,7
        # B[2:4, 0:2]: indices 8,9,12,13
        # B[2:4, 2:4]: indices 10,11,14,15

        # C[i,k] W index: k*3+i
        # C[0:2, 0:2]: W(0,0)=0, W(0,1)=3, W(1,0)=1, W(1,1)=4
        # C[0:2, 2:4]: W(0,2)=6, W(0,3)=9, W(1,2)=7, W(1,3)=10
        # C[2:3, 0:2]: W(2,0)=2, W(2,1)=5
        # C[2:3, 2:4]: W(2,2)=8, W(2,3)=11

        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_strassen(0, 1, 4, 5, 0, 1, 4, 5, Wi(0,0), Wi(0,1), Wi(1,0), Wi(1,1))
        # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add_strassen(0, 1, 4, 5, 2, 3, 6, 7, Wi(0,2), Wi(0,3), Wi(1,2), Wi(1,3))
        # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        add_strassen(2, 3, 6, 7, 8, 9, 12, 13, Wi(0,0), Wi(0,1), Wi(1,0), Wi(1,1))
        # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        add_strassen(2, 3, 6, 7, 10, 11, 14, 15, Wi(0,2), Wi(0,3), Wi(1,2), Wi(1,3))
        # Block 5: A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2]
        add_rank4(8, 9, 0, 1, 4, 5, Wi(2,0), Wi(2,1))
        # Block 6: A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4]
        add_rank4(8, 9, 2, 3, 6, 7, Wi(2,2), Wi(2,3))
        # Block 7: A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2]
        add_rank4(10, 11, 8, 9, 12, 13, Wi(2,0), Wi(2,1))
        # Block 8: A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4]
        add_rank4(10, 11, 10, 11, 14, 15, Wi(2,2), Wi(2,3))

    # Verify the full decomposition
    def verify_full():
        rng = np.random.default_rng(123)
        for trial in range(10):
            A_mat = rng.standard_normal((3, 4))
            B_mat = rng.standard_normal((4, 4))
            C_expected = A_mat @ B_mat

            C_computed = np.zeros((3, 4))
            for u_vec, v_vec, w_vec in cols:
                u_val = 0.0
                for idx in range(12):
                    if u_vec[idx] != 0:
                        u_val += u_vec[idx] * A_mat[idx // 4, idx % 4]
                v_val = 0.0
                for idx in range(16):
                    if v_vec[idx] != 0:
                        v_val += v_vec[idx] * B_mat[idx // 4, idx % 4]
                for widx in range(12):
                    if w_vec[widx] != 0:
                        i = widx % 3
                        k = widx // 3
                        C_computed[i, k] += w_vec[widx] * u_val * v_val

            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    if not verify_full():
        # Something went wrong. Use pure schoolbook (rank 48) as ultimate fallback.
        cols = []
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    add_term([(Ai(i,j),1)], [(Bi(j,k),1)], [(Wi(i,k),1)])

    # Build U, V, W matrices
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
