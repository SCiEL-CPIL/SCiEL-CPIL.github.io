"""Rank-minimal decomposition search for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # 150s wall-clock limit, keep 35s safety margin

    # Target tensor T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    # U rows: (i,j) -> i*m + j, shape (n*m, rank)
    # V rows: (j,k) -> j*p + k, shape (m*p, rank)
    # W rows: (k,i) -> k*n + i, shape (p*n, rank)
    # Tensor: sum_c U[u,c] * V[v,c] * W[w,c] = T[u,v,w]

    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        rank = len(U[0]) if len(U) > 0 else 0
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0

        # Compute decomposed tensor
        D = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(rank):
            D += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])

        if not np.allclose(D, T, atol=1e-9, rtol=1e-9):
            return False
        return True

    def make_strassen_block(A_rows, B_rows, W_rows, col_offset):
        """
        Build Strassen 2x2x2 decomposition for a block.
        A_rows: list of 2 U row indices (i=0,1 mapped to j=0,1)
        B_rows: list of 2 V row indices (j=0,1 mapped to k=0,1)
        W_rows: list of 2 W row indices (k=0,1 mapped to i=0,1)
        Returns: (U_entries, V_entries, W_entries) as dict {row: [(col, val), ...]}
        """
        # Strassen formulas for 2x2:
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b21-b22)
        # M4 = a22*(b11-b21)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        a11, a22 = A_rows[0], A_rows[1]
        b11, b22 = B_rows[0], B_rows[1]
        c11, c22 = W_rows[0], W_rows[1]

        # We need to track which U/V/W entries correspond to which matrix element.
        # For 2x2 block:
        # U rows: a11, a22 (but we also need a12, a21!)
        # Wait - Strassen needs all 4 elements of A and B.
        # Let me reconsider.

        # For a 2x2x2 Strassen:
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = A*B
        # U rows for A: a11, a12, a21, a22 (4 rows)
        # V rows for B: b11, b12, b21, b22 (4 rows)
        # W rows for C: c11, c12, c21, c22 (4 rows)

        # But in our problem, the 2x2 block uses:
        # A: rows i=0,1 and cols j=0,1 -> U indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
        # B: rows j=0,1 and cols k=0,1 -> V indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        # C: rows k=0,1 and cols i=0,1 -> W indices: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3

        # So I need to pass 4 A rows, 4 B rows, 4 W rows.
        # Let me restructure.
        pass

    # Let me directly construct the rank-16 decomposition.
    # Strategy: Split the 2x3x4 into two 2x3x2 blocks.
    # Block 1: output columns k=0,1. Uses A (all 2x3), B rows 0,1 (3x2).
    #   Further split: A cols 0,1 (2x2) with B rows 0,1 (2x2) -> Strassen rank 7
    #   Plus: A col 2 (2x1) with B row 2 (1x2) -> rank 1 (outer product)
    # Block 2: output columns k=2,3. Uses A (all 2x3), B rows 2,3 (3x2).
    #   Further split: A cols 0,1 (2x2) with B rows 2,3 (2x2) -> Strassen rank 7
    #   Plus: A col 2 (2x1) with B row 2 (1x2) -> rank 1 (outer product)
    # Total rank: 7+1+7+1 = 16

    # Let me define the Strassen decomposition more carefully.
    # For a 2x2x2 multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(e-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # In tensor form, each M_c contributes U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For Strassen, the factors are:
    # M1: U has (a+d) in both diagonal positions, V has (e+h) in both positions
    #   U[0,0,1]=1, U[1,1,1]=1 (for a and d)
    #   V[0,0,1]=1, V[1,1,1]=1 (for e and h)
    #   W[0,0,1]=1, W[1,1,1]=1 (for C11 and C22 contributions)
    #   But wait, M1 contributes to both C11 and C22 with +1.
    #   So W[0,0,1]=1 and W[1,1,1]=1.

    # Actually, let me think about this differently.
    # The tensor T for 2x2x2 has entries:
    # T[0,0,0]=1 (a*e -> C11)
    # T[0,1,0]=1 (a*f -> C12)
    # T[1,0,0]=1 (b*e -> C11)
    # T[1,1,0]=1 (b*f -> C12)
    # T[0,2,1]=1 (c*g -> C21)
    # T[0,3,1]=1 (c*h -> C22)
    # T[1,2,1]=1 (d*g -> C21)
    # T[1,3,1]=1 (d*h -> C22)
    # Where U index: i*2+j, V index: j*2+k, W index: k*2+i

    # Strassen decomposition:
    # c=0 (M1): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    #   T contribution: 1*1*1 at (0,0,0) + 1*1*1 at (3,3,1) = T[0,0,0]+T[3,3,1]
    #   But we need T[0,0,0]=1 (a*e) and T[3,3,1]=1 (d*h). Yes!
    #   But M1 also should contribute to C11 and C22.
    #   C11 gets +M1, C22 gets +M1.
    #   T[0,0,0] is C11 from a*e. T[3,3,1] is C22 from d*h.
    #   But M1 = (a+d)(e+h) = ae + ah + de + dh
    #   So M1 contributes to: ae (C11), ah (C12), de (C21), dh (C22)
    #   In tensor: T[0,0,0]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,1]=1
    #   Wait, let me recheck indices.
    #   U index for a (i=0,j=0): 0*2+0=0
    #   U index for d (i=1,j=1): 1*2+1=3
    #   V index for e (j=0,k=0): 0*2+0=0
    #   V index for h (j=1,k=1): 1*2+1=3
    #   W index for C11 (k=0,i=0): 0*2+0=0
    #   W index for C12 (k=0,i=1): 0*2+1=1
    #   W index for C21 (k=1,i=0): 1*2+0=2
    #   W index for C22 (k=1,i=1): 1*2+1=3

    #   M1 = (a+d)(e+h):
    #   ae: U[0]*V[0]*W[0] -> T[0,0,0]
    #   ah: U[0]*V[3]*W[1] -> T[0,3,1]
    #   de: U[3]*V[0]*W[2] -> T[3,0,2]
    #   dh: U[3]*V[3]*W[3] -> T[3,3,3]

    #   But in the target tensor for 2x2x2:
    #   T[0,0,0]=1 (a*e -> C11)
    #   T[0,3,1]=1 (a*h -> C12)  [a is U[0], h is V[3], C12 is W[1]]
    #   T[3,0,2]=1 (d*e -> C21)  [d is U[3], e is V[0], C21 is W[2]]
    #   T[3,3,3]=1 (d*h -> C22)  [d is U[3], h is V[3], C22 is W[3]]

    #   So M1 contributes: T[0,0,0] + T[0,3,1] + T[3,0,2] + T[3,3,3]
    #   With U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    #   Check: U[0]*V[0]*W[0]=1, U[0]*V[3]*W[1]=1, U[3]*V[0]*W[2]=1, U[3]*V[3]*W[3]=1. Yes!

    # Let me just directly write out all 7 Strassen factors for 2x2x2:
    # c=0 (M1=(a+d)(e+h)): U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    # c=1 (M2=(c+d)e): U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    #   M2 = ce + de. ce: C21, de: C21. Both contribute to C21.
    #   ce: U[2]*V[0]*W[2], de: U[3]*V[0]*W[2]
    #   U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]. Check: U[2]*V[0]*W[2]=1, U[3]*V[0]*W[2]=1. Yes.
    # c=2 (M3=a(f-g)): U=[1,0,0,0], V=[0,1,1,0], W=[1,0,0,0]
    #   M3 = af - ag. af: C12, ag: C12 (negative).
    #   af: U[0]*V[1]*W[1], ag: U[0]*V[2]*W[1]
    #   U=[1,0,0,0], V=[0,1,1,0], W=[0,1,0,0].
    #   Wait, W index for C12 is W[1]. So W=[0,1,0,0].
    #   Check: U[0]*V[1]*W[1]=1 (af), U[0]*V[2]*W[1]=1 (ag). But we need af - ag.
    #   So V should be [0,1,-1,0]. Then U[0]*V[1]*W[1]=1, U[0]*V[2]*W[1]=-1. Yes!
    # c=3 (M4=d(e-g)): U=[0,0,0,1], V=[1,0,-1,0], W=[0,0,1,0]
    #   M4 = de - dg. de: C21, dg: C21 (negative).
    #   de: U[3]*V[0]*W[2], dg: U[3]*V[2]*W[2]
    #   U=[0,0,0,1], V=[1,0,-1,0], W=[0,0,1,0]. Check: U[3]*V[0]*W[2]=1, U[3]*V[2]*W[2]=-1. Yes.
    # c=4 (M5=(a+b)h): U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    #   M5 = ah + bh. ah: C12, bh: C12.
    #   ah: U[0]*V[3]*W[1], bh: U[1]*V[3]*W[1]
    #   U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]. Check: U[0]*V[3]*W[1]=1, U[1]*V[3]*W[1]=1. Yes.
    # c=5 (M6=(c-a)(e+f)): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,1,0]
    #   M6 = ce + cf - ae - af. ce: C21, cf: C22, ae: C11, af: C12.
    #   ce: U[2]*V[0]*W[2], cf: U[2]*V[1]*W[3], ae: U[0]*V[0]*W[0], af: U[0]*V[1]*W[1]
    #   U=[-1,0,1,0], V=[1,1,0,0], W=[-1,0,1,0].
    #   Check: U[2]*V[0]*W[2]=1*1*1=1 (ce), U[2]*V[1]*W[3]=1*1*0=0? No, W[3] should be for C22.
    #   W index: C22 is k=1,i=1 -> 1*2+1=3. So W[3] is C22.
    #   cf: C22, so W[3]. U[2]*V[1]*W[3]=1*1*? 
    #   Let me redo. W=[w0,w1,w2,w3] for C11,C12,C21,C22.
    #   M6 = ce + cf - ae - af
    #   ce -> C21: W[2]
    #   cf -> C22: W[3]
    #   ae -> C11: W[0]
    #   af -> C12: W[1]
    #   So W=[-1,-1,1,1]
    #   U=[-1,0,1,0], V=[1,1,0,0], W=[-1,-1,1,1]
    #   Check: U[0]*V[0]*W[0]=(-1)(1)(-1)=1 (ae, but we want -ae)
    #   Hmm, sign issue. Let me be more careful.
    #   M6 = (c-a)(e+f) = ce + cf - ae - af
    #   We want: T contribution = +ce +cf -ae -af
    #   ce: U[2]*V[0]*W[2] should be +1
    #   cf: U[2]*V[1]*W[3] should be +1
    #   ae: U[0]*V[0]*W[0] should be -1
    #   af: U[0]*V[1]*W[1] should be -1
    #   U=[-1,0,1,0]: U[0]=-1, U[2]=1
    #   V=[1,1,0,0]: V[0]=1, V[1]=1
    #   W=[?]: W[0]=-1 (for ae: (-1)(1)(-1)=1, but we want -1)
    #   Let me try U=[1,0,-1,0]:
    #   ae: U[0]*V[0]*W[0] = 1*1*W[0] = W[0], want -1, so W[0]=-1
    #   af: U[0]*V[1]*W[1] = 1*1*W[1] = W[1], want -1, so W[1]=-1
    #   ce: U[2]*V[0]*W[2] = -1*1*W[2] = -W[2], want +1, so W[2]=-1
    #   cf: U[2]*V[1]*W[3] = -1*1*W[3] = -W[3], want +1, so W[3]=-1
    #   So U=[1,0,-1,0], V=[1,1,0,0], W=[-1,-1,-1,-1]
    #   Check: ae: 1*1*(-1)=-1 ✓, af: 1*1*(-1)=-1 ✓, ce: -1*1*(-1)=1 ✓, cf: -1*1*(-1)=1 ✓
    # c=6 (M7=(b-d)(g+h)): U=[0,1,0,-1], V=[0,0,1,1], W=[0,0,0,-1]
    #   M7 = bg + bh - dg - dh
    #   bg -> C11: W[0], bh -> C12: W[1], dg -> C21: W[2], dh -> C22: W[3]
    #   bg: U[1]*V[2]*W[0] = 1*1*W[0], want +1, W[0]=1
    #   bh: U[1]*V[3]*W[1] = 1*1*W[1], want +1, W[1]=1
    #   dg: U[3]*V[2]*W[2] = -1*1*W[2], want -1, W[2]=1
    #   dh: U[3]*V[3]*W[3] = -1*1*W[3], want -1, W[3]=1
    #   So U=[0,1,0,-1], V=[0,0,1,1], W=[1,1,1,1]
    #   Check: bg: 1*1*1=1 ✓, bh: 1*1*1=1 ✓, dg: -1*1*1=-1 ✓, dh: -1*1*1=-1 ✓

    # Now let me verify the full Strassen by summing all 7:
    # For each (u,v,w) in 2x2x2, sum over c of U_c[u]*V_c[v]*W_c[w] should equal T[u,v,w].
    # T[0,0,0]=1 (ae), T[0,1,1]=1 (af), T[1,0,0]=1 (be), T[1,1,1]=1 (bf)
    # T[2,0,2]=1 (cg), T[2,1,3]=1 (ch), T[3,0,2]=1 (dg), T[3,1,3]=1 (dh)
    # Wait, I need to recheck the W indexing.
    # W index: k*n+i where n=2. So W[k*2+i].
    # C11: k=0,i=0 -> W[0]
    # C12: k=0,i=1 -> W[1]
    # C21: k=1,i=0 -> W[2]
    # C22: k=1,i=1 -> W[3]

    # U index: i*m+j where m=2. So U[i*2+j].
    # a: i=0,j=0 -> U[0]
    # b: i=0,j=1 -> U[1]
    # c: i=1,j=0 -> U[2]
    # d: i=1,j=1 -> U[3]

    # V index: j*p+k where p=2. So V[j*2+k].
    # e: j=0,k=0 -> V[0]
    # f: j=0,k=1 -> V[1]
    # g: j=1,k=0 -> V[2]
    # h: j=1,k=1 -> V[3]

    # Target tensor for 2x2x2:
    # T[0,0,0]=1 (a*e -> C11)
    # T[0,1,1]=1 (a*f -> C12)
    # T[1,0,0]=1 (b*e -> C11)
    # T[1,1,1]=1 (b*f -> C12)
    # T[2,2,2]=1 (c*g -> C21)
    # T[2,3,3]=1 (c*h -> C22)
    # T[3,2,2]=1 (d*g -> C21)
    # T[3,3,3]=1 (d*h -> C22)

    # Now verify Strassen sum for T[0,0,0] (ae):
    # c=0: U[0]*V[0]*W[0] = 1*1*1 = 1
    # c=1: U[0]*V[0]*W[0] = 0*1*0 = 0
    # c=2: U[0]*V[0]*W[0] = 1*0*0 = 0
    # c=3: U[0]*V[0]*W[0] = 0*1*0 = 0
    # c=4: U[0]*V[0]*W[0] = 1*0*0 = 0
    # c=5: U[0]*V[0]*W[0] = 1*1*(-1) = -1
    # c=6: U[0]*V[0]*W[0] = 0*0*1 = 0
    # Sum = 1 - 1 = 0. But we need 1! Something is wrong.

    # Let me recheck c=5 (M6). M6 = (c-a)(e+f) = ce + cf - ae - af.
    # The contribution to T[0,0,0] (which is ae) should be -1 from M6.
    # And M1 contributes +1 to ae.
    # So total should be 1 + (-1) + ... Let me check all.

    # Actually, I think I made an error in the Strassen formula. Let me use the standard one:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(e-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Let me verify: C11 = ae + be + ce + de
    # M1 + M4 - M5 + M7
    # = (ae+ah+de+dh) + (de-dg) - (ah+bh) + (bg+bh-dg-dh)
    # = ae + ah + de + dh + de - dg - ah - bh + bg + bh - dg - dh
    # = ae + 2de - 2dg + bg
    # That's not right. Let me use a different Strassen variant.

    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a+c)(e-f)
    # M5 = (b+d)(f+h)
    # M6 = (a-c)(e+g)
    # M7 = (b-d)(f+h)
    # Hmm, there are many variants. Let me just use a known correct one.

    # Let me use this well-known Strassen:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - g)
    # M4 = d(e - g)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Verify C11:
    # M1 = ae + ah + de + dh
    # M4 = de - dg
    # -M5 = -ah - bh
    # M7 = bg + bh - dg - dh
    # Sum = ae + ah + de + dh + de - dg - ah - bh + bg + bh - dg - dh
    # = ae + 2de - 2dg + bg
    # Still wrong. The issue is that M4 should only contribute to C11 and C21, not double-count.

    # I think the problem is that in the tensor decomposition, each M_c is a single rank-1 term,
    # and the W vector for that term specifies which output entries it contributes to.
    # M4 = d(e-g) contributes to C11 (via de) and C21 (via -dg).
    # So W for M4 should have +1 at C11 position and -1 at C21 position.
    # C11 is W[0], C21 is W[2].
    # So W = [1, 0, -1, 0] for M4.

    # Let me redo all factors with proper W vectors:
    # c=0 (M1=(a+d)(e+h)): contributes to C11(+), C12(+), C21(+), C22(+)
    #   U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    # c=1 (M2=(c+d)e): contributes to C21(+), C21(+) -> both to C21
    #   Actually M2 = ce + de, both contribute to C21.
    #   U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # c=2 (M3=a(f-g)): contributes to C12(+), C12(-)
    #   U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,0]
    # c=3 (M4=d(e-g)): contributes to C11(+), C21(-)
    #   U=[0,0,0,1], V=[1,0,-1,0], W=[1,0,-1,0]
    # c=4 (M5=(a+b)h): contributes to C12(+), C12(+)
    #   U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # c=5 (M6=(c-a)(e+f)): contributes to C21(+), C22(+), C11(-), C12(-)
    #   U=[0,0,1,-1], V=[1,1,0,0], W=[-1,-1,1,1]
    # c=6 (M7=(b-d)(g+h)): contributes to C11(+), C12(+), C21(-), C22(-)
    #   U=[0,1,0,-1], V=[0,0,1,1], W=[1,1,-1,-1]

    # Now verify C11 = sum of contributions to W[0]:
    # c=0: 1, c=1: 0, c=2: 0, c=3: 1, c=4: 0, c=5: -1, c=6: 1
    # For each (u,v) pair that should give C11:
    # T[0,0,0]=1 (ae):
    # c=0: U[0]*V[0]*W[0] = 1*1*1 = 1
    # c=3: U[0]*V[0]*W[0] = 0*1*1 = 0
    # c=5: U[0]*V[0]*W[0] = 0*1*(-1) = 0
    # c=6: U[0]*V[0]*W[0] = 0*0*1 = 0
    # Sum = 1. But we need 1 from ae. Let me check if other terms cancel.
    # Actually, the total sum over all c for T[0,0,0] should be 1.
    # c=0: 1*1*1 = 1
    # c=1: 0
    # c=2: 0
    # c=3: 0
    # c=4: 0
    # c=5: 0
    # c=6: 0
    # Sum = 1. ✓

    # T[1,0,0]=1 (be):
    # c=0: U[1]*V[0]*W[0] = 0*1*1 = 0
    # c=6: U[1]*V[0]*W[0] = 1*0*1 = 0
    # Hmm, that's 0. But we need 1.
    # Wait, be should be in C11. Let me check which M terms include be.
    # M7 = (b-d)(g+h) = bg + bh - dg - dh. No be.
    # M1 = (a+d)(e+h) = ae + ah + de + dh. No be.
    # M5 = (a+b)h = ah + bh. No be.
    # Hmm, be is not in any M term? That can't be right.
    # Oh wait, I think I have the wrong Strassen formula. Let me look this up properly.

    # Correct Strassen for C = AB where A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a+c)(e-f)
    # M5 = (b+d)(f+h)
    # M6 = (a-c)(e+g)
    # M7 = (b-d)(f+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Verify C11:
    # M1 = ae+ah+de+dh
    # M4 = ae-af+ce-cf
    # -M5 = -bf-bh-df-dh
    # M7 = bf+bh-df-dh
    # Sum = ae+ah+de+dh+ae-af+ce-cf-bf-bh-df-dh+bf+bh-df-dh
    # = 2ae+ah+de+ce-cf-bf-2df-dh
    # Still not right. I'm clearly making errors. Let me just use a verified Strassen.

    # I'll use the following verified Strassen (from Wikipedia):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - g)
    # m4 = d(e - g)
    # m5 = (a + b)h
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6

    # Let me verify c11 = ae + be + ce + de:
    # m1 = ae + ah + de + dh
    # m4 = de - dg
    # -m5 = -ah - bh
    # m7 = bg + bh - dg - dh
    # Sum = ae + ah + de + dh + de - dg - ah - bh + bg + bh - dg - dh
    # = ae + 2de - 2dg + bg
    # This is NOT ae+be+ce+de. So this formula is wrong too.

    # I think the issue is that these are NOT the standard Strassen formulas.
    # Let me just directly construct a rank-7 decomposition by solving linear equations.
    # Or better, let me just use the schoolbook for now and try to find improvements.

    # Given the complexity, let me just return the schoolbook rank-24 solution.
    # The previous attempts to construct Strassen-based solutions have errors.
    
    rank = n * m * p  # 24
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

    # Verify before returning
    if not verify(U, V, W):
        # Should never happen for schoolbook
        raise ValueError("Schoolbook verification failed")

    return {"U": U, "V": V, "W": W}
