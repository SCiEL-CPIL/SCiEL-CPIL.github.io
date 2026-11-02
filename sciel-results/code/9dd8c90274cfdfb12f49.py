"""Matrix multiplication 3x3x7 - tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The decomposition has the form:
    For each (i,j,k) where i in [0,n), j in [0,m), k in [0,p):
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    
    # Strategy: Decompose into 3x3x3 (columns 0-2) + 3x3x4 (columns 3-6)
    # For 3x3x3, use Strassen-based block decomposition to get rank 26
    # For 3x3x4, use schoolbook (rank 36) or try to find better
    
    # Actually, let me just use schoolbook for the whole thing first to ensure correctness,
    # then try to improve with Strassen for the 3x3x3 part.
    
    # Schoolbook gives rank = 3*3*7 = 63
    # With Strassen for the 3x3x3 part: 26 + 3*3*4 = 26 + 36 = 62
    
    # Let me implement Strassen for 3x3x3 properly.
    # 
    # For 3x3 * 3x3, partition as:
    # A = [[A00(2x2), a0(2x1)], [a1^T(1x2), a2(1x1)]]
    # B = [[B00(2x2), b0(2x1)], [b1^T(1x2), b2(1x1)]]
    # C = A*B = [[C00(2x2), c0(2x1)], [c1^T(1x2), c2(1x1)]]
    #
    # C00 = A00*B00 + a0*b1^T  (2x2)
    # c0  = A00*b0 + a0*b2     (2x1)
    # c1^T = a1^T*B00 + a2*b1^T (1x2)
    # c2  = a1^T*b0 + a2*b2     (1x1)
    #
    # Strassen for A00*B00 (2x2*2x2): rank 7
    # a0*b1^T (2x1 * 1x2): rank 4 (schoolbook, optimal)
    # A00*b0 (2x2 * 2x1): rank 4 (schoolbook)
    # a0*b2 (2x1 * 1x1): rank 2
    # a1^T*B00 (1x2 * 2x2): rank 4
    # a2*b1^T (1x1 * 1x2): rank 2
    # a1^T*b0 (1x2 * 2x1): rank 2
    # a2*b2 (1x1 * 1x1): rank 1
    # Total: 7+4+4+2+4+2+2+1 = 26
    
    # Now I need to express each of these as rank-1 terms in the tensor decomposition.
    # 
    # For a term like A00*B00 using Strassen:
    # P1 = (A00[0,0]+A00[1,1])*(B00[0,0]+B00[1,1])
    # In tensor terms, this is ONE rank-1 term where:
    # U has 1 at A[0,0] and A[1,1] (indices 0 and 4 in U, since U[i*m+j])
    # V has 1 at B[0,0] and B[1,1] (indices 0 and p in V, since V[j*p+k])
    # W has contributions to C[0,0] and C[1,1]
    #
    # Wait, I need to be more careful. Let me think about what a rank-1 term means.
    #
    # A rank-1 term r is defined by vectors u_r (dim n*m), v_r (dim m*p), w_r (dim p*n).
    # Its contribution to the (i,j,k) entry of the tensor is:
    #   u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    #
    # For the decomposition to be correct, we need:
    #   sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1  for all i,j,k
    #
    # For schoolbook, each term r corresponds to a specific (i,j,k) triple:
    #   u_r[i*m+j] = 1 if (i,j) matches, else 0
    #   v_r[j*p+k] = 1 if (j,k) matches, else 0
    #   w_r[k*n+i] = 1 if (k,i) matches, else 0
    #
    # For Strassen, a term like P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) contributes to C[0,0] and C[1,1].
    # This is represented as:
    #   u_r has 1 at positions (0,0) and (1,1) in A, i.e., U[0] = 1, U[4] = 1
    #   v_r has 1 at positions (0,0) and (1,1) in B, i.e., V[0] = 1, V[p] = 1
    #   w_r has 1 at positions (0,0) and (1,1) in C, i.e., W[0] = 1, W[n] = 1
    #
    # But wait, this doesn't work because the product u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # would give non-zero values for many (i,j,k) triples, not just the ones we want.
    #
    # Let me reconsider. For Strassen 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    # This product appears in C00 = P1 + P4 - P5 + P7 and C11 = P1 - P2 + P3 + P6
    #
    # In the tensor decomposition, P1 is a single rank-1 term. The way it works:
    # The term "computes" the product (A00+A11)*(B00+B11) and places it into C00 and C11.
    # 
    # For this to work as a rank-1 term:
    # u_r: 1 at A00 (index 0), 1 at A11 (index 4)
    # v_r: 1 at B00 (index 0), 1 at B11 (index p=3)
    # w_r: 1 at C00 (index 0), 1 at C11 (index n=3)
    #
    # Now check: for (i,j,k) = (0,0,0): u[0]*v[0]*w[0] = 1*1*1 = 1 ✓ (C00 gets A00*B00)
    # For (i,j,k) = (0,0,1): u[0]*v[1]*w[3] = 1*0*1 = 0 (V[1] is B[0,1], which is 0)
    # For (i,j,k) = (0,1,0): u[1]*v[3]*w[0] = 0*1*1 = 0 (U[1] is A[0,1], which is 0)
    # For (i,j,k) = (0,1,1): u[1]*v[4]*w[3] = 0*0*1 = 0
    # For (i,j,k) = (1,0,0): u[3]*v[0]*w[3] = 0*1*1 = 0 (U[3] is A[1,0], which is 0)
    # For (i,j,k) = (1,0,1): u[3]*v[1]*w[0] = 0*0*1 = 0
    # For (i,j,k) = (1,1,0): u[4]*v[3]*w[0] = 1*1*1 = 1 ✓ (C11 gets A11*B11)
    # For (i,j,k) = (1,1,1): u[4]*v[4]*w[3] = 1*0*1 = 0 (V[4] is B[1,1]... wait)
    #
    # Hmm, V[j*p+k] where j=1, k=1 gives V[1*3+1] = V[4]. But B[1,1] is at V[1*3+1] = V[4].
    # So v_r[4] should be 1 for B[1,1]. Let me redo:
    # v_r: 1 at B00 (V[0*3+0]=V[0]), 1 at B11 (V[1*3+1]=V[4])
    # So v_r[0] = 1, v_r[4] = 1
    #
    # For (i,j,k) = (1,1,1): u[4]*v[4]*w[3] = 1*1*1 = 1
    # But this should contribute to C[1,1] from A[1,1]*B[1,1], which is correct!
    # Wait, but C[1,1] = sum_k A[1,k]*B[k,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # So (i,j,k)=(1,1,1) means A[1,1]*B[1,1] contributes to C[1,1]. That's correct.
    #
    # But we also get (i,j,k)=(0,0,0): A[0,0]*B[0,0] contributes to C[0,0]. Correct.
    # And (i,j,k)=(1,1,1): A[1,1]*B[1,1] contributes to C[1,1]. Correct.
    #
    # But what about the cross terms? (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # The term should contribute A00*B00 to C00, A00*B11 to C01, A11*B00 to C10, A11*B11 to C11.
    # But in my construction, it only contributes to C00 and C11, not C01 and C10.
    #
    # This is the issue. Strassen's algorithm works by having the intermediate products
    # contribute to specific output entries with specific coefficients, and the final
    # combination cancels out the unwanted terms.
    #
    # So for P1 = (A00+A11)(B00+B11):
    # It contributes to C00 with coefficient +1, and to C11 with coefficient +1.
    # But the actual expansion A00*B00 + A00*B11 + A11*B00 + A11*B11 needs to be
    # distributed correctly.
    #
    # I think the correct way to view this is:
    # P1 as a rank-1 term has:
    # u_r: 1 at A00, 1 at A11
    # v_r: 1 at B00, 1 at B11  
    # w_r: 1 at C00, 1 at C11
    #
    # The contribution to (i,j,k) is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # For (0,0,0): 1*1*1 = 1 → A00*B00 → C00 ✓
    # For (0,0,1): 1*0*0 = 0 → (B01 not in v_r)
    # For (0,1,0): 0*1*1 = 0 → (A01 not in u_r)
    # For (0,1,1): 0*0*0 = 0
    # For (1,0,0): 0*1*1 = 0 → (A10 not in u_r)
    # For (1,0,1): 0*0*0 = 0
    # For (1,1,0): 1*1*1 = 1 → A11*B11 → C11 ✓
    # For (1,1,1): 1*1*1 = 1 → A11*B11 → C11 ✓ (duplicate?)
    #
    # Wait, I'm confusing myself with the indices. Let me be very explicit.
    #
    # For 2x2x2 (n=m=p=2):
    # A indices: (0,0)->U[0], (0,1)->U[1], (1,0)->U[2], (1,1)->U[3]
    # B indices: (0,0)->V[0], (0,1)->V[1], (1,0)->V[2], (1,1)->V[3]
    # C indices: (0,0)->W[0], (0,1)->W[1], (1,0)->W[2], (1,1)->W[3]
    #
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # u_r: U[0]=1, U[3]=1
    # v_r: V[0]=1, V[3]=1
    # w_r: W[0]=1, W[3]=1
    #
    # Contribution to (i,j,k):
    # (0,0,0): U[0]*V[0]*W[0] = 1*1*1 = 1
    # (0,0,1): U[0]*V[1]*W[2] = 1*0*0 = 0
    # (0,1,0): U[1]*V[2]*W[0] = 0*0*1 = 0
    # (0,1,1): U[1]*V[3]*W[2] = 0*1*0 = 0
    # (1,0,0): U[2]*V[0]*W[1] = 0*1*0 = 0
    # (1,0,1): U[2]*V[1]*W[3] = 0*0*1 = 0
    # (1,1,0): U[3]*V[2]*W[1] = 1*0*0 = 0
    # (1,1,1): U[3]*V[3]*W[3] = 1*1*1 = 1
    #
    # So P1 contributes 1 to (0,0,0) and 1 to (1,1,1).
    # (0,0,0) means A[0,0]*B[0,0] → C[0,0] ✓
    # (1,1,1) means A[1,1]*B[1,1] → C[1,1] ✓
    #
    # But what about A[0,0]*B[1,1] → C[0,1] and A[1,1]*B[0,0] → C[1,0]?
    # These are NOT captured by P1 alone. They must be captured by other Strassen terms
    # with appropriate coefficients that cancel out.
    #
    # Let me verify with the full Strassen 2x2x2 decomposition:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Let me expand C00:
    # P1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # P4 = A11*B01 - A11*B00
    # P5 = A00*B11 + A01*B11
    # P7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    #
    # C00 = P1 + P4 - P5 + P7
    #     = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    #       + (A11*B01 - A11*B00)
    #       - (A00*B11 + A01*B11)
    #       + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    #     = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #       + A11*B01 - A11*B00
    #       - A00*B11 - A01*B11
    #       + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #     = A00*B00 + A11*B01 + A01*B10
    #     = A00*B00 + A01*B10 + A11*B01  ... wait, that's not right.
    #
    # C00 should be A00*B00 + A01*B10.
    # Let me recheck:
    # C00 = A00*B00 + A01*B10
    #
    # From expansion:
    # A00*B00: +1 (from P1)
    # A00*B11: +1 (P1) - 1 (P5) = 0 ✓
    # A11*B00: +1 (P1) - 1 (P4) = 0 ✓
    # A11*B11: +1 (P1) - 1 (P7) = 0 ✓
    # A11*B01: +1 (P4) ✓ (but this should be 0!)
    # A01*B11: -1 (P5) + 1 (P7) = 0 ✓
    # A01*B10: +1 (P7) ✓
    # A11*B10: -1 (P7) (should be 0)
    #
    # Hmm, I get A00*B00 + A11*B01 + A01*B10 - A11*B10, which is not correct.
    #
    # I think I made an error in the Strassen formula. Let me look it up properly.
    #
    # Standard Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(g-h)
    # P4 = d(f-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Let me verify C00 = P1 + P4 - P5 + P7:
    # P1 = (a+d)(e+h) = ae + ah + de + dh
    # P4 = d(f-e) = df - de
    # P5 = (a+b)h = ah + bh
    # P7 = (b-d)(g+h) = bg + bh - dg - dh
    #
    # C00 = ae + ah + de + dh + df - de - ah - bh + bg + bh - dg - dh
    #     = ae + df + bg - dg
    #     = a*e + b*g + d*f - d*g
    #
    # But C00 should be a*e + b*g. So there's a -d*g term that shouldn't be there.
    #
    # I think the standard Strassen formula I'm using is wrong. Let me look it up again.
    #
    # Correct Strassen:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a)(g-h)
    # P4 = (d)(f-e)
    # P5 = (a+b)(h)
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Let me recompute C00:
    # P1 = ae + ah + de + dh
    # P4 = df - de
    # P5 = ah + bh
    # P7 = bg + bh - dg - dh
    #
    # C00 = (ae + ah + de + dh) + (df - de) - (ah + bh) + (bg + bh - dg - dh)
    #     = ae + ah + de + dh + df - de - ah - bh + bg + bh - dg - dh
    #     = ae + df + bg - dg
    #
    # Still getting -dg. This doesn't seem right.
    #
    # Let me try a different source. The correct Strassen decomposition is:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(g-h)
    # P4 = d(f-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C00 = P1 - P5 + P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Let me verify C11 = P1 - P2 + P3 + P6:
    # P1 = ae + ah + de + dh
    # P2 = ce + de
    # P3 = ag - ah
    # P6 = (c-a)(e+f) = ce + cf - ae - af
    #
    # C11 = (ae + ah + de + dh) - (ce + de) + (ag - ah) + (ce + cf - ae - af)
    #     = ae + ah + de + dh - ce - de + ag - ah + ce + cf - ae - af
    #     = dh + ag + cf - af
    #     = d*h + a*g + c*f - a*f
    #
    # But C11 should be c*f + d*h. So there's an extra a*g - a*f term.
    #
    # I'm clearly making errors. Let me just use a known correct implementation.
    #
    # Actually, I think the issue is that I'm confusing which entries go where.
    # Let me just use schoolbook for now to ensure correctness, and accept rank 63.
    # Then I can try to verify Strassen separately.
    
    # For now, let's use schoolbook: rank = 3*3*7 = 63
    rank = 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # With schoolbook, each (i,j,k) has exactly one term where all three are 1, so sum = 1. ✓
    
    # Convert to list of lists for the return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
