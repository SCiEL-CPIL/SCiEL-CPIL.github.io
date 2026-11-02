import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Schoolbook decomposition of rank n*m*p = 18
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    # Try to find a rank-17 decomposition by identifying a linear dependency
    # among the 18 schoolbook terms.
    #
    # The multiplication tensor T has entries T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We can view the 18 basis tensors e_{ijk} = U_{ijk} ⊗ V_{ijk} ⊗ W_{ijk}
    # where U_{ijk}[i',j'] = δ_{i,i'}δ_{j,j'}, etc.
    #
    # A rank-17 decomposition exists if we can find coefficients c_{ijk} not all zero
    # such that sum_{i,j,k} c_{ijk} * e_{ijk} = 0 (as a tensor).
    #
    # This is equivalent to: for all (i',j',k'), sum_{i,j,k} c_{ijk} δ_{i,i'}δ_{j,j'}δ_{k,k'} = 0
    # which simplifies to c_{i'j'k'} = 0 for all i',j',k'.
    #
    # Wait, that's not right. The tensor decomposition is:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where T is the multiplication tensor with T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    #
    # Actually, the multiplication tensor for C = A*B has:
    # T[i,k,j] = 1 (using the convention that C[i,k] = sum_j A[i,j]B[j,k])
    #
    # In the problem's convention: W[k*n+i, r] so the output index is k*n+i.
    # U has shape (n*m, rank) with row index i*m+j
    # V has shape (m*p, rank) with row index j*p+k
    # W has shape (p*n, rank) with row index k*n+i
    #
    # So the tensor is: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # This is a 3-way tensor of shape (n*m) x (m*p) x (p*n) = 6 x 9 x 6
    #
    # For rank reduction, we need to find if the 18 schoolbook terms are linearly dependent.
    # The r-th schoolbook term has:
    # U_r[i*m+j] = 1 if (i,j) = (i_r, j_r), else 0
    # V_r[j*p+k] = 1 if (j,k) = (j_r, k_r), else 0
    # W_r[k*n+i] = 1 if (k,i) = (k_r, i_r), else 0
    #
    # A linear combination sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0 means:
    # For all (a, b, c) in the tensor indices:
    # sum_r c_r * U_r[a] * V_r[b] * W_r[c] = 0
    #
    # U_r[a] is nonzero only when a = i_r*m + j_r, in which case it's 1.
    # V_r[b] is nonzero only when b = j_r*p + k_r, in which case it's 1.
    # W_r[c] is nonzero only when c = k_r*n + i_r, in which case it's 1.
    #
    # So the condition is: for all (a, b, c),
    # sum_{r: i_r*m+j_r=a, j_r*p+k_r=b, k_r*n+i_r=c} c_r = 0
    #
    # For a given (a, b, c), let's see which r satisfy all three conditions.
    # a = i*m + j => i = a//m, j = a%m
    # b = j'*p + k => j' = b//p, k = b%p
    # c = k'*n + i' => k' = c//n, i' = c%n
    #
    # For consistency: j = j' and i = i' and k = k'
    # So the only r that contributes is r = (i, j, k) where i = a//m = c%n, j = a%m = b//p, k = b%p = c//n
    #
    # If these are consistent (i.e., a//m == c%n, a%m == b//p, b%p == c//n), then
    # the condition is c_{i,j,k} = 0.
    #
    # If they're inconsistent, the sum is empty and the condition is trivially satisfied.
    #
    # So actually, the 18 schoolbook terms are linearly INDEPENDENT! There's no nontrivial
    # linear combination that gives zero. This means we can't just drop one term.
    #
    # We need a different approach. Let's use the known rank-17 construction.
    #
    # Based on the subproblem decomposition:
    # Split A = [A1 | A2] where A1 is 2x2 (columns 0,1) and A2 is 2x1 (column 2)
    # Split B = [B1; B2] where B1 is 2x3 (rows 0,1) and B2 is 1x3 (row 2)
    #
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1: 2x2 @ 2x3
    # A2 @ B2: 2x1 @ 1x3
    #
    # For A1 @ B1, we can use a Strassen-like approach.
    # Let A1 = [[a, b], [c, d]], B1 = [[e, f, g], [h, i, j]]
    #
    # We need to compute:
    # C11 = ae + bh
    # C12 = af + bi
    # C13 = ag + bj
    # C21 = ce + dh
    # C22 = cf + di
    # C23 = cg + dj
    #
    # Using Strassen for the 2x2 part (columns 0,1 of B1):
    # m1 = (a+d)(e+h) = ae+ah+de+dh
    # m2 = (c+d)e = ce+de
    # m3 = a(f-h) = af-ah
    # m4 = d(g-e) = dg-de  -- wait, this doesn't work for 2x3
    #
    # Let me use a different approach. For 2x2 @ 2x3, we can think of it as:
    # Compute A1 @ B1[:,0:2] using Strassen (7 mults) to get a 2x2 result
    # Then compute A1 @ B1[:,2] using 4 mults (schoolbook)
    # Total: 7 + 4 = 11 mults for A1 @ B1
    #
    # For A2 @ B2: 2x1 @ 1x3, this is just:
    # C13 += a2*e2, C23 += a3*e2 where A2 = [a2, a3], B2 = [e2, f2, g2]
    # Wait, A2 is 2x1: [a2; a3], B2 is 1x3: [e2, f2, g2]
    # C13 += a2*e2
    # C14... wait, B2 has 3 columns.
    # A2 @ B2 = [a2*e2, a2*f2, a2*g2; a3*e2, a3*f2, a3*g2]
    # This requires 6 mults (schoolbook) or we can do it with... well, it's just 6 scalar mults.
    #
    # Total: 11 + 6 = 17.
    #
    # Let me construct this explicitly.
    #
    # A = [[a0, a1, a2], [a3, a4, a5]]  (2x3)
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]  (3x3)
    #
    # A1 = [[a0, a1], [a3, a4]]  (2x2, columns 0,1 of A)
    # A2 = [[a2], [a5]]  (2x1, column 2 of A)
    # B1 = [[b0, b1, b2], [b3, b4, b5]]  (2x3, rows 0,1 of B)
    # B2 = [[b6, b7, b8]]  (1x3, row 2 of B)
    #
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1:
    # Let's use Strassen for the first two columns of B1, and schoolbook for the third.
    #
    # Strassen for A1 @ B1[:,0:2]:
    # A1 = [[a0, a1], [a3, a4]], B1[:,0:2] = [[b0, b1], [b3, b4]]
    #
    # m1 = (a0+a4)(b0+b3) = a0*b0 + a0*b3 + a4*b0 + a4*b3
    # m2 = (a3+a4)*b0 = a3*b0 + a4*b0
    # m3 = a0*(b1-b3) = a0*b1 - a0*b3
    # m4 = a4*(b4-b0) = a4*b4 - a4*b0
    # m5 = (a0+a1)*b3 = a0*b3 + a1*b3
    # m6 = (a3-a0)*(b0+b1) = a3*b0 + a3*b1 - a0*b0 - a0*b1
    # m7 = (a1-a4)*(b4+b3) = a1*b4 + a1*b3 - a4*b4 - a4*b3
    #
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m7
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # Schoolbook for A1 @ B1[:,2]:
    # C13 = a0*b2 + a1*b5
    # C23 = a3*b2 + a4*b5
    #
    # A2 @ B2:
    # C13 += a2*b6
    # C14... wait, B2 is 1x3, so A2 @ B2 is 2x3:
    # Row 0: [a2*b6, a2*b7, a2*b8]
    # Row 1: [a5*b6, a5*b7, a5*b8]
    #
    # So:
    # C[0,0] = C11 = m1 - m4 + m5 - m7
    # C[0,1] = C12 = m3 + m7
    # C[0,2] = C13 + a2*b6 = a0*b2 + a1*b5 + a2*b6
    # C[1,0] = C21 = m2 + m4
    # C[1,1] = C22 = m1 - m2 + m3 + m6
    # C[1,2] = C23 + a5*b6 = a3*b2 + a4*b5 + a5*b6
    #
    # Wait, I need to be more careful. Let me list all 17 multiplications:
    #
    # m1 = (a0+a4)(b0+b3)
    # m2 = (a3+a4)*b0
    # m3 = a0*(b1-b3)
    # m4 = a4*(b4-b0)
    # m5 = (a0+a1)*b3
    # m6 = (a3-a0)*(b0+b1)
    # m7 = (a1-a4)*(b4+b3)
    # m8 = a0*b2
    # m9 = a1*b5
    # m10 = a3*b2
    # m11 = a4*b5
    # m12 = a2*b6
    # m13 = a2*b7
    # m14 = a2*b8
    # m15 = a5*b6
    # m16 = a5*b7
    # m17 = a5*b8
    #
    # Now express C in terms of these:
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m7
    # C[0,2] = m8 + m9 + m12
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    # C[1,2] = m10 + m11 + m15
    #
    # Let me verify C[0,0]:
    # m1 - m4 + m5 - m7
    # = (a0*b0 + a0*b3 + a4*b0 + a4*b3) - (a4*b4 - a4*b0) + (a0*b3 + a1*b3) - (a1*b4 + a1*b3 - a4*b4 - a4*b3)
    # = a0*b0 + a0*b3 + a4*b0 + a4*b3 - a4*b4 + a4*b0 + a0*b3 + a1*b3 - a1*b4 - a1*b3 + a4*b4 + a4*b3
    # = a0*b0 + 2*a0*b3 + 2*a4*b0 + 2*a4*b3 - a1*b4
    #
    # But C[0,0] should be a0*b0 + a1*b3. This doesn't match!
    #
    # I think I made an error in the Strassen formulas. Let me re-derive.
    #
    # Standard Strassen for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m7
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # Let me verify C11:
    # m1 - m4 + m5 - m7
    # = (a+d)(e+h) - d(g-e) + (a+b)h - (b-d)(g+h)
    # = (ae+ah+de+dh) - (dg-de) + (ah+bh) - (bg+bh-dg-dh)
    # = ae+ah+de+dh-dg+de+ah+bh-bg-bh+dg+dh
    # = ae+2ah+2de+2dh-bg
    #
    # This should equal ae+bg. So: 2ah+2de+2dh-bg = bg, which is not an identity.
    #
    # I'm making errors. Let me look up the correct Strassen formulas.
    #
    # Correct Strassen:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e + f)
    # m3 = (a + b)(h)
    # m4 = (d - b)(e + f)
    # m5 = (a - c)(g + h)
    # m6 = (c + a)(e)
    # m7 = (b - d)(h)
    #
    # C11 = m1 + m4 - m5 - m7
    # C12 = m3 + m5
    # C21 = m2 + m7
    # C22 = m1 - m2 + m3 + m6
    #
    # Let me verify C11:
    # m1 + m4 - m5 - m7
    # = (a+d)(e+h) + (d-b)(e+f) - (a-c)(g+h) - (b-d)h
    # = (ae+ah+de+dh) + (de+df-be-bf) - (ag+ah-cg-ch) - (bh-dh)
    # = ae+ah+de+dh+de+df-be-bf-ag-ah+cg+ch-bh+dh
    # = ae+2de+2dh+df-be-bf-ag+cg+ch-bh
    #
    # This should equal ae+bg. Still not matching.
    #
    # I think I should just use a known correct rank-17 construction. Let me try a different approach.
    #
    # Given the complexity, let me return the schoolbook decomposition to ensure correctness.
    # The rank-17 construction is known to exist but requires careful implementation.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
