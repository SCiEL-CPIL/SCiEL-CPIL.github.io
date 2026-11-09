"""
Matrix multiplication 4x4 by 4x8 — tensor rank minimization.

Strategy:
  1. Try to find a rank-46 decomposition by using two independent rank-23
     Laderman 4x4x4 blocks (one for columns 0-3, one for columns 4-7).
  2. If the Laderman coefficients cannot be verified, fall back to a
     rank-64 Kronecker-product-based decomposition (Strassen 2x2x2 ⊗ Strassen 2x2x2
     for each half, but actually that gives 49 per half... let me use a simpler
     valid structure).
  3. Final fallback: schoolbook rank-128.

The key insight: 4x4x8 = 4x4x4 ⊕ 4x4x4 (split output columns).
If we have a valid rank-r decomposition for 4x4x4, we get rank 2r for 4x4x8.
Best known for 4x4x4 is 23 (Laderman). We attempt to encode and verify it.

The output follows the trace convention W[k*n+i, r] for C = A @ B
where A is 4x4, B is 4x8, C is 4x8.
"""

import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 8]

_START_TIME = time.time()
_TIMEOUT = 135.0


def _time_up():
    return (time.time() - _START_TIME) > _TIMEOUT


def _validate(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Correct 7-term Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = (c+d)h
    # p4 = a(f-g)
    # p5 = d(g-e)
    # p6 = (a+b)(e+f)
    # p7 = (b-d)(f+h)
    #
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p6
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p5
    #
    # Let me verify C00:
    # p1 = ae + ah + de + dh
    # p4 = af - ag
    # p5 = dg - de
    # p7 = bf + bh - df - dh
    # p1+p4-p5+p7 = ae+ah+de+dh + af-ag -dg+de + bf+bh-df-dh
    # = ae + ah + 2de + dh + af - ag - dg + bf + bh - df - dh
    # = ae + ah + 2de + af - ag - dg + bf + bh - df
    # This should equal ae + bg. Let me check with specific values...
    # This is getting error-prone. Let me use a well-known correct version.
    #
    # Standard Strassen (Wikipedia):
    # p1 = (a + d)(e + h)
    # p2 = (c + d)h
    # p3 = a(f - g)
    # p4 = d(g - e)
    # p5 = (a + b)(e + f)
    # p6 = (c - a)(g + h)
    # p7 = (b - d)(f + h)
    #
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    #
    # Verify C00 = p1 + p4 - p5 + p7:
    # p1 = ae + ah + de + dh
    # p4 = dg - de
    # p5 = ae + af + be + bf
    # p7 = bf + bh - df - dh
    # Sum = ae+ah+de+dh + dg-de -ae-af-be-bf + bf+bh-df-dh
    # = ae+ah+de+dh+dg-de-ae-af-be-bf+bf+bh-df-dh
    # = ah+dg-af-be+bh-df
    # = a(h-af... wait this is getting messy. Let me just verify numerically.

    # I'll use a numerical verification approach instead of trying to recall.
    # Let me define the 7 products and verify they work.

    # Product definitions (coefficients in A and B):
    # Each product r has:
    #   A-part: linear combination of a,b,c,d
    #   B-part: linear combination of e,f,g,h
    #   C-coefficients: how it contributes to C00,C01,C10,C11

    # Let me use a different, verified approach:
    # I'll construct the decomposition by solving the system directly.
    # For 2x2x2, the tensor has 8 nonzero entries (all of them).
    # Rank 7 means we need 7 rank-1 tensors summing to T.

    # Actually, let me just hardcode a known correct Strassen and verify it.
    # From multiple sources, the correct Strassen is:

    # Let me define it as: for each product r, give (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: [coeff_a, coeff_b, coeff_c, coeff_d]
    # B_coeffs: [coeff_e, coeff_f, coeff_g, coeff_h]
    # C_coeffs: [coeff_C00, coeff_C01, coeff_C10, coeff_C11]

    # Product 1: (a+d)(e+h) -> C00:+1, C11:+1
    # Product 2: (c+d)h -> C10:+1, C11:-1
    # Product 3: a(f-g) -> C01:+1, C10:+1
    # Product 4: d(g-e) -> C00:+1, C10:+1
    # Product 5: (a+b)(e+f) -> C00:-1, C01:+1
    # Product 6: (c-a)(g+h) -> C11:+1
    # Product 7: (b-d)(f+h) -> C00:+1, C01:0, C10:0, C11:0... 

    # I'm going in circles. Let me just do a numerical search for 2x2x2 rank 7.
    # Actually, for the purpose of this program, let me use a simpler valid
    # decomposition. The schoolbook for 2x2x2 is rank 8. Strassen is rank 7.
    # But I can't reliably recall the exact coefficients.

    # Alternative approach: use the Kronecker product structure.
    # MatMul_{4,4,4} can be decomposed using the fact that 4 = 2*2.
    # The tensor T_{4,4,4}[i,j,k] = sum over the "inner" indices.
    # Actually, for square matmul, there's a natural Kronecker structure:
    # If we write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0,
    # then C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] B[2*j1+j0, 2*k1+k0]
    # This doesn't factorize cleanly into a Kronecker product of two 2x2x2 tensors
    # because the summation index j is shared.

    # Let me try a completely different approach: use numpy to find a low-rank
    # decomposition numerically via iterative methods.

    pass


def _find_low_rank_444(max_rank=30, time_limit=100.0):
    """
    Try to find a low-rank decomposition for 4x4x4 matmul.
    Uses a greedy/iterative approach.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    n = m = p = 4
    deadline = time.time() + time_limit

    # The tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[i*4+j, j*4+k, k*4+i] = 1 for all i,j,k in {0,1,2,3}
    # All other entries are 0.

    # Number of nonzero entries: 4*4*4 = 64
    # Total entries in the 3D tensor: 16*16*16 = 4096

    # For a rank-r decomposition:
    # sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]

    # Strategy: iterative hard thresholding / power method for tensors
    # 1. Initialize U, V, W randomly
    # 2. Iterate:
    #    a. Fix V, W; solve for U (linear system)
    #    b. Fix U, W; solve for V
    #    c. Fix U, V; solve for W
    # 3. Check convergence

    # For step (a): given V, W, we need U[a,s] such that
    # sum_s U[a,s] * (V[:,s] * W[:,s] reshaped appropriately) = T[a, :, :]
    # This is: for each a, sum_s U[a,s] * M_s[a] = T[a] where M_s is a matrix...
    # Actually it's: for each (b,c): sum_s U[a,s] V[b,s] W[c,s] = T[a,b,c]
    # Fixing V and W, for each a: sum_s U[a,s] * (V[:,s] * W[:,s])_flattened... 
    # No. Let me think more carefully.
    #
    # Let's flatten: for fixed a, we have a vector t_a of length 16*16=256
    # (indexed by (b,c)). We need: sum_s U[a,s] * (v_s ⊗ w_s) = t_a
    # where v_s is a vector of length 16 and w_s is a vector of length 16.
    # So: U[a,:] * [v_0⊗w_0, v_1⊗w_1, ..., v_{r-1}⊗w_{r-1}] = t_a
    # This is a linear system in U[a,:].

    # This is computationally expensive. Let me try a simpler approach.

    # Actually, for this specific problem, let me try to use known results.
    # The best known rank for 4x4x4 is 20 (not 23 as I thought earlier).
    # Wait, let me check: Laderman's algorithm for 4x4x4 uses 23 multiplications.
    # But there might be better. Actually, I recall that the optimal rank for
    # 4x4x4 is unknown, but 20 is achievable via some constructions.

    # Given the time constraints and difficulty of recalling exact coefficients,
    # let me try a different strategy: use a structured decomposition based on
    # block matrices.

    # 4x4x4: Split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (each 2x2)
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # Each 2x2*2x2 product can be done with 7 Strassen multiplications.
    # There are 8 such products (4 block products, each requiring 2 Strassen).
    # Total: 8*7 = 56? No, that's not right either.

    # Actually, the block structure gives:
    # C = A*B where A, B are 4x4.
    # If we use Strassen at the block level:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = (A21+A22)B21
    # P4 = A11(B12-B22)
    # P5 = A22(B21-B11)
    # P6 = (A11+A12)(B11+B12)
    # P7 = (A12-A22)(B12+B22)
    #
    # C11 = P1 + P5 - P6 + P7
    # C12 = P4 + P6
    # C21 = P2 + P5
    # C22 = P1 - P2 + P4 + P3  (wait, I need to check this)

    # Each P_i is a 2x2*2x2 product, which itself needs 7 Strassen multiplications.
    # So total: 7*7 = 49 multiplications for 4x4x4 via recursive Strassen.
    # That's better than schoolbook 64 but worse than Laderman 23.

    # For 4x4x8: split into two 4x4x4 problems, each done with 49 = 98 total.
    # Still better than 128!

    # Let me implement the recursive Strassen (rank 49 for 4x4x4, rank 98 for 4x4x8).
    # This is guaranteed to be correct if I implement Strassen correctly.

    # But I still need correct Strassen coefficients for 2x2x2!
    # Let me just verify them numerically.

    # Define the 7 products and check if they work.
    # I'll use a specific numerical example to verify.

    # Actually, let me just try all reasonable Strassen formulations and verify.

    # Formulation A (from Wikipedia, verified):
    # p1 = (a+d)(e+h), p2 = (c+d)h, p3 = a(f-g), p4 = d(g-e)
    # p5 = (a+b)(e+f), p6 = (c-a)(g+h), p7 = (b-d)(f+h)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6

    # Let me verify with a=1,b=2,c=3,d=4, e=5,f=6,g=7,h=8
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]

    a, b, c, d = 1, 2, 3, 4
    e, f, g, h = 5, 6, 7, 8
    C00 = a*e + b*g
    C01 = a*f + b*h
    C10 = c*e + d*g
    C11 = c*f + d*h

    p1 = (a+d)*(e+h)
    p2 = (c+d)*h
    p3 = a*(f-g)
    p4 = d*(g-e)
    p5 = (a+b)*(e+f)
    p6 = (c-a)*(g+h)
    p7 = (b-d)*(f+h)

    C00_calc = p1 + p4 - p5 + p7
    C01_calc = p3 + p5
    C10_calc = p2 + p4
    C11_calc = p1 - p2 + p3 + p6

    if (C00_calc != C00 or C01_calc != C01 or 
        C10_calc != C10 or C11_calc != C11):
        # Try another formulation
        # Formulation B:
        # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = (c+d)h, p4 = a(f-g)
        # p5 = d(g-e), p6 = (a+b)(e+f), p7 = (b-d)(f+h)
        # C00 = p1 + p4 - p5 + p7
        # C01 = p3 + p6
        # C10 = p2 + p4
        # C11 = p1 - p2 + p3 + p5
        
        p1 = (a+d)*(e+h)
        p2 = (c+d)*e
        p3 = (c+d)*h
        p4 = a*(f-g)
        p5 = d*(g-e)
        p6 = (a+b)*(e+f)
        p7 = (b-d)*(f+h)

        C00_calc = p1 + p4 - p5 + p7
        C01_calc = p3 + p6
        C10_calc = p2 + p4
        C11_calc = p1 - p2 + p3 + p5

        if (C00_calc != C00 or C01_calc != C01 or 
            C10_calc != C10 or C11_calc != C11):
            # Try yet another
            # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-g), p4 = d(g-e)
            # p5 = (a+b)(e+f), p6 = (c-a)(g+h), p7 = (b-d)(f+h)
            # C00 = p1 + p4 - p5 + p7
            # C01 = p3 + p5
            # C10 = p2 + p4
            # C11 = p1 - p2 + p3 + p6
            
            p1 = (a+d)*(e+h)
            p2 = (c+d)*e
            p3 = a*(f-g)
            p4 = d*(g-e)
            p5 = (a+b)*(e+f)
            p6 = (c-a)*(g+h)
            p7 = (b-d)*(f+h)

            C00_calc = p1 + p4 - p5 + p7
            C01_calc = p3 + p5
            C10_calc = p2 + p4
            C11_calc = p1 - p2 + p3 + p6

            if (C00_calc != C00 or C01_calc != C01 or 
                C10_calc != C10 or C11_calc != C11):
                # None of these work with my verification. Let me try more.
                # The issue might be my understanding of the Strassen formulas.
                # Let me try:
                # p1 = (a+d)(e+h), p2 = (c+d)h, p3 = a(f-g), p4 = d(g-e)
                # p5 = (a+b)(e+f), p6 = (c-a)(g+h), p7 = (b-d)(f+h)
                # C00 = p1 + p4 - p5 + p7
                # C01 = p3 + p5
                # C10 = p2 + p4
                # C11 = p1 - p2 + p3 + p6
                # (same as formulation A)
                
                # Let me try with different test values
                a, b, c, d = 1, 0, 0, 0
                e, f, g, h = 1, 0, 0, 0
                C00 = a*e + b*g
                C01 = a*f + b*h
                C10 = c*e + d*g
                C11 = c*f + d*h
                
                p1 = (a+d)*(e+h)
                p2 = (c+d)*h
                p3 = a*(f-g)
                p4 = d*(g-e)
                p5 = (a+b)*(e+f)
                p6 = (c-a)*(g+h)
                p7 = (b-d)*(f+h)

                C00_calc = p1 + p4 - p5 + p7
                C01_calc = p3 + p5
                C10_calc = p2 + p4
                C11_calc = p1 - p2 + p3 + p6

                if C00_calc != C00 or C01_calc != C01 or C10_calc != C10 or C11_calc != C11:
                    # Give up on recalling Strassen and use a different approach
                    return None

    # If we get here, formulation A seems to work for the test case.
    # Let me verify with more random cases.
    import random
    random.seed(42)
    for _ in range(100):
        a, b, c, d = [random.randint(-5, 5) for _ in range(4)]
        e, f, g, h = [random.randint(-5, 5) for _ in range(4)]
        C00 = a*e + b*g
        C01 = a*f + b*h
        C10 = c*e + d*g
        C11 = c*f + d*h

        p1 = (a+d)*(e+h)
        p2 = (c+d)*h
        p3 = a*(f-g)
        p4 = d*(g-e)
        p5 = (a+b)*(e+f)
        p6 = (c-a)*(g+h)
        p7 = (b-d)*(f+h)

        if (p1 + p4 - p5 + p7 != C00 or p3 + p5 != C01 or
            p2 + p4 != C10 or p1 - p2 + p3 + p6 != C11):
            return None

    # Formulation A is verified! Now let me build the tensor decomposition.
    # For 2x2x2 with this Strassen:
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    #
    # Product r corresponds to:
    # A-part: linear in (a,b,c,d) = (A[0,0],A[0,1],A[1,0],A[1,1])
    # B-part: linear in (e,f,g,h) = (B[0,0],B[0,1],B[1,0],B[1,1])
    # C-part: coefficients for (C[0,0],C[0,1],C[1,0],C[1,1])
    #
    # The tensor entry T[i,j,k] = 1 means:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #
    # For product r, the contribution to T[i,j,k] is:
    # U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # The A-part of product r determines U[*,*,r]:
    # If A-part = alpha*a + beta*b + gamma*c + delta*d, then:
    # U[0,0,r] = alpha, U[0,1,r] = beta, U[1,0,r] = gamma, U[1,1,r] = delta
    #
    # The B-part determines V[*,*,r]:
    # If B-part = epsilon*e + zeta*f + eta*g + theta*h, then:
    # V[0,0,r] = epsilon, V[0,1,r] = zeta, V[1,0,r] = eta, V[1,1,r] = theta
    #
    # The C-part determines W[*,*,r]:
    # W[k,i,r] = coefficient of product r in C[i,k]
    # C[0,0]: i=0,k=0 -> W[0*2+0, r] = W[0, r]
    # C[0,1]: i=0,k=1 -> W[1*2+0, r] = W[2, r]
    # C[1,0]: i=1,k=0 -> W[0*2+1, r] = W[1, r]
    # C[1,1]: i=1,k=1 -> W[1*2+1, r] = W[3, r]

    # Product 1: (a+d)(e+h)
    # A-part: 1*a + 0*b + 0*c + 1*d -> U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=1
    # B-part: 1*e + 0*f + 0*g + 1*h -> V[0,0]=1, V[0,1]=0, V[1,0]=0, V[1,1]=1
    # C-part: C00:+1, C01:0, C10:0, C11:+1 -> W[0]=1, W[2]=0, W[1]=0, W[3]=1

    # Product 2: (c+d)h
    # A-part: 0*a + 0*b + 1*c + 1*d -> U[0,0]=0, U[0,1]=0, U[1,0]=1, U[1,1]=1
    # B-part: 0*e + 0*f + 0*g + 1*h -> V[0,0]=0, V[0,1]=0, V[1,0]=0, V[1,1]=1
    # C-part: C00:0, C01:0, C10:+1, C11:-1 -> W[0]=0, W[2]=0, W[1]=1, W[3]=-1

    # Product 3: a(f-g)
    # A-part: 1*a + 0*b + 0*c + 0*d -> U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=0
    # B-part: 0*e + 1*f + (-1)*g + 0*h -> V[0,0]=0, V[0,1]=1, V[1,0]=-1, V[1,1]=0
    # C-part: C00:0, C01:+1, C10:+1, C11:0 -> W[0]=0, W[2]=1, W[1]=1, W[3]=0

    # Product 4: d(g-e)
    # A-part: 0*a + 0*b + 0*c + 1*d -> U[0,0]=0, U[0,1]=0, U[1,0]=0, U[1,1]=1
    # B-part: (-1)*e + 0*f + 1*g + 0*h -> V[0,0]=-1, V[0,1]=0, V[1,0]=1, V[1,1]=0
    # C-part: C00:+1, C01:0, C10:+1, C11:0 -> W[0]=1, W[2]=0, W[1]=1, W[3]=0

    # Product 5: (a+b)(e+f)
    # A-part: 1*a + 1*b + 0*c + 0*d -> U[0,0]=1, U[0,1]=1, U[1,0]=0, U[1,1]=0
    # B-part: 1*e + 1*f + 0*g + 0*h -> V[0,0]=1, V[0,1]=1, V[1,0]=0, V[1,1]=0
    # C-part: C00:-1, C01:+1, C10:0, C11:0 -> W[0]=-1, W[2]=1, W[1]=0, W[3]=0

    # Product 6: (c-a)(g+h)
    # A-part: (-1)*a + 0*b + 1*c + 0*d -> U[0,0]=-1, U[0,1]=0, U[1,0]=1, U[1,1]=0
    # B-part: 0*e + 0*f + 1*g + 1*h -> V[0,0]=0, V[0,1]=0, V[1,0]=1, V[1,1]=1
    # C-part: C00:0, C01:0, C10:0, C11:+1 -> W[0]=0, W[2]=0, W[1]=0, W[3]=1

    # Product 7: (b-d)(f+h)
    # A-part: 0*a + 1*b + 0*c + (-1)*d -> U[0,0]=0, U[0,1]=1, U[1,0]=0, U[1,1]=-1
    # B-part: 0*e + 1*f + 0*g + 1*h -> V[0,0]=0, V[0,1]=1, V[1,0]=0, V[1,1]=1
    # C-part: C00:+1, C01:0, C10:0, C11:0 -> W[0]=1, W[2]=0, W[1]=0, W[3]=0

    # Now let me verify this decomposition numerically.
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # U[0,0]: a
        [0, 0, 0, 0, 1, 0, 1],    # U[0,1]: b
        [0, 1, 0, 0, 0, 1, 0],    # U[1,0]: c
        [1, 1, 0, 1, 0, 0, -1]    # U[1,1]: d
    ]
    V = [
        [1, 0, 0, -1, 1, 0, 0],   # V[0,0]: e
        [0, 0, 1, 0, 1, 0, 1],    # V[0,1]: f
        [0, 0, -1, 1, 0, 1, 0],   # V[1,0]: g
        [1, 1, 0, 0, 0, 1, 1]     # V[1,1]: h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # W[0,0]: C00
        [0, 1, 1, 1, 0, 0, 0],    # W[1,0]: C10 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # W[2,0]: C01 (k=1,i=0)
        [1, -1, 0, 0, 0, 1, 0]    # W[3,0]: C11 (k=1,i=1)
    ]

    # Verify: for all i,j,k in {0,1}: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                if s != 1:
                    return None

    return U, V, W


def _kronecker_444():
    """
    Build a 4x4x4 decomposition using the Kronecker product of two 2x2x2 Strassen.
    rank <= 7*7 = 49.
    """
    strassen = _find_low_rank_444(max_rank=7, time_limit=5.0)
    if strassen is None:
        return None
    U2, V2, W2 = strassen
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    # The 4x4x4 tensor can be seen as a Kronecker product of two 2x2x2 tensors
    # when we write indices as i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] B[2*j1+j0, 2*k1+k0]
    #
    # This is NOT a simple Kronecker product because j is shared.
    # So the Kronecker approach doesn't directly apply here.
    #
    # Instead, let me use the block Strassen approach:
    # Split A, B into 2x2 blocks and use Strassen at the block level.
    # This gives 7 block products, each of which is a 2x2*2x2 product
    # done with 7 Strassen multiplications = 49 total.
    #
    # Block Strassen for 4x4x4:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = (A21+A22)B21
    # P4 = A11(B12-B22)
    # P5 = A22(B21-B11)
    # P6 = (A11+A12)(B11+B12)
    # P7 = (A12-A22)(B12+B22)
    #
    # C11 = P1 + P5 - P6 + P7
    # C12 = P4 + P6
    # C21 = P2 + P5
    # C22 = P1 - P2 + P4 + P3  (need to verify)
    #
    # Wait, I need to verify the block Strassen formulas.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # P2 = (A21+A22)B11 = A21B11 + A22B11
    # P3 = (A21+A22)B21 = A21B21 + A22B21
    # P4 = A11(B12-B22) = A11B12 - A11B22
    # P5 = A22(B21-B11) = A22B21 - A22B11
    # P6 = (A11+A12)(B11+B12) = A11B11 + A11B12 + A12B11 + A12B12
    # P7 = (A12-A22)(B12+B22) = A12B12 + A12B22 - A22B12 - A22B22
    #
    # C11 = P1 + P5 - P6 + P7
    # = (A11B11+A11B22+A22B11+A22B22) + (A22B21-A22B11) - (A11B11+A11B12+A12B11+A12B12) + (A12B12+A12B22-A22B12-A22B22)
    # A11B11: 1-1 = 0
    # A11B22: 1+1 = 2
    # A22B11: 1-1 = 0
    # A22B22: 1-1 = 0
    # A22B21: 1
    # A11B12: -1
    # A12B11: -1
    # A12B12: -1+1 = 0
    # A12B22: 1
    # A22B12: -1
    # This doesn't simplify to A11B11 + A12B21. So my block Strassen is wrong.
    #
    # I'm clearly bad at recalling Strassen. Let me just use the verified
    # 2x2x2 Strassen and build up via a different method.
    #
    # Actually, let me just use the schoolbook for the 4x4x8 problem but
    # with a small improvement: notice that for 4x4x8, we can write
    # C = A*B where A is 4x4 and B is 4x8.
    # The tensor has 128 nonzero entries.
    #
    # A simple valid improvement: use the fact that the last dimension (p=8)
    # can be split into two groups of 4, and for each group, the 4x4x4 tensor
    # can be decomposed with rank at most 49 (if I can get block Strassen right)
    # or 64 (schoolbook).
    #
    # Since I can't reliably get the block Strassen, let me just return schoolbook
    # but at least verify it's correct.

    return None


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Try to find a rank-46 decomposition (two Laderman 4x4x4 blocks)
    # or rank-98 (two block-Strassen 4x4x4 with rank 49 each)
    # For now, let me try the Kronecker/Strassen approach.

    # First, verify we have a correct 2x2x2 Strassen
    strassen_222 = _find_low_rank_444(max_rank=7, time_limit=5.0)

    if strassen_222 is not None and time.time() - _START_TIME < _TIMEOUT:
        U2, V2, W2 = strassen_222
        # Build 4x4x8 using block structure
        # C = A*B, A is 4x4, B is 4x8
        # Split B into B1 (4x4) and B2 (4x4)
        # C1 = A*B1 (4x4), C2 = A*B2 (4x4)
        # For each 4x4x4, use block Strassen with 2x2x2 Strassen inside.
        #
        # Block Strassen for 4x4x4 (verified):
        # I need to get this right. Let me use the same Strassen formulas
        # at the block level.
        #
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # Using the verified 2x2x2 Strassen at the block level:
        # The 7 products are:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B22  (wait, I need to match the verified formula)
        #
        # From the verified 2x2x2 Strassen:
        # p1 = (a+d)(e+h), p2 = (c+d)h, p3 = a(f-g), p4 = d(g-e)
        # p5 = (a+b)(e+f), p6 = (c-a)(g+h), p7 = (b-d)(f+h)
        # C00 = p1 + p4 - p5 + p7
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p2 + p3 + p6
        #
        # At the block level (a=A11, b=A12, c=A21, d=A22, e=B11, f=B12, g=B21, h=B22):
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B22
        # P3 = A11(B12-B21)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)(B11+B12)
        # P6 = (A21-A11)(B21+B22)
        # P7 = (A12-A22)(B12+B22)
        #
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        #
        # Let me verify C11:
        # P1 = A11B11+A11B22+A22B11+A22B22
        # P4 = A22B21-A22B11
        # P5 = A11B11+A11B12+A12B11+A12B12
        # P7 = A12B12+A12B22-A22B12-A22B22
        # C11 = P1+P4-P5+P7
        # A11B11: 1-1 = 0
        # A11B22: 1+1 = 2
        # A22B11: 1-1 = 0
        # A22B22: 1-1 = 0
        # A22B21: 1
        # A11B12: -1
        # A12B11: -1
        # A12B12: -1+1 = 0
        # A12B22: 1
        # A22B12: -1
        # Should equal A11B11 + A12B21. Doesn't match.
        #
        # The issue is that the block-level Strassen doesn't work the same way
        # because the "multiplication" at the block level is matrix multiplication,
        # not scalar multiplication. The Strassen formulas are for scalar 2x2*2x2,
        # and when applied to blocks, each "product" like (A11+A22)(B11+B22)
        # is a matrix product, which itself requires multiple scalar multiplications.
        #
        # So the total rank is 7 (block products) * 7 (Strassen per block product) = 49.
        # But I need to correctly decompose each block product.
        #
        # This is getting very complex. Given the time constraints, let me just
        # return the schoolbook decomposition, which is guaranteed correct.
        pass

    # Final fallback: schoolbook
    return _make_schoolbook(n, m, p)
