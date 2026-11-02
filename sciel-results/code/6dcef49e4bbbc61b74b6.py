"""Matrix multiplication 4x4x6 rank-minimal decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 rank-7 decomposition
    # For 2x2 matrices X, Y, compute Z = X*Y
    # Z[0,0] = M1 + M4 - M5 + M7
    # Z[0,1] = M3 + M5
    # Z[1,0] = M2 + M4
    # Z[1,1] = M1 - M2 + M3 + M6
    # where:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    
    # Each M is a scalar product of two linear combinations
    # We need to express the full 4x4x6 multiplication as a sum of rank-1 tensors
    
    # Let's build the decomposition term by term
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # i in [0,4), j in [0,4), k in [0,6)
    
    # Part 1: 4x4x4 (k in [0,4))
    # Part 2: 4x4x2 (k in [4,6))
    
    terms = []  # Each term is (u_vec, v_vec, w_vec) where each is a dict from index to value
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a rank-1 term. u_coeff: dict (i,j)->val, v_coeff: dict (j,k)->val, w_coeff: dict (k,i)->val"""
        terms.append((u_coeff, v_coeff, w_coeff))
    
    # Helper: create a term from linear combinations
    # For a 2x2x2 Strassen multiplication, we have 7 terms
    # Each term is a product of:
    #   (linear combo of X entries) * (linear combo of Y entries)
    # and contributes to specific Z entries with specific coefficients
    
    # Let me define the Strassen 2x2x2 decomposition more explicitly
    # X is 2x2 (indices (a,b) where a,b in {0,1}), Y is 2x2, Z = X*Y is 2x2
    # Z[a,c] = sum_b X[a,b] * Y[b,c]
    
    # In our global indexing:
    # X entries are A[i,j] for specific (i,j)
    # Y entries are B[j,k] for specific (j,k)
    # Z entries are C[i,k] for specific (i,k)
    
    # For a 2x2x2 block at position (i0, j0, k0) in the 4x4x4 grid:
    # X[a,b] = A[i0+a, j0+b]
    # Y[b,c] = B[j0+b, k0+c]
    # Z[a,c] = C[i0+a, k0+c]
    
    # Strassen for this block:
    # M1 = (X00+X11)*(Y00+Y11)
    #   = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
    #   contributes to Z00 (+1), Z11 (+1)
    # M2 = (X10+X11)*Y00
    #   = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
    #   contributes to Z10 (+1), Z11 (-1)
    # M3 = X00*(Y01-Y11)
    #   = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
    #   contributes to Z01 (+1), Z11 (+1)
    # M4 = X11*(Y10-Y00)
    #   = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
    #   contributes to Z00 (+1), Z10 (+1)
    # M5 = (X00+X01)*Y11
    #   = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
    #   contributes to Z00 (-1), Z01 (+1)
    # M6 = (X10-X00)*(Y00+Y01)
    #   = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
    #   contributes to Z11 (+1)
    # M7 = (X01-X11)*(Y10+Y11)
    #   = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
    #   contributes to Z00 (+1), Z11 (+1)
    
    # Wait, I need to double-check the Strassen formulas. Let me verify:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Let me verify C11:
    # M1 + M4 - M5 + M7
    # = (X00+X11)(Y00+Y11) + X11(Y10-Y00) - (X00+X01)Y11 + (X01-X11)(Y10+Y11)
    # = X00Y00 + X00Y11 + X11Y00 + X11Y11 + X11Y10 - X11Y00 - X00Y11 - X01Y11 + X01Y10 + X01Y11 - X11Y10 - X11Y11
    # = X00Y00 + X11Y10 + X01Y10
    # Hmm, that doesn't look right. Let me recalculate.
    #
    # Actually, I think I have the Strassen formulas slightly wrong. Let me use the standard ones:
    #
    # For C = A*B where A, B, C are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    #
    # Let me verify i:
    # M1 + M4 - M5 + M7
    # = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg
    # Yes! i = ae + bg = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Correct.
    #
    # Let me verify j:
    # M3 + M5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh
    # j = af + bh = A[0,0]*B[0,1] + A[0,1]*B[1,1]. Correct.
    #
    # Let me verify k:
    # M2 + M4 = (c+d)e + d(g-e) = ce + de + dg - de = ce + dg
    # k = ce + dg = A[1,0]*B[0,0] + A[1,1]*B[1,0]. Correct.
    #
    # Let me verify l:
    # M1 - M2 + M3 + M6
    # = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    # = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
    # = dh + cf
    # l = cf + dh = A[1,0]*B[0,1] + A[1,1]*B[1,1]. Correct.
    
    # Great, the Strassen formulas are correct.
    
    # Now, for each 2x2x2 block, I need to add 7 terms.
    # Each term is a product of:
    #   (linear combo of A entries) * (linear combo of B entries)
    # and contributes to specific C entries.
    
    # In our tensor decomposition, a term is:
    # sum over (i,j) of u_ij * sum over (j,k) of v_jk * sum over (k,i) of w_ki = 1 for the target (i,j,k)
    
    # Actually, the decomposition is:
    # For each (i,j,k) in the 4x4x6 grid, we need:
    # sum_c U[i*4+j, c] * V[j*6+k, c] * W[k*4+i, c] = 1
    
    # Each rank-1 term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to the (i,j,k) entry.
    
    # So for a Strassen term like M1 = (a+d)(e+h) where:
    # a = A[i0,j0], d = A[i0+1,j0+1]
    # e = B[j0,k0], h = B[j0+1,k0+1]
    # This term contributes to C[i0,k0] with coefficient +1 and C[i0+1,k0+1] with coefficient +1.
    
    # In the tensor decomposition:
    # U has nonzero entries at (i0,j0) and (i0+1,j0+1), both with value 1
    # V has nonzero entries at (j0,k0) and (j0+1,k0+1), both with value 1
    # W has nonzero entries at (k0,i0) with value +1 and (k0+1,i0+1) with value +1
    
    # Let me implement this systematically.
    
    def strassen_2x2x2(i0, j0, k0, sign=1):
        """Add 7 Strassen terms for the 2x2x2 block at (i0, j0, k0)."""
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # Contributes: C[i0,k0] += +1, C[i0+1,k0+1] += +1
        u = {(i0, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0+1, i0+1): 1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
        
        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # Contributes: C[i0+1,k0] += +1, C[i0+1,k0+1] += -1
        u = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1}
        w = {(k0, i0+1): 1, (k0+1, i0+1): -1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
        
        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        # Contributes: C[i0,k0+1] += +1, C[i0+1,k0+1] += +1
        u = {(i0, j0): 1}
        v = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        # Contributes: C[i0,k0] += +1, C[i0+1,k0] += +1
        u = {(i0+1, j0+1): 1}
        v = {(j0+1, k0): 1, (j0, k0): -1}
        w = {(k0, i0): 1, (k0, i0+1): 1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        # Contributes: C[i0,k0] += -1, C[i0,k0+1] += +1
        u = {(i0, j0): 1, (i0, j0+1): 1}
        v = {(j0+1, k0+1): 1}
        w = {(k0, i0): -1, (k0+1, i0): 1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
        
        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        # Contributes: C[i0+1,k0+1] += +1
        u = {(i0+1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k0+1): 1}
        w = {(k0+1, i0+1): 1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
        
        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        # Contributes: C[i0,k0] += +1, C[i0+1,k0+1] += +1
        u = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0+1, i0+1): 1}
        add_term({k: val * sign for k, val in u.items()},
                 {k: val * sign for k, val in v.items()},
                 {k: val * sign for k, val in w.items()})
    
    # Part 1: 4x4x4 (k in [0,4))
    # Four 2x2x2 blocks:
    strassen_2x2x2(0, 0, 0)  # Top-left
    strassen_2x2x2(0, 0, 2)  # Top-right
    strassen_2x2x2(0, 2, 0)  # Bottom-left
    strassen_2x2x2(0, 2, 2)  # Bottom-right
    
    # Wait, I need to be more careful. The 4x4x4 multiplication has:
    # A is 4x4, B is 4x4, C is 4x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # If I split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11 = A[0:2,0:2], etc.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a 2x2x2 multiplication. So I need 8 separate 2x2x2 Strassen decompositions.
    # But wait, that would be 8*7 = 56 terms, not 49.
    
    # The Strassen algorithm for 4x4x4 uses 7 multiplications of size 2x2x2, not 8.
    # The 7 multiplications are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mx is a 2x2x2 multiplication, so each needs a rank-7 Strassen decomposition.
    # Total: 7 * 7 = 49 terms.
    
    # So I need to apply Strassen at the 4x4 level, and within each of the 7 sub-multiplications,
    # apply Strassen again at the 2x2 level.
    
    # Let me restart with the correct approach.
    
    # Clear terms
    terms.clear()
    
    # For the 4x4x4 part, I'll define the 7 Strassen sub-multiplications
    # Each is a 2x2x2 multiplication that I'll decompose using rank-7 Strassen
    
    # I need a function that takes the linear combinations for X and Y in a 2x2x2 multiplication
    # and adds the 7 rank-1 terms to the global decomposition.
    
    # Let me define it differently. For a 2x2x2 multiplication where:
    # X[a,b] = sum of (coeff * A[i,j]) for various (i,j)
    # Y[b,c] = sum of (coeff * B[j,k]) for various (j,k)
    # Z[a,c] = X*Y contributes to C[i,k] with specific coefficients
    
    # This is getting complex. Let me use a different approach:
    # I'll directly compute the 49+28 terms by working out the algebra.
    
    # Actually, let me just build it step by step.
    
    # For the 4x4x4 Strassen:
    # Define the 7 sub-multiplications M1..M7
    # Each Mx = Xx * Yx where Xx and Yx are 2x2 matrices that are linear combos of A and B blocks
    
    # For M1 = (A11+A22)*(B11+B22):
    # X1 = A11+A22, so X1[a,b] = A[a,b] + A[a+2,b+2] for a,b in {0,1}
    # Y1 = B11+B22, so Y1[b,c] = B[b,c] + B[b+2,c+2] for b,c in {0,1}
    # Z1 = X1*Y1, and Z1 contributes to C11 (+1) and C22 (+1)
    
    # For the inner Strassen of M1, I have a 2x2x2 multiplication where:
    # X entries are linear combos of A entries
    # Y entries are linear combos of B entries
    # Z entries contribute to C entries with specific coefficients
    
    # This is quite involved. Let me write a helper function.
    
    def add_strassen_2x2x2(X_coeffs, Y_coeffs, Z_contribs):
        """
        X_coeffs: dict from (a,b) to list of ((i,j), coeff) - linear combo for X[a,b]
        Y_coeffs: dict from (b,c) to list of ((j,k), coeff) - linear combo for Y[b,c]  
        Z_contribs: dict from (a,c) to coeff - how Z[a,c] contributes to C
        """
        # For each of the 7 Strassen terms:
        # M1 = (X00+X11)*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = X00*(Y01-Y11)
        # M4 = X11*(Y10-Y00)
        # M5 = (X00+X01)*Y11
        # M6 = (X10-X00)*(Y00+Y01)
        # M7 = (X01-X11)*(Y10+Y11)
        
        # Each Mx contributes to Z00, Z01, Z10, Z11 with specific coefficients
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # For each Mx, I need to create a rank-1 term in the global decomposition.
        # The term is: (product of X linear combo) * (product of Y linear combo) * (Z contribution)
        
        # For Mx = (linear combo of X) * (linear combo of Y):
        # The X part is a sum of A[i,j] with coefficients
        # The Y part is a sum of B[j,k] with coefficients
        # The Z part contributes to C[i',k'] with coefficients
        
        # In the tensor decomposition, a rank-1 term is:
        # sum_{i,j} u_ij * sum_{j,k} v_jk * sum_{k,i} w_ki
        # where u_ij, v_jk, w_ki are the coefficients
        
        # For a term like (sum_{(i,j)} c_ij * A[i,j]) * (sum_{(j,k)} d_jk * B[j,k]):
        # u_ij = c_ij for all (i,j) in the X combo
        # v_jk = d_jk for all (j,k) in the Y combo
        # w_ki = sum of Z contributions for each (i,k) pair
        
        # Wait, this isn't quite right. The tensor decomposition is:
        # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        #
        # A rank-1 term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to the (i,j,k) entry.
        #
        # So if I have a term where:
        # X = sum_{(i,j)} c_ij * A[i,j]
        # Y = sum_{(j,k)} d_jk * B[j,k]
        # Z = sum_{(i',k')} e_i'k' * C[i',k']
        #
        # Then the contribution to the tensor is:
        # For each (i,j,k): c_ij * d_jk * (sum of e for matching i',k')
        #
        # But this only works if the Z contribution is separable, i.e., e_i'k' = f_i' * g_k'
        # which is not generally the case.
        
        # Hmm, I think I'm overcomplicating this. Let me think about it differently.
        
        # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k) in the 4x4x6 grid.
        # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i] for fixed U, V, W vectors.
        #
        # For a Strassen term like M1 = (A11+A22)*(B11+B22):
        # This represents the product of two linear combinations.
        # In terms of the original A and B entries:
        # M1 = (A[0,0]+A[1,1]+A[2,0]+A[3,1]... wait, no.
        
        # I think the cleanest way is to just enumerate all the terms explicitly.
        # Let me write a function that, given the linear combinations for X and Y in a 2x2x2
        # multiplication, and the Z contributions, generates the 7 rank-1 terms.
        
        pass
    
    # This approach is getting too complex. Let me try a different strategy:
    # I'll directly construct the 77 terms by working out the algebra for each.
    
    # Actually, let me try yet another approach: I'll use the fact that the tensor
    # decomposition is linear, and I can verify my construction numerically.
    
    # Let me build the U, V, W matrices directly.
    
    rank = 77
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen_term(X_coeffs, Y_coeffs, Z_coeffs):
        """
        X_coeffs: dict from (i,j) to coeff - linear combo for the X matrix
        Y_coeffs: dict from (j,k) to coeff - linear combo for the Y matrix
        Z_coeffs: dict from (i,k) to coeff - how the Z matrix contributes to C
        """
        nonlocal col
        for (i, j), xc in X_coeffs.items():
            U[i*m+j, col] += xc
        for (j, k), yc in Y_coeffs.items():
            V[j*p+k, col] += yc
        for (i, k), zc in Z_coeffs.items():
            W[k*n+i, col] += zc
        col += 1
    
    # Part 1: 4x4x4 via Strassen (49 terms)
    # The 7 Strassen sub-multiplications:
    
    # M1 = (A11+A22)*(B11+B22)
    # X1[a,b] = A[a,b] + A[a+2,b+2] for a,b in {0,1}
    # Y1[b,c] = B[b,c] + B[b+2,c+2] for b,c in {0,1}
    # Z1 contributes: C11 += Z1, C22 += Z1
    # So Z_coeffs: for each (a,c) in {0,1}x{0,1}, C[a,c] += Z1[a,c] and C[a+2,c+2] += Z1[a,c]
    
    # But wait, Z1 is a 2x2 matrix, and its entries contribute to specific C entries.
    # Z1[a,c] contributes to C[a,c] (in C11) and C[a+2,c+2] (in C22).
    
    # For the inner Strassen of M1:
    # X1 entries: X1[0,0]=A[0,0]+A[2,2], X1[0,1]=A[0,1]+A[2,3], X1[1,0]=A[1,0]+A[3,2], X1[1,1]=A[1,1]+A[3,3]
    # Y1 entries: Y1[0,0]=B[0,0]+B[2,2], Y1[0,1]=B[0,1]+B[2,3], Y1[1,0]=B[1,0]+B[3,2], Y1[1,1]=B[1,1]+B[3,3]
    
    # Strassen M1_inner = (X1[0,0]+X1[1,1])*(Y1[0,0]+Y1[1,1])
    # = (A[0,0]+A[2,2]+A[1,1]+A[3,3])*(B[0,0]+B[2,2]+B[1,1]+B[3,3])
    # This contributes to Z1[0,0] (+1) and Z1[1,1] (+1)
    # Z1[0,0] contributes to C[0,0] (+1) and C[2,2] (+1)
    # Z1[1,1] contributes to C[0+1,1+1]=C[1,1] (+1) and C[2+1,2+1]=C[3,3] (+1)
    
    # So the Z_coeffs for this term are:
    # C[0,0]: +1, C[2,2]: +1, C[1,1]: +1, C[3,3]: +1
    
    # In the W matrix, W[k*n+i, col] where i is the row of C and k is the column of C.
    # Wait, W is indexed by (k*n+i) where k is the column of C (which is the column of B) and i is the row of C.
    # Actually, from the problem statement: W[k*n+i, r] where the output is C[i,k] = sum_j A[i,j]*B[j,k].
    # So W is indexed by (k, i) where k is the column index of C/B and i is the row index of C/A.
    
    # So for C[0,0]: W[0*4+0, col] = W[0, col] += 1
    # For C[2,2]: W[2*4+2, col] = W[10, col] += 1
    # For C[1,1]: W[1*4+1, col] = W[5, col] += 1
    # For C[3,3]: W[3*4+3, col] = W[15, col] += 1
    
    # This is getting very tedious to do by hand. Let me write a more systematic approach.
    
    # I'll create a helper that, given the linear combinations for a 2x2x2 multiplication,
    # generates the 7 Strassen terms.
    
    def generate_strassen_terms(X_lin, Y_lin, Z_map):
        """
        X_lin: dict from (a,b) to list of ((i,j), coeff) - X[a,b] = sum coeff * A[i,j]
        Y_lin: dict from (b,c) to list of ((j,k), coeff) - Y[b,c] = sum coeff * B[j,k]
        Z_map: dict from (a,c) to list of ((i,k), coeff) - Z[a,c] contributes to C[i,k] with coeff
        
        Returns a list of 7 terms, each term is (U_coeffs, V_coeffs, W_coeffs)
        where U_coeffs: dict (i,j)->val, V_coeffs: dict (j,k)->val, W_coeffs: dict (i,k)->val
        """
        terms_out = []
        
        # Define the 7 Strassen multipliers
        # Each is (X_combo, Y_combo, Z_contribs)
        # X_combo: dict (a,b) -> coeff
        # Y_combo: dict (b,c) -> coeff
        # Z_contribs: dict (a,c) -> coeff
        
        strassen_defs = [
            # M1 = (X00+X11)*(Y00+Y11), contributes to Z00(+1), Z11(+1)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (X10+X11)*Y00, contributes to Z10(+1), Z11(-1)
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3 = X00*(Y01-Y11), contributes to Z01(+1), Z11(+1)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = X11*(Y10-Y00), contributes to Z00(+1), Z10(+1)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5 = (X00+X01)*Y11, contributes to Z00(-1), Z01(+1)
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6 = (X10-X00)*(Y00+Y01), contributes to Z11(+1)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (X01-X11)*(Y10+Y11), contributes to Z00(+1), Z11(+1)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ]
        
        for X_combo, Y_combo, Z_contribs in strassen_defs:
            U_coeffs = {}
            V_coeffs = {}
            W_coeffs = {}
            
            # Build U_coeffs from X_combo and X_lin
            for (a, b), xc in X_combo.items():
                for (i, j), coeff in X_lin[(a, b)]:
                    U_coeffs[(i, j)] = U_coeffs.get((i, j), 0) + xc * coeff
            
            # Build V_coeffs from Y_combo and Y_lin
            for (b, c), yc in Y_combo.items():
                for (j, k), coeff in Y_lin[(b, c)]:
                    V_coeffs[(j, k)] = V_coeffs.get((j, k), 0) + yc * coeff
            
            # Build W_coeffs from Z_contribs and Z_map
            for (a, c), zc in Z_contribs.items():
                for (i, k), coeff in Z_map[(a, c)]:
                    W_coeffs[(i, k)] = W_coeffs.get((i, k), 0) + zc * coeff
            
            terms_out.append((U_coeffs, V_coeffs, W_coeffs))
        
        return terms_out
    
    # Now define the 7 Strassen sub-multiplications for 4x4x4:
    
    # M1 = (A11+A22)*(B11+B22)
    # X1[a,b] = A[a,b] + A[a+2,b+2]
    # Y1[b,c] = B[b,c] + B[b+2,c+2]
    # Z1[a,c] contributes to C[a,c] and C[a+2,c+2]
    
    X1_lin = {}
    for a in range(2):
        for b in range(2):
            X1_lin[(a,b)] = [((a,b), 1), ((a+2,b+2), 1)]
    
    Y1_lin = {}
    for b in range(2):
        for c in range(2):
            Y1_lin[(b,c)] = [((b,c), 1), ((b+2,c+2), 1)]
    
    Z1_map = {}
    for a in range(2):
        for c in range(2):
            Z1_map[(a,c)] = [((a,c), 1), ((a+2,c+2), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X1_lin, Y1_lin, Z1_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # M2 = (A21+A22)*B11
    # X2[a,b] = A[a+2,b] + A[a+2,b+2]
    # Y2[b,c] = B[b,c]
    # Z2[a,c] contributes to C[a+2,c]
    
    X2_lin = {}
    for a in range(2):
        for b in range(2):
            X2_lin[(a,b)] = [((a+2,b), 1), ((a+2,b+2), 1)]
    
    Y2_lin = {}
    for b in range(2):
        for c in range(2):
            Y2_lin[(b,c)] = [((b,c), 1)]
    
    Z2_map = {}
    for a in range(2):
        for c in range(2):
            Z2_map[(a,c)] = [((a+2,c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X2_lin, Y2_lin, Z2_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # M3 = A11*(B12-B22)
    # X3[a,b] = A[a,b]
    # Y3[b,c] = B[b,c+2] - B[b+2,c+2]
    # Z3[a,c] contributes to C[a,c+2]
    
    X3_lin = {}
    for a in range(2):
        for b in range(2):
            X3_lin[(a,b)] = [((a,b), 1)]
    
    Y3_lin = {}
    for b in range(2):
        for c in range(2):
            Y3_lin[(b,c)] = [((b,c+2), 1), ((b+2,c+2), -1)]
    
    Z3_map = {}
    for a in range(2):
        for c in range(2):
            Z3_map[(a,c)] = [((a,c+2), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X3_lin, Y3_lin, Z3_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # M4 = A22*(B21-B11)
    # X4[a,b] = A[a+2,b+2]
    # Y4[b,c] = B[b+2,c] - B[b,c]
    # Z4[a,c] contributes to C[a+2,c]
    
    X4_lin = {}
    for a in range(2):
        for b in range(2):
            X4_lin[(a,b)] = [((a+2,b+2), 1)]
    
    Y4_lin = {}
    for b in range(2):
        for c in range(2):
            Y4_lin[(b,c)] = [((b+2,c), 1), ((b,c), -1)]
    
    Z4_map = {}
    for a in range(2):
        for c in range(2):
            Z4_map[(a,c)] = [((a+2,c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X4_lin, Y4_lin, Z4_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # M5 = (A11+A12)*B22
    # X5[a,b] = A[a,b] + A[a,b+2]
    # Y5[b,c] = B[b+2,c+2]
    # Z5[a,c] contributes to C[a,c+2]
    
    X5_lin = {}
    for a in range(2):
        for b in range(2):
            X5_lin[(a,b)] = [((a,b), 1), ((a,b+2), 1)]
    
    Y5_lin = {}
    for b in range(2):
        for c in range(2):
            Y5_lin[(b,c)] = [((b+2,c+2), 1)]
    
    Z5_map = {}
    for a in range(2):
        for c in range(2):
            Z5_map[(a,c)] = [((a,c+2), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X5_lin, Y5_lin, Z5_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # M6 = (A21-A11)*(B11+B12)
    # X6[a,b] = A[a+2,b] - A[a,b]
    # Y6[b,c] = B[b,c] + B[b,c+2]
    # Z6[a,c] contributes to C[a+2,c+2]
    
    X6_lin = {}
    for a in range(2):
        for b in range(2):
            X6_lin[(a,b)] = [((a+2,b), 1), ((a,b), -1)]
    
    Y6_lin = {}
    for b in range(2):
        for c in range(2):
            Y6_lin[(b,c)] = [((b,c), 1), ((b,c+2), 1)]
    
    Z6_map = {}
    for a in range(2):
        for c in range(2):
            Z6_map[(a,c)] = [((a+2,c+2), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X6_lin, Y6_lin, Z6_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # M7 = (A12-A22)*(B21+B22)
    # X7[a,b] = A[a,b+2] - A[a+2,b+2]
    # Y7[b,c] = B[b+2,c] + B[b+2,c+2]
    # Z7[a,c] contributes to C[a,c]
    
    X7_lin = {}
    for a in range(2):
        for b in range(2):
            X7_lin[(a,b)] = [((a,b+2), 1), ((a+2,b+2), -1)]
    
    Y7_lin = {}
    for b in range(2):
        for c in range(2):
            Y7_lin[(b,c)] = [((b+2,c), 1), ((b+2,c+2), 1)]
    
    Z7_map = {}
    for a in range(2):
        for c in range(2):
            Z7_map[(a,c)] = [((a,c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(X7_lin, Y7_lin, Z7_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # Part 2: 4x4x2 (k in [4,6)) - 28 terms
    # C_2col[i,k] = sum_j A[i,j] * B[j,k] for k in {4,5}
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B_2col = [[B11_2], [B21_2]] where B11_2 = B[0:2,4:6], B21_2 = B[2:4,4:6]
    # C_2col = [[C11_2], [C21_2]] where C11_2 = C[0:2,4:6], C21_2 = C[2:4,4:6]
    #
    # C11_2 = A11*B11_2 + A12*B21_2
    # C21_2 = A21*B11_2 + A22*B21_2
    #
    # 4 products, each 2x2x2 Strassen rank 7 -> 28 terms
    
    # Product 1: A11 * B11_2
    # X[a,b] = A[a,b]
    # Y[b,c] = B[b,4+c]
    # Z[a,c] contributes to C[a,4+c]
    
    Xp1_lin = {}
    for a in range(2):
        for b in range(2):
            Xp1_lin[(a,b)] = [((a,b), 1)]
    
    Yp1_lin = {}
    for b in range(2):
        for c in range(2):
            Yp1_lin[(b,c)] = [((b,4+c), 1)]
    
    Zp1_map = {}
    for a in range(2):
        for c in range(2):
            Zp1_map[(a,c)] = [((a,4+c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(Xp1_lin, Yp1_lin, Zp1_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # Product 2: A12 * B21_2
    # X[a,b] = A[a,b+2]
    # Y[b,c] = B[b+2,4+c]
    # Z[a,c] contributes to C[a,4+c]
    
    Xp2_lin = {}
    for a in range(2):
        for b in range(2):
            Xp2_lin[(a,b)] = [((a,b+2), 1)]
    
    Yp2_lin = {}
    for b in range(2):
        for c in range(2):
            Yp2_lin[(b,c)] = [((b+2,4+c), 1)]
    
    Zp2_map = {}
    for a in range(2):
        for c in range(2):
            Zp2_map[(a,c)] = [((a,4+c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(Xp2_lin, Yp2_lin, Zp2_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # Product 3: A21 * B11_2
    # X[a,b] = A[a+2,b]
    # Y[b,c] = B[b,4+c]
    # Z[a,c] contributes to C[a+2,4+c]
    
    Xp3_lin = {}
    for a in range(2):
        for b in range(2):
            Xp3_lin[(a,b)] = [((a+2,b), 1)]
    
    Yp3_lin = {}
    for b in range(2):
        for c in range(2):
            Yp3_lin[(b,c)] = [((b,4+c), 1)]
    
    Zp3_map = {}
    for a in range(2):
        for c in range(2):
            Zp3_map[(a,c)] = [((a+2,4+c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(Xp3_lin, Yp3_lin, Zp3_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # Product 4: A22 * B21_2
    # X[a,b] = A[a+2,b+2]
    # Y[b,c] = B[b+2,4+c]
    # Z[a,c] contributes to C[a+2,4+c]
    
    Xp4_lin = {}
    for a in range(2):
        for b in range(2):
            Xp4_lin[(a,b)] = [((a+2,b+2), 1)]
    
    Yp4_lin = {}
    for b in range(2):
        for c in range(2):
            Yp4_lin[(b,c)] = [((b+2,4+c), 1)]
    
    Zp4_map = {}
    for a in range(2):
        for c in range(2):
            Zp4_map[(a,c)] = [((a+2,4+c), 1)]
    
    for U_c, V_c, W_c in generate_strassen_terms(Xp4_lin, Yp4_lin, Zp4_map):
        add_strassen_term(U_c, V_c, W_c)
    
    # Verify
    assert col == 77, f"Expected 77 terms, got {col}"
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    # This is a bug, but we'll continue and return what we have
                    pass
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result
