"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, coeff=1.0):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j, col] += coeff
        V[j * p + k, col] += coeff
        W[k * n + i, col] += coeff
        col += 1

    # Helper to add Strassen 2x2x2 block
    def add_strassen(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices for the 2x2x2 block
        # Local indices 0,1 map to global i_base+0, i_base+1 etc.
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00 -> U[i_base, j_base], A11 -> U[i_base+1, j_base+1]
        # B00 -> V[j_base, k_base], B11 -> V[j_base+1, k_base+1]
        # C00 -> W[k_base, i_base], C11 -> W[k_base+1, i_base+1]
        
        # Term 1: A00 B00 -> C00
        add_term(i_base, j_base, k_base, 1)
        # Term 2: A00 B11 -> C00
        add_term(i_base, j_base, k_base + 1, 1)
        # Term 3: A11 B00 -> C11
        add_term(i_base + 1, j_base + 1, k_base, 1)
        # Term 4: A11 B11 -> C11
        add_term(i_base + 1, j_base + 1, k_base + 1, 1)
        
        # Wait, Strassen combines these. We need to represent the linear combinations in U and V.
        # The standard way is to have separate columns for the products.
        # M1 requires 1 column? No, M1 is a product of sums.
        # (A00+A11) is a linear combo in U. (B00+B11) is a linear combo in V.
        # The product contributes to C00 and C11.
        # C00 gets +M1. C11 gets +M1.
        
        # Let's implement Strassen by adding 7 specific rank-1 terms that form the decomposition.
        # This requires careful coefficient management.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: 1 at (i0, j0), 1 at (i1, j1)
        # V: 1 at (j0, k0), 1 at (j1, k1)
        # W: +1 at (k0, i0), +1 at (k1, i1)
        
        nonlocal col
        if col + 7 > rank: return
        
        # M1
        U[i_base * m + j_base, col] = 1
        U[(i_base + 1) * m + (j_base + 1), col] = 1
        V[j_base * p + k_base, col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = 1
        W[k_base * n + i_base, col] = 1
        W[(k_base + 1) * n + (i_base + 1), col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[i_base * m + j_base, col] = 0
        U[(i_base + 1) * m + j_base, col] = 1
        U[(i_base + 1) * m + (j_base + 1), col] = 1
        V[j_base * p + k_base, col] = 1
        W[k_base * n + (i_base + 1), col] = 1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[i_base * m + j_base, col] = 1
        V[(j_base + 1) * p + k_base, col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = -1
        W[k_base * n + i_base, col] = 1
        W[(k_base + 1) * n + i_base, col] = 1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[(i_base + 1) * m + (j_base + 1), col] = 1
        V[j_base * p + (k_base + 1), col] = 1
        V[j_base * p + k_base, col] = -1
        W[k_base * n + (i_base + 1), col] = 1
        W[(k_base + 1) * n + (i_base + 1), col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[i_base * m + j_base, col] = 1
        U[i_base * m + (j_base + 1), col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = 1
        W[(k_base + 1) * n + i_base, col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[i_base * m + j_base, col] = -1
        U[(i_base + 1) * m + j_base, col] = 1
        V[j_base * p + k_base, col] = 1
        V[j_base * p + (k_base + 1), col] = 1
        W[k_base * n + (i_base + 1), col] = 1
        W[(k_base + 1) * n + (i_base + 1), col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i_base * m + (j_base + 1), col] = 1
        U[(i_base + 1) * m + (j_base + 1), col] = -1
        V[(j_base + 1) * p + k_base, col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = 1
        W[k_base * n + i_base, col] = 1
        W[(k_base + 1) * n + i_base, col] = 1
        col += 1
        
        # Now we must correct the W matrix to match the Strassen output formula:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We have added coefficients to W above. Let's trace them.
        # M1: W[k0,i0]+=1, W[k1,i1]+=1
        # M2: W[k0,i1]+=1
        # M3: W[k0,i0]+=1, W[k1,i0]+=1
        # M4: W[k0,i1]+=1, W[k1,i1]+=1
        # M5: W[k1,i0]+=1
        # M6: W[k0,i1]+=1, W[k1,i1]+=1
        # M7: W[k0,i0]+=1, W[k1,i0]+=1
        
        # Current W sums:
        # W[k0,i0]: 1(M1) + 1(M3) + 1(M7) = 3. Target: 1 (from M1) + 0 + 0 + 1 (M7) = 2?
        # Formula: C00 = M1 + M4 - M5 + M7.
        # Coeff of M1 in C00 is 1.
        # Coeff of M4 in C00 is 1.
        # Coeff of M5 in C00 is -1.
        # Coeff of M7 in C00 is 1.
        
        # My add_strassen logic above just added 1s to W. I need to set the coefficients correctly.
        # It is easier to clear the W entries for this block and set them explicitly.
        
        # Let's restart the Strassen implementation with explicit W coefficients.
        
    # The above helper is getting complex. Let's just use the schoolbook for the 2x2x2 parts 
    # if Strassen is too risky to implement correctly in this constrained format without verification.
    # Schoolbook for 2x2x2 is rank 4. Strassen is 7.
    # Using Schoolbook everywhere:
    # Part 1: 21
    # Part 2a (2x3x3): 18 (Schoolbook)
    # Part 2b (2x3x4): 24 (Schoolbook)
    # Total: 21 + 18 + 24 = 63. No improvement.
    
    # I will stick to the plan of using Strassen for 2x2x2 blocks.
    # Let's rewrite add_strassen to be correct.
    
    def add_strassen_correct(i_base, j_base, k_base):
        nonlocal col
        if col + 7 > rank: return
        
        # M1
        U[i_base * m + j_base, col] = 1
        U[(i_base + 1) * m + (j_base + 1), col] = 1
        V[j_base * p + k_base, col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = 1
        W[k_base * n + i_base, col] = 1
        W[(k_base + 1) * n + (i_base + 1), col] = 1
        col += 1
        
        # M2
        U[(i_base + 1) * m + j_base, col] = 1
        U[(i_base + 1) * m + (j_base + 1), col] = 1
        V[j_base * p + k_base, col] = 1
        W[k_base * n + (i_base + 1), col] = 1
        col += 1
        
        # M3
        U[i_base * m + j_base, col] = 1
        V[(j_base + 1) * p + k_base, col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = -1
        W[k_base * n + i_base, col] = 1
        W[(k_base + 1) * n + i_base, col] = 1
        col += 1
        
        # M4
        U[(i_base + 1) * m + (j_base + 1), col] = 1
        V[j_base * p + (k_base + 1), col] = 1
        V[j_base * p + k_base, col] = -1
        W[k_base * n + (i_base + 1), col] = 1
        W[(k_base + 1) * n + (i_base + 1), col] = 1
        col += 1
        
        # M5
        U[i_base * m + j_base, col] = 1
        U[i_base * m + (j_base + 1), col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = 1
        W[(k_base + 1) * n + i_base, col] = 1
        col += 1
        
        # M6
        U[i_base * m + j_base, col] = -1
        U[(i_base + 1) * m + j_base, col] = 1
        V[j_base * p + k_base, col] = 1
        V[j_base * p + (k_base + 1), col] = 1
        W[k_base * n + (i_base + 1), col] = 1
        W[(k_base + 1) * n + (i_base + 1), col] = 1
        col += 1
        
        # M7
        U[i_base * m + (j_base + 1), col] = 1
        U[(i_base + 1) * m + (j_base + 1), col] = -1
        V[(j_base + 1) * p + k_base, col] = 1
        V[(j_base + 1) * p + (k_base + 1), col] = 1
        W[k_base * n + i_base, col] = 1
        W[(k_base + 1) * n + i_base, col] = 1
        col += 1
        
        # Apply Strassen correction to W
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices for W:
        # w00 = k_base * n + i_base
        # w01 = (k_base + 1) * n + i_base
        # w10 = k_base * n + (i_base + 1)
        # w11 = (k_base + 1) * n + (i_base + 1)
        
        # Current W values from above additions:
        # M1: w00=1, w11=1
        # M2: w10=1
        # M3: w00=1, w01=1
        # M4: w10=1, w11=1
        # M5: w01=1
        # M6: w10=1, w11=1
        # M7: w00=1, w01=1
        
        # Sum:
        # w00 = 1+1+1 = 3. Target: 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2.
        # w01 = 1+1+1 = 3. Target: 1 (M3) + 1 (M5) = 2.
        # w10 = 1+1+1 = 3. Target: 1 (M2) + 1 (M4) = 2.
        # w11 = 1+1+1 = 3. Target: 1 (M1) - 1 (M2) + 1 (M3) + 1 (M6) = 2.
        
        # We need to subtract 1 from each.
        W[w00] -= 1
        W[w01] -= 1
        W[w10] -= 1
        W[w11] -= 1
        # Wait, w00 index is k_base * n + i_base.
        W[k_base * n + i_base] -= 1
        W[(k_base + 1) * n + i_base] -= 1
        W[k_base * n + (i_base + 1)] -= 1
        W[(k_base + 1) * n + (i_base + 1)] -= 1
        
    # Part 1: i=0
    for j in range(3):
        for k in range(7):
            add_term(0, j, k, 1)
            
    # Part 2a: i in 1,2; k in 0,1,2
    # 2a-i: j in 0,1; k in 0,1 (2x2x2 Strassen)
    add_strassen_correct(1, 0, 0)
    
    # 2a-i-b: j in 0,1; k=2 (2x2x1 Schoolbook)
    add_term(1, 0, 2, 1)
    add_term(1, 1, 2, 1)
    add_term(2, 0, 2, 1)
    add_term(2, 1, 2, 1)
    
    # 2a-ii: j=2; k in 0,1 (2x1x2 Schoolbook)
    add_term(1, 2, 0, 1)
    add_term(1, 2, 1, 1)
    add_term(2, 2, 0, 1)
    add_term(2, 2, 1, 1)
    
    # 2a-ii-b: j=2; k=2 (2x1x1 Schoolbook)
    add_term(1, 2, 2, 1)
    add_term(2, 2, 2, 1)
    
    # Part 2b: i in 1,2; k in 3,4,5,6
    # 2b-i-a: j in 0,1; k in 3,4 (2x2x2 Strassen)
    add_strassen_correct(1, 0, 3)
    
    # 2b-i-b: j in 0,1; k in 5,6 (2x2x2 Strassen)
    add_strassen_correct(1, 0, 5)
    
    # 2b-ii-a: j=2; k in 3,4 (2x1x2 Schoolbook)
    add_term(1, 2, 3, 1)
    add_term(1, 2, 4, 1)
    add_term(2, 2, 3, 1)
    add_term(2, 2, 4, 1)
    
    # 2b-ii-b: j=2; k in 5,6 (2x1x2 Schoolbook)
    add_term(1, 2, 5, 1)
    add_term(1, 2, 6, 1)
    add_term(2, 2, 5, 1)
    add_term(2, 2, 6, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
