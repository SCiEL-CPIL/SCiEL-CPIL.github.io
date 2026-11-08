"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Target rank estimate: 87
    # We will build the decomposition dynamically
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        """
        u_idx: tuple (i, j) for U
        v_idx: tuple (j, k) for V
        w_idx: tuple (k, i) for W
        """
        # Convert to flat indices
        u_flat = u_idx[0] * m + u_idx[1]
        v_flat = v_idx[0] * p + v_idx[1]
        w_flat = w_idx[0] * n + w_idx[1]
        terms.append((u_flat, v_flat, w_flat, val))
        
    # Helper for Strassen 2x2x2
    # Indices relative to local 2x2x2
    def add_strassen_2x2x2(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices (0-based)
        # Local i: 0, 1. Local j: 0, 1. Local k: 0, 1.
        
        # m1 = (a11+a22)(b11+b22)
        # u: a11 (i=0,j=0), a22 (i=1,j=1)
        # v: b11 (j=0,k=0), b22 (j=1,k=1)
        # w: c11 (k=0,i=0), c12 (k=1,i=0), c21 (k=0,i=1), c22 (k=1,i=1)
        # C11 = m1 - m4 + m5 - m7
        # C12 = m3 + m7
        # C21 = m2 + m4
        # C22 = m1 - m3 + m6 + m7
        
        # We add 7 terms.
        # Term 1: m1
        # u: (i=0,j=0), (i=1,j=1)
        # v: (j=0,k=0), (j=1,k=1)
        # w: (k=0,i=0), (k=1,i=0), (k=0,i=1), (k=1,i=1) with coeffs
        # This is not a single rank-1 term in the standard sense unless we expand.
        # Strassen multiplies sums.
        # (a11+a22) is sum of u components.
        # So the term is:
        # u vector has 1s at (0,0) and (1,1).
        # v vector has 1s at (0,0) and (1,1).
        # w vector has coeffs for c11, c12, c21, c22.
        
        # Let's implement the 7 Strassen terms explicitly as rank-1 tensors.
        
        # m1
        u1 = [(0,0), (1,1)]
        v1 = [(0,0), (1,1)]
        w1 = [(0,0, 1), (1,0, 1), (0,1, 1), (1,1, 1)] # (k,i, coeff)
        # C11 gets +1, C12 gets +1, C21 gets +1, C22 gets +1
        # Wait, m1 contributes to all 4 with specific signs in the final sum.
        # It is easier to just define the 7 multipliers and then apply the linear combinations.
        # But the output requires U, V, W matrices such that sum U[:,r] V[:,r] == T.
        # So each column r in U,V,W defines a rank-1 tensor.
        # For m1:
        # U_col has 1 at (i=0,j=0) and (i=1,j=1).
        # V_col has 1 at (j=0,k=0) and (j=1,k=1).
        # W_col must be such that U V outer product W matches m1 contribution.
        # m1 = (a11+a22)(b11+b22)
        # Expansion: a11b11 + a11b22 + a22b11 + a22b22.
        # This is a sum of 4 rank-1 tensors.
        # But Strassen says we only need 7 multiplications.
        # The "multiplication" is the dot product of the U and V vectors.
        # The result is a scalar $s$.
        # Then we add $s \cdot w$ to the result.
        # So W vector contains the coefficients for the output C.
        
        # So for m1:
        # U has 1 at (0,0) and (1,1).
        # V has 1 at (0,0) and (1,1).
        # W has coefficients for C11, C12, C21, C22.
        # C11 += s
        # C12 += s
        # C21 += s
        # C22 += s
        # So W has 1 at (0,0), (1,0), (0,1), (1,1). (k,i)
        
        def add_rank1(u_list, v_list, w_list):
            # u_list: list of (i,j)
            # v_list: list of (j,k)
            # w_list: list of (k,i)
            # All have coefficient 1.
            # We need to find a column index for this term.
            # We just append to terms.
            pass
        
        # This approach is complex to encode directly as single rank-1 terms if the U/V vectors have multiple non-zeros.
        # The definition of tensor rank decomposition allows U, V, W to have any entries.
        # So U[:, r] can be sparse vector.
        # So we can just define the vectors.
        
        # Let's collect the vectors for the 7 Strassen terms.
        # m1
        u_m1 = {(0,0): 1, (1,1): 1}
        v_m1 = {(0,0): 1, (1,1): 1}
        w_m1 = {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}
        
        # m2 = (a21+a22)b11
        u_m2 = {(1,0): 1, (1,1): 1}
        v_m2 = {(0,0): 1}
        w_m2 = {(0,1): 1} # C21 += m2
        
        # m3 = a11(b12-b22)
        u_m3 = {(0,0): 1}
        v_m3 = {(0,1): 1, (1,1): -1}
        w_m3 = {(1,0): 1, (1,1): -1} # C12 += m3, C22 += -m3? 
        # C12 = m3 + m7. C22 = m1 - m3 + m6 + m7.
        # So w_m3 has 1 at (k=1, i=0) and -1 at (k=1, i=1).
        
        # m4 = a22(b21-b11)
        u_m4 = {(1,1): 1}
        v_m4 = {(1,0): 1, (0,0): -1}
        w_m4 = {(0,1): 1, (0,0): -1} # C21 += m4, C11 += -m4
        
        # m5 = (a11+a12)b22
        u_m5 = {(0,0): 1, (0,1): 1}
        v_m5 = {(1,1): 1}
        w_m5 = {(0,0): 1, (1,0): 1} # C11 += m5, C12 += m5? 
        # C11 = ... + m5 ...
        # C12 = m3 + m7. No m5 in C12.
        # C11 = m1 - m4 + m5 - m7.
        # C22 = m1 - m3 + m6 + m7.
        # So m5 only in C11?
        # Let's check Strassen again.
        # C11 = m1 - m4 + m5 - m7
        # C12 = m3 + m7
        # C21 = m2 + m4
        # C22 = m1 - m3 + m6 + m7
        # Yes, m5 is only in C11.
        
        # m6 = (a21-a11)(b11+b12)
        u_m6 = {(1,0): 1, (0,0): -1}
        v_m6 = {(0,0): 1, (0,1): 1}
        w_m6 = {(1,1): 1} # C22 += m6
        
        # m7 = (a12-a22)(b21+b22)
        u_m7 = {(0,1): 1, (1,1): -1}
        v_m7 = {(1,0): 1, (1,1): 1}
        w_m7 = {(0,0): -1, (1,0): 1, (1,1): 1} # C11 -= m7, C12 += m7, C22 += m7
        
        # We need to add these to the global terms.
        # For each term r, we create a column in U, V, W.
        
        # m1
        terms.append((
            { (i_base+0, j_base+0): 1, (i_base+1, j_base+1): 1 },
            { (j_base+0, k_base+0): 1, (j_base+1, k_base+1): 1 },
            { (k_base+0, i_base+0): 1, (k_base+1, i_base+0): 1, (k_base+0, i_base+1): 1, (k_base+1, i_base+1): 1 }
        ))
        
        # m2
        terms.append((
            { (i_base+1, j_base+0): 1, (i_base+1, j_base+1): 1 },
            { (j_base+0, k_base+0): 1 },
            { (k_base+0, i_base+1): 1 }
        ))
        
        # m3
        terms.append((
            { (i_base+0, j_base+0): 1 },
            { (j_base+0, k_base+1): 1, (j_base+1, k_base+1): -1 },
            { (k_base+1, i_base+0): 1, (k_base+1, i_base+1): -1 }
        ))
        
        # m4
        terms.append((
            { (i_base+1, j_base+1): 1 },
            { (j_base+1, k_base+0): 1, (j_base+0, k_base+0): -1 },
            { (k_base+0, i_base+1): 1, (k_base+0, i_base+0): -1 }
        ))
        
        # m5
        terms.append((
            { (i_base+0, j_base+0): 1, (i_base+0, j_base+1): 1 },
            { (j_base+1, k_base+1): 1 },
            { (k_base+0, i_base+0): 1 }
        ))
        
        # m6
        terms.append((
            { (i_base+1, j_base+0): 1, (i_base+0, j_base+0): -1 },
            { (j_base+0, k_base+0): 1, (j_base+0, k_base+1): 1 },
            { (k_base+1, i_base+1): 1 }
        ))
        
        # m7
        terms.append((
            { (i_base+0, j_base+1): 1, (i_base+1, j_base+1): -1 },
            { (j_base+1, k_base+0): 1, (j_base+1, k_base+1): 1 },
            { (k_base+0, i_base+0): -1, (k_base+1, i_base+0): 1, (k_base+1, i_base+1): 1 }
        ))

    def add_schoolbook_1x2x2(i_base, j_base, k_base):
        # i_base is the row index (only 1 row)
        # j_base, k_base start indices
        # A: (i, j0), (i, j1)
        # B: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C: (k0, i), (k1, i)
        
        # c00 = a0 b00 + a1 b10
        # c01 = a0 b01 + a1 b11
        
        # Term 1: a0 b00 -> c00
        terms.append((
            { (i_base, j_base+0): 1 },
            { (j_base+0, k_base+0): 1 },
            { (k_base+0, i_base): 1 }
        ))
        # Term 2: a1 b10 -> c00
        terms.append((
            { (i_base, j_base+1): 1 },
            { (j_base+1, k_base+0): 1 },
            { (k_base+0, i_base): 1 }
        ))
        # Term 3: a0 b01 -> c01
        terms.append((
            { (i_base, j_base+0): 1 },
            { (j_base+0, k_base+1): 1 },
            { (k_base+1, i_base): 1 }
        ))
        # Term 4: a1 b11 -> c01
        terms.append((
            { (i_base, j_base+1): 1 },
            { (j_base+1, k_base+1): 1 },
            { (k_base+1, i_base): 1 }
        ))

    def add_schoolbook_3x3x3(i_base, j_base, k_base):
        # Standard 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    terms.append((
                        { (i_base+i, j_base+j): 1 },
                        { (j_base+j, k_base+k): 1 },
                        { (k_base+k, i_base+i): 1 }
                    ))

    # Block 1: j in {0, 1} (0-based)
    # k blocks: {0,1}, {2,3}, {4,5}
    for k_start in [0, 2, 4]:
        # Strassen for i in {0, 1}
        add_strassen_2x2x2(0, 0, k_start)
        # Schoolbook for i=2
        add_schoolbook_1x2x2(2, 0, k_start)

    # Block 2: j in {2, 3, 4}
    # k blocks: {0,1,2}, {3,4,5}
    for k_start in [0, 3]:
        # Schoolbook for i in {0, 1, 2}
        add_schoolbook_3x3x3(0, 2, k_start)

    # Convert terms to U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
