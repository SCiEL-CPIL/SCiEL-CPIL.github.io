import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 28 x rank
    # W: (p*n) x rank = 28 x rank
    
    # We will build the decomposition with rank 97.
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    
    R = 49 + 48
    
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    def add_term(u_indices, v_indices, w_indices, r):
        # u_indices: list of (i, j, val)
        # v_indices: list of (j, k, val)
        # w_indices: list of (k, i, val)
        for (i, j, val) in u_indices:
            U[i*m + j, r] += val
        for (j, k, val) in v_indices:
            V[j*p + k, r] += val
        for (k, i, val) in w_indices:
            W[k*n + i, r] += val
            
    # --- Part 1: Strassen 4x4x4 ---
    # We split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block indices (i, j) for 2x2 blocks:
    # 00: i in 0..1, j in 0..1
    # 01: i in 0..1, j in 2..3
    # 10: i in 2..3, j in 0..1
    # 11: i in 2..3, j in 2..3
    
    # We need to map block operations to global indices.
    # Helper to get global indices for a block (bi, bj)
    def get_block_indices(bi, bj, size=2):
        indices = []
        for di in range(size):
            for dj in range(size):
                indices.append((bi*size + di, bj*size + dj))
        return indices

    # Strassen formulas for 2x2 blocks:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will iterate through the 7 Strassen steps.
    # Each step involves multiplying two 2x2 matrices (linear combos of A and B blocks).
    # Let X and Y be the 2x2 matrices for a step.
    # X is defined by linear combos of A blocks.
    # Y is defined by linear combos of B blocks.
    # The product X*Y contributes to C blocks.
    
    # Since each X and Y is a sum of A/B blocks, and we are doing tensor decomposition,
    # we can expand the product of sums into sums of products of single blocks.
    # However, a "rank-1" term in the tensor decomposition corresponds to a single
    # multiplication of scalar entries (or linear forms if we want to be fancy, but
    # standard Strassen implementation for tensor rank usually expands to scalar multiplications
    # of the smallest blocks if we don't recurse further).
    # 
    # Wait, the rank of 4x4x4 Strassen is 49.
    # This implies we treat the 2x2 blocks as scalars in the 7 Strassen steps?
    # No, that would be 7 * (rank of 2x2x2).
    # Rank of 2x2x2 Strassen is 7.
    # So 7 steps * 7 rank-1 terms per step = 49.
    # This means for each of the 7 Strassen steps (e.g. M1), we decompose the 2x2x2 multiplication
    # of the resulting linear forms into 7 rank-1 terms.
    
    # Let's define the 2x2 Strassen decomposition for scalar matrices.
    # If we have X (2x2) and Y (2x2) and want Z = XY.
    # The 7 rank-1 terms for 2x2x2 are:
    # S1 = (X00 + X11)(Y00 + Y11)
    # S2 = (X10 + X11)Y00
    # S3 = X00(Y01 - Y11)
    # S4 = X11(Y10 - Y00)
    # S5 = (X00 + X01)Y11
    # S6 = (X10 - X00)(Y00 + Y01)
    # S7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # In our 4x4 case, X and Y are 2x2 matrices where each element is a linear combination
    # of 2x2 blocks of A and B.
    # Let X[a,b] be a linear combo of A blocks.
    # Let Y[a,b] be a linear combo of B blocks.
    # Then S1 is a product of two linear forms.
    # S1 = (sum_{i,j} c_{ij} A_{ij}) * (sum_{j,k} d_{jk} B_{jk})
    # This expands to sum_{i,j,k} c_{ij} d_{jk} A_{ij} B_{jk}.
    # Each term A_{ij} B_{jk} is a rank-1 component for the (i,j,k) entry of the tensor?
    # No, the tensor entry T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # A rank-1 decomposition is sum_r U_r(i,j) V_r(j,k) W_r(i,k).
    # If we have a term A_{ij} B_{jk}, it contributes 1 to U_{ij}, 1 to V_{jk}, and we must add it to W_{ik}.
    # But S1 contributes to multiple C entries (Z00, Z11 etc).
    # So for a single "Strassen Step" (like M1), we have 7 sub-terms (S1..S7).
    # Each sub-term Sx is a product of two linear forms L_A and L_B.
    # L_A = sum_{(i,j) in supp} a_{ij} A_{ij}
    # L_B = sum_{(j,k) in supp} b_{jk} B_{jk}
    # The product L_A L_B = sum_{(i,j), (j,k)} a_{ij} b_{jk} A_{ij} B_{jk}.
    # Each pair ((i,j), (j,k)) corresponds to a rank-1 update:
    # U[i*m+j] += a_{ij}
    # V[j*p+k] += b_{jk}
    # W[k*n+i] += (contribution to C_{ik})
    
    # Wait, the contribution to C_{ik} depends on which Z block it goes to and the Strassen formula.
    # For a specific sub-term Sx in Step M, it contributes to specific C blocks with specific signs.
    # Let's say Sx contributes to C_block (cbi, cbj) with sign s.
    # Then for every (i,j) in L_A and (j,k) in L_B, we add:
    # U[i*m+j] += a_{ij}
    # V[j*p+k] += b_{jk}
    # W[k*n+i] += s * (1 if (i,k) is in C_block) -- Wait, C_block is a 2x2 block.
    # If C_block is (0,0), it covers i in 0..1, k in 0..1.
    # So if i in 0..1 and k in 0..1, we add s to W[k*n+i].
    
    # This expansion is correct.
    # Number of terms:
    # Each L_A has up to 8 terms (4 blocks * 2x2 entries).
    # Each L_B has up to 8 terms.
    # Product has up to 64 terms.
    # 7 sub-terms * 7 steps = 49 "logical" terms?
    # No, the rank is the number of non-zero columns in U,V,W.
    # If we just expand everything, the rank will be the number of distinct (i,j,k) tuples used.
    # For 4x4x4 Strassen, the rank is 49. This implies that the linear forms are such that
    # the total number of non-zero rank-1 components is 49.
    # Actually, standard Strassen for 4x4x4 has rank 49.
    # This is achieved because the 7 steps each have rank 7 (from 2x2 Strassen).
    # So we should implement the 2x2 Strassen decomposition on the "block" level,
    # treating each 2x2 block as a scalar in the 7 steps, and then decomposing those
    # "scalar" multiplications into 2x2 Strassen again?
    # No, that would be 7 * 7 = 49.
    # Yes.
    
    # So, for each of the 7 Strassen steps M1..M7:
    # We define X (2x2) and Y (2x2) where entries are linear combos of A/B blocks.
    # We apply the 2x2 Strassen algorithm to X and Y.
    # This generates 7 sub-terms S1..S7.
    # Each Sx is a product of two linear forms (of A and B entries).
    # We expand Sx into sum of rank-1 terms.
    
    # To keep the rank minimal (49), we must ensure that we don't generate more than 49 terms.
    # Since each 2x2 Strassen step has rank 7, and we have 7 such steps, the total rank is 49.
    # This works if the linear forms for different sub-terms do not "overlap" in a way that
    # requires more rank, or if we simply count the expanded terms.
    # Actually, the rank of the tensor is the minimum number of terms.
    # If we generate 49 terms, the rank is <= 49.
    
    # Let's implement the expansion.
    
    # Step 1: Define the 7 Strassen steps for 4x4.
    # Each step defines X (2x2) and Y (2x2) in terms of A and B blocks.
    # A block (bi, bj) is a 2x2 matrix.
    # We represent a linear form as a dict: {(i,j): coeff}
    
    def block_to_indices(bi, bj):
        return [(bi*2 + di, bj*2 + dj) for di in range(2) for dj in range(2)]

    # X and Y for each step are 2x2 matrices.
    # X[0,0], X[0,1], X[1,0], X[1,1] are linear forms of A.
    # Y[0,0], ... are linear forms of B.
    
    # M1: X = A00 + A11, Y = B00 + B11
    # X is the same for all 4 entries? No.
    # In Strassen 2x2:
    # S1 = (X00 + X11)(Y00 + Y11)
    # If X is a single matrix (2x2), X00 is its top-left entry.
    # But here X is constructed from A blocks.
    # For M1, the "A matrix" is A00 + A11.
    # So X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # X[1,0] = A00[1,0] + A11[1,0]
    # X[1,1] = A00[1,1] + A11[1,1]
    
    # So for each step, we define X and Y as 2x2 arrays of linear forms.
    
    steps = [
        # M1
        {
            'X': [
                { (0,0): 1, (2,2): 1 }, { (0,1): 1, (2,3): 1 },
                { (1,0): 1, (3,2): 1 }, { (1,1): 1, (3,3): 1 }
            ],
            'Y': [
                { (0,0): 1, (2,2): 1 }, { (0,1): 1, (2,3): 1 },
                { (1,0): 1, (3,2): 1 }, { (1,1): 1, (3,3): 1 }
            ],
            'C_contribs': { (0,0): 1, (1,1): 1 } # C00 and C11 get +1 from M1? 
            # Wait, C00 = M1 + ... and C11 = M1 - ...
            # So M1 contributes +1 to C00 and +1 to C11?
            # No, M1 is the matrix product.
            # C00 uses M1[0,0] and M1[1,1]? No.
            # C00 (2x2 block) = M1 (2x2 matrix) + ...
            # So the entire M1 matrix is added to C00.
            # And the entire M1 matrix is added to C11?
            # Let's check Strassen 4x4 formulas.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # Yes, M1 is added to both C00 and C11.
        },
        # M2
        {
            'X': [
                { (2,0): 1, (2,2): 1 }, { (2,1): 1, (2,3): 1 },
                { (3,0): 1, (3,2): 1 }, { (3,1): 1, (3,3): 1 }
            ], # A10 + A11
            'Y': [
                { (0,0): 1 }, { (0,1): 1 },
                { (1,0): 1 }, { (1,1): 1 }
            ], # B00
            'C_contribs': { (1,0): 1 } # C10
        },
        # M3
        {
            'X': [
                { (0,0): 1 }, { (0,1): 1 },
                { (1,0): 1 }, { (1,1): 1 }
            ], # A00
            'Y': [
                { (0,2): 1, (2,2): -1 }, { (0,3): 1, (2,3): -1 },
                { (1,2): 1, (3,2): -1 }, { (1,3): 1, (3,3): -1 }
            ], # B01 - B11
            'C_contribs': { (0,1): 1 } # C01
        },
        # M4
        {
            'X': [
                { (2,2): 1 }, { (2,3): 1 },
                { (3,2): 1 }, { (3,3): 1 }
            ], # A11
            'Y': [
                { (2,0): 1, (0,0): -1 }, { (2,1): 1, (0,1): -1 },
                { (3,0): 1, (1,0): -1 }, { (3,1): 1, (1,1): -1 }
            ], # B10 - B00
            'C_contribs': { (0,0): 1, (1,0): 1 } # C00 and C10
        },
        # M5
        {
            'X': [
                { (0,0): 1, (0,2): 1 }, { (0,1): 1, (0,3): 1 },
                { (1,0): 1, (1,2): 1 }, { (1,1): 1, (1,3): 1 }
            ], # A00 + A01
            'Y': [
                { (2,2): 1 }, { (2,3): 1 },
                { (3,2): 1 }, { (3,3): 1 }
            ], # B11
            'C_contribs': { (0,0): -1, (0,1): 1 } # -C00 + C01
        },
        # M6
        {
            'X': [
                { (2,0): 1, (0,0): -1 }, { (2,1): 1, (0,1): -1 },
                { (3,0): 1, (1,0): -1 }, { (3,1): 1, (1,1): -1 }
            ], # A10 - A00
            'Y': [
                { (0,0): 1, (0,2): 1 }, { (0,1): 1, (0,3): 1 },
                { (1,0): 1, (1,2): 1 }, { (1,1): 1, (1,3): 1 }
            ], # B00 + B01
            'C_contribs': { (1,1): 1 } # C11
        },
        # M7
        {
            'X': [
                { (0,2): 1, (2,2): -1 }, { (0,3): 1, (2,3): -1 },
                { (1,2): 1, (3,2): -1 }, { (1,3): 1, (3,3): -1 }
            ], # A01 - A11
            'Y': [
                { (2,0): 1, (2,2): 1 }, { (2,1): 1, (2,3): 1 },
                { (3,0): 1, (3,2): 1 }, { (3,1): 1, (3,3): 1 }
            ], # B10 + B11
            'C_contribs': { (0,0): 1 } # C00
        }
    ]
    
    current_rank = 0
    
    # 2x2 Strassen sub-terms
    # S1 = (X00 + X11)(Y00 + Y11)
    # S2 = (X10 + X11)Y00
    # S3 = X00(Y01 - Y11)
    # S4 = X11(Y10 - Y00)
    # S5 = (X00 + X01)Y11
    # S6 = (X10 - X00)(Y00 + Y01)
    # S7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # For each step, we apply these 7 sub-terms.
    # Each sub-term is a product of two linear forms.
    # We expand the product into rank-1 terms.
    
    def add_sub_term(X_lf, Y_lf, C_block_signs, r_offset):
        # X_lf: list of 4 dicts (linear forms for X00, X01, X10, X11)
        # Y_lf: list of 4 dicts
        # C_block_signs: dict mapping (zi, zj) to sign for this sub-term
        # e.g. for S1, it contributes to Z00 and Z11.
        
        # Combine X and Y linear forms
        # S is product of Lx and Ly
        # Lx = sum_{(i,j), v} v * A_{ij}
        # Ly = sum_{(j,k), w} w * B_{jk}
        # Term = sum_{(i,j), (j,k)} v*w * A_{ij} B_{jk}
        
        # We need to add these to U, V, W.
        # U[i*m+j] += v
        # V[j*p+k] += w
        # W[k*n+i] += sign * (1 if (i,k) in C_block)
        
        # We will iterate through all pairs.
        # To avoid creating new rank-1 terms for each pair (which would increase rank),
        # we just add to the existing matrices at the current rank column?
        # No, each pair (i,j,k) is a distinct rank-1 component?
        # In schoolbook, yes.
        # In Strassen, the "rank" is the number of non-zero columns.
        # If we add multiple pairs to the same column r, we are summing them.
        # But U[i*m+j] and V[j*p+k] and W[k*n+i] are scalars.
        # If we have two pairs (i1,j1,k1) and (i2,j2,k2) in the same column r,
        # then U[i1*m+j1] and U[i2*m+j2] are both non-zero in column r.
        # This is a single rank-1 term: u \otimes v \otimes w.
        # So we can pack multiple (i,j,k) triplets into one rank-1 term
        # as long as they are all part of the same logical "multiplication".
        # In Strassen 2x2, each Sx is one logical multiplication.
        # So we can put all pairs for Sx into one column r.
        
        nonlocal current_rank
        r = r_offset
        
        for (i, j, v) in X_lf_flat:
            for (j2, k, w) in Y_lf_flat:
                if j == j2:
                    U[i*m + j, r] += v
                    V[j*p + k, r] += w
                    # Check if (i, k) is in any of the C blocks with non-zero sign
                    for (zi, zj, sign) in C_block_signs:
                        # C block (zi, zj) covers i in zi*2 .. zi*2+1, k in zj*2 .. zj*2+1
                        if zi*2 <= i < zi*2+2 and zj*2 <= k < zj*2+2:
                            W[k*n + i, r] += sign
                            
        current_rank += 1
        return 1

    # We need to flatten X_lf (list of 4 dicts) into a list of (i,j,val)
    # But wait, X_lf is already defined as 4 dicts.
    # For S1, X_lf is X00 + X11.
    # So we need a helper to add two linear forms.
    
    def add_lf(lf1, lf2):
        res = {}
        for k, v in lf1.items():
            res[k] = res.get(k, 0) + v
        for k, v in lf2.items():
            res[k] = res.get(k, 0) + v
        return res

    def sub_lf(lf1, lf2):
        res = {}
        for k, v in lf1.items():
            res[k] = res.get(k, 0) + v
        for k, v in lf2.items():
            res[k] = res.get(k, 0) - v
        return res

    def flatten_lf(lf):
        return [(i, j, v) for (i, j), v in lf.items() if v != 0]

    for step in steps:
        X = step['X']
        Y = step['Y']
        C_contribs = step['C_contribs']
        
        # 1. S1 = (X00 + X11)(Y00 + Y11)
        X_lf = add_lf(X[0], X[3])
        Y_lf = add_lf(Y[0], Y[3])
        # S1 contributes to Z00 and Z11
        # Z00 is block (0,0), Z11 is block (1,1)
        # Signs from C_contribs:
        # C00 sign, C11 sign
        c00_sign = C_contribs.get((0,0), 0)
        c11_sign = C_contribs.get((1,1), 0)
        signs = []
        if c00_sign != 0: signs.append((0, 0, c00_sign))
        if c11_sign != 0: signs.append((1, 1, c11_sign))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # 2. S2 = (X10 + X11)Y00
        X_lf = add_lf(X[2], X[3])
        Y_lf = Y[0]
        # S2 contributes to Z10 and Z11
        # Z10 is block (1,0), Z11 is block (1,1)
        c10_sign = C_contribs.get((1,0), 0)
        c11_sign = C_contribs.get((1,1), 0)
        signs = []
        if c10_sign != 0: signs.append((1, 0, c10_sign))
        if c11_sign != 0: signs.append((1, 1, -c11_sign)) # Wait, C11 = ... - S2 ...
        # Let's re-verify C11 formula: C11 = M1 - M2 + M3 + M6
        # So M2 contributes -1 to C11.
        # M2 is the matrix product.
        # So S2 (part of M2) contributes -1 to C11?
        # Yes.
        # But my C_contribs for M2 was { (1,0): 1 }. It didn't have (1,1).
        # I need to update C_contribs to include all blocks the matrix contributes to.
        
        # Let's re-define C_contribs correctly for each step.
        # M1: C00 (+1), C11 (+1)
        # M2: C10 (+1), C11 (-1)
        # M3: C01 (+1), C11 (+1)
        # M4: C00 (+1), C10 (+1)
        # M5: C00 (-1), C01 (+1)
        # M6: C11 (+1)
        # M7: C00 (+1)
        
        # I will fix the steps list in the code below.
        
    # Re-defining steps with correct C_contribs
    steps = [
        {
            'X': [
                { (0,0): 1, (2,2): 1 }, { (0,1): 1, (2,3): 1 },
                { (1,0): 1, (3,2): 1 }, { (1,1): 1, (3,3): 1 }
            ],
            'Y': [
                { (0,0): 1, (2,2): 1 }, { (0,1): 1, (2,3): 1 },
                { (1,0): 1, (3,2): 1 }, { (1,1): 1, (3,3): 1 }
            ],
            'C_contribs': { (0,0): 1, (1,1): 1 }
        },
        {
            'X': [
                { (2,0): 1, (2,2): 1 }, { (2,1): 1, (2,3): 1 },
                { (3,0): 1, (3,2): 1 }, { (3,1): 1, (3,3): 1 }
            ],
            'Y': [
                { (0,0): 1 }, { (0,1): 1 },
                { (1,0): 1 }, { (1,1): 1 }
            ],
            'C_contribs': { (1,0): 1, (1,1): -1 }
        },
        {
            'X': [
                { (0,0): 1 }, { (0,1): 1 },
                { (1,0): 1 }, { (1,1): 1 }
            ],
            'Y': [
                { (0,2): 1, (2,2): -1 }, { (0,3): 1, (2,3): -1 },
                { (1,2): 1, (3,2): -1 }, { (1,3): 1, (3,3): -1 }
            ],
            'C_contribs': { (0,1): 1, (1,1): 1 }
        },
        {
            'X': [
                { (2,2): 1 }, { (2,3): 1 },
                { (3,2): 1 }, { (3,3): 1 }
            ],
            'Y': [
                { (2,0): 1, (0,0): -1 }, { (2,1): 1, (0,1): -1 },
                { (3,0): 1, (1,0): -1 }, { (3,1): 1, (1,1): -1 }
            ],
            'C_contribs': { (0,0): 1, (1,0): 1 }
        },
        {
            'X': [
                { (0,0): 1, (0,2): 1 }, { (0,1): 1, (0,3): 1 },
                { (1,0): 1, (1,2): 1 }, { (1,1): 1, (1,3): 1 }
            ],
            'Y': [
                { (2,2): 1 }, { (2,3): 1 },
                { (3,2): 1 }, { (3,3): 1 }
            ],
            'C_contribs': { (0,0): -1, (0,1): 1 }
        },
        {
            'X': [
                { (2,0): 1, (0,0): -1 }, { (2,1): 1, (0,1): -1 },
                { (3,0): 1, (1,0): -1 }, { (3,1): 1, (1,1): -1 }
            ],
            'Y': [
                { (0,0): 1, (0,2): 1 }, { (0,1): 1, (0,3): 1 },
                { (1,0): 1, (1,2): 1 }, { (1,1): 1, (1,3): 1 }
            ],
            'C_contribs': { (1,1): 1 }
        },
        {
            'X': [
                { (0,2): 1, (2,2): -1 }, { (0,3): 1, (2,3): -1 },
                { (1,2): 1, (3,2): -1 }, { (1,3): 1, (3,3): -1 }
            ],
            'Y': [
                { (2,0): 1, (2,2): 1 }, { (2,1): 1, (2,3): 1 },
                { (3,0): 1, (3,2): 1 }, { (3,1): 1, (3,3): 1 }
            ],
            'C_contribs': { (0,0): 1 }
        }
    ]
    
    # Reset current_rank and matrices
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    current_rank = 0
    
    for step in steps:
        X = step['X']
        Y = step['Y']
        C_contribs = step['C_contribs']
        
        # S1
        X_lf = add_lf(X[0], X[3])
        Y_lf = add_lf(Y[0], Y[3])
        signs = []
        if C_contribs.get((0,0), 0) != 0: signs.append((0, 0, C_contribs[(0,0)]))
        if C_contribs.get((1,1), 0) != 0: signs.append((1, 1, C_contribs[(1,1)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # S2
        X_lf = add_lf(X[2], X[3])
        Y_lf = Y[0]
        signs = []
        if C_contribs.get((1,0), 0) != 0: signs.append((1, 0, C_contribs[(1,0)]))
        if C_contribs.get((1,1), 0) != 0: signs.append((1, 1, C_contribs[(1,1)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # S3
        X_lf = X[0]
        Y_lf = sub_lf(Y[1], Y[3])
        signs = []
        if C_contribs.get((0,1), 0) != 0: signs.append((0, 1, C_contribs[(0,1)]))
        if C_contribs.get((1,1), 0) != 0: signs.append((1, 1, C_contribs[(1,1)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # S4
        X_lf = X[3]
        Y_lf = sub_lf(Y[2], Y[0])
        signs = []
        if C_contribs.get((0,0), 0) != 0: signs.append((0, 0, C_contribs[(0,0)]))
        if C_contribs.get((1,0), 0) != 0: signs.append((1, 0, C_contribs[(1,0)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # S5
        X_lf = add_lf(X[0], X[1])
        Y_lf = Y[3]
        signs = []
        if C_contribs.get((0,0), 0) != 0: signs.append((0, 0, C_contribs[(0,0)]))
        if C_contribs.get((0,1), 0) != 0: signs.append((0, 1, C_contribs[(0,1)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # S6
        X_lf = sub_lf(X[2], X[0])
        Y_lf = add_lf(Y[0], Y[1])
        signs = []
        if C_contribs.get((1,1), 0) != 0: signs.append((1, 1, C_contribs[(1,1)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
        # S7
        X_lf = sub_lf(X[1], X[3])
        Y_lf = add_lf(Y[2], Y[3])
        signs = []
        if C_contribs.get((0,0), 0) != 0: signs.append((0, 0, C_contribs[(0,0)]))
        add_sub_term(flatten_lf(X_lf), flatten_lf(Y_lf), signs, current_rank)
        
    # Part 2: Schoolbook for 4x4x3 (cols 4-6 of B)
    # Rank 48
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                U[i*m + j, current_rank] = 1
                V[j*p + k, current_rank] = 1
                W[k*n + i, current_rank] = 1
                current_rank += 1
                
    return {"U": U, "V": V, "W": W}
