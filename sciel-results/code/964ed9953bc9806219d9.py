"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + 48 for 4x4x3 (Schoolbook) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U_idx, V_idx, W_idx, sign
    # Actually, we need to add linear combinations.
    # U is a vector of coefficients for A elements.
    # V is a vector of coefficients for B elements.
    # W is a vector of coefficients for C elements.
    # A term is a product of a linear combo of A and a linear combo of B.
    # This maps to a single column in U, V, W if we expand?
    # No, tensor decomposition $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # $T_{ijk} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # Each column r corresponds to one "multiplication" step.
    # If a step involves $(A_a + A_b)(B_c + B_d)$, it expands to 4 terms:
    # $A_a B_c + A_a B_d + A_b B_c + A_b B_d$.
    # This would require 4 rank-1 tensors.
    # BUT Strassen's algorithm works by defining intermediate sums.
    # The "multiplication" in Strassen is a scalar multiplication of two *computed sums*.
    # In the tensor decomposition, each "scalar multiplication" of the algorithm corresponds to one rank-1 term?
    # No. The tensor decomposition rank is the number of scalar multiplications in the straight-line program.
    # Strassen's algorithm for $2 \times 2$ uses 7 multiplications.
    # Each multiplication is $X \cdot Y$ where $X$ is a linear combo of A's and $Y$ is a linear combo of B's.
    # The result $Z = X \cdot Y$ is then added to the final C with some coefficient.
    # So, yes, each of the 7 Strassen multiplications corresponds to ONE column in U, V, W.
    # $U$ contains the coefficients of $X$.
    # $V$ contains the coefficients of $Y$.
    # $W$ contains the coefficients of how $Z$ contributes to C.
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        # u_coeffs: dict { (i, j): coeff } for A
        # v_coeffs: dict { (j, k): coeff } for B
        # w_coeffs: dict { (i, k): coeff } for C
        for (i, j), c in u_coeffs.items():
            if c != 0:
                U[i * m + j][col_idx] += c
        for (j, k), c in v_coeffs.items():
            if c != 0:
                V[j * p + k][col_idx] += c
        for (i, k), c in w_coeffs.items():
            if c != 0:
                W[k * n + i][col_idx] += c

    # Part 1: 4x4x4 using Strassen
    # Block indices I, J, K in {0, 1}
    # A[I, J] is 2x2. A[I, J][a, b] corresponds to A[I*2+a, J*2+b]
    
    # Strassen steps for 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to expand each M_k into 7 scalar multiplications using Strassen 2x2.
    # Let's define a function that takes the linear combos for A_block and B_block
    # and generates the 7 U,V,W columns for that block multiplication.
    
    def expand_block_mult(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        # a_block_coeffs: dict {(I,J): coeff} where I,J in {0,1} for the 2x2 block
        # Actually, let's pass the global indices.
        # Let's define the 7 Strassen steps for a 2x2 matrix mult.
        # Inputs: A (2x2), B (2x2).
        # Outputs: 7 columns.
        
        # Step 1: (A00+A11)(B00+B11)
        u1 = { (0,0): 1, (1,1): 1 }
        v1 = { (0,0): 1, (1,1): 1 }
        w1 = { (0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1 } # This is wrong. W depends on how M1 is used.
        
        # This approach is getting complicated because W coefficients depend on the outer Strassen level.
        # Let's do it recursively or explicitly.
        
        pass

    # Explicit construction for 4x4x4
    # There are 7 block multiplications. Each has 7 scalar multiplications.
    # Total 49 columns.
    
    # Let's define the 7 block multiplications' A and B linear combos.
    # A_block(I, J) refers to the 2x2 submatrix at block row I, col J.
    # A_block(I, J)[a, b] is A[I*2+a, J*2+b].
    
    # M1: A=(A00+A11), B=(B00+B11)
    # M2: A=(A10+A11), B=B00
    # M3: A=A00, B=(B01-B11)
    # M4: A=A11, B=(B10-B00)
    # M5: A=(A00+A01), B=B11
    # M6: A=(A10-A00), B=(B00+B01)
    # M7: A=(A01-A11), B=(B10+B11)
    
    # For each Mk, we perform a 2x2 Strassen multiplication.
    # The result Mk is a 2x2 matrix.
    # We need to know how Mk contributes to C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each scalar multiplication inside Mk, we need to add its contribution to the corresponding C elements.
    
    # Let's write a helper to process one 2x2 Strassen multiplication given:
    # - A linear combo coeffs for the 4 elements of the 2x2 A block
    # - B linear combo coeffs for the 4 elements of the 2x2 B block
    # - A dict mapping C_block_element (0,0), (0,1), (1,0), (1,1) to coefficients
    #   (This dict represents how the resulting 2x2 matrix Mk is added to C)
    
    def process_2x2_strassen(a_coeffs, b_coeffs, c_contribs, start_col):
        # a_coeffs: {(0,0): c, (0,1): c, (1,0): c, (1,1): c} for the 2x2 block
        # b_coeffs: {(0,0): c, (0,1): c, (1,0): c, (1,1): c}
        # c_contribs: {(0,0): c, (0,1): c, (1,0): c, (1,1): c}
        
        # Strassen 2x2 steps:
        # N1 = (A00+A11)(B00+B11)
        # N2 = (A10+A11)B00
        # N3 = A00(B01-B11)
        # N4 = A11(B10-B00)
        # N5 = (A00+A01)B11
        # N6 = (A10-A00)(B00+B01)
        # N7 = (A01-A11)(B10+B11)
        
        # Results:
        # R00 = N1 + N4 - N5 + N7
        # R01 = N3 + N5
        # R10 = N2 + N4
        # R11 = N1 - N2 + N3 + N6
        
        # We need to generate 7 columns.
        # For each Ni, we define u_i, v_i, w_i.
        # w_i will be the combination of c_contribs based on R formulas.
        
        steps = [
            # (u_coeffs, v_coeffs, r_contribs)
            # r_contribs: {(0,0): c, (0,1): c, (1,0): c, (1,1): c} for the result of this step
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # N1 -> R00, R11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # N2 -> R10, R11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # N3 -> R01, R11
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # N4 -> R00, R10
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # N5 -> R00, R01
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # N6 -> R11
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}) # N7 -> R00, R11
        ]
        
        for i, (u_step, v_step, r_step) in enumerate(steps):
            col_idx = start_col + i
            
            # Expand u_step to global A indices
            # a_coeffs maps local 2x2 indices to global coeffs
            # u_step maps local 2x2 indices to step coeffs
            # We need to combine them.
            # This is a convolution of sorts.
            # Actually, the "A" in the 2x2 Strassen is already a linear combo of global A elements.
            # So if A_local[0,0] = 2*A_global[0,0] + 3*A_global[1,1], then u_step[0,0]=1 means we use that whole combo.
            
            # So, for each local index (lr, lc) in u_step:
            #   coeff_u = u_step[(lr, lc)]
            #   global_coeffs = a_coeffs[(lr, lc)]
            #   For each global (gi, gj) in global_coeffs:
            #       U[gi, gj] += coeff_u * global_coeffs[(gi, gj)]
            
            u_global = {}
            for (lr, lc), cu in u_step.items():
                if cu == 0: continue
                # a_coeffs must contain (lr, lc)
                if (lr, lc) in a_coeffs:
                    for (gi, gj), cg in a_coeffs[(lr, lc)].items():
                        if cg == 0: continue
                        u_global[(gi, gj)] = u_global.get((gi, gj), 0) + cu * cg
                else:
                    # If not present, it's 0
                    pass
            
            v_global = {}
            for (lr, lc), cv in v_step.items():
                if cv == 0: continue
                if (lr, lc) in b_coeffs:
                    for (gj, gk), cg in b_coeffs[(lr, lc)].items():
                        if cg == 0: continue
                        v_global[(gj, gk)] = v_global.get((gj, gk), 0) + cv * cg
            
            w_global = {}
            for (lr, lc), cw in r_step.items():
                if cw == 0: continue
                if (lr, lc) in c_contribs:
                    # c_contribs maps local C indices to global coeffs
                    # But wait, c_contribs is for the BLOCK result.
                    # The block result Mk is 2x2.
                    # c_contribs[(lr, lc)] is a scalar coefficient?
                    # No, c_contribs should map local C indices to global C indices with coefficients.
                    # In our case, the block C is a 2x2 submatrix of the global 4x4 C.
                    # So local (0,0) -> global (I*2, K*2).
                    # I will pass c_contribs as a dict { (lr, lc): { (gi, gk): coeff } }
                    
                    for (gi, gk), cg in c_contribs[(lr, lc)].items():
                        if cg == 0: continue
                        w_global[(gi, gk)] = w_global.get((gi, gk), 0) + cw * cg
            
            # Fill U, V, W
            for (gi, gj), c in u_global.items():
                if c != 0:
                    U[gi * m + gj][col_idx] += c
            for (gj, gk), c in v_global.items():
                if c != 0:
                    V[gj * p + gk][col_idx] += c
            for (gi, gk), c in w_global.items():
                if c != 0:
                    W[gk * n + gi][col_idx] += c

    # Define the 7 block multiplications
    # a_block_coeffs: { (0,0): { (gi, gj): c }, ... }
    # We need to build these for each of the 7 M_k.
    
    # Helper to get A block coeffs
    def get_a_block(I, J):
        # Returns dict { (lr, lc): { (gi, gj): 1 } }
        res = {}
        for lr in range(2):
            for lc in range(2):
                gi = I * 2 + lr
                gj = J * 2 + lc
                res[(lr, lc)] = { (gi, gj): 1 }
        return res

    def get_b_block(J, K):
        res = {}
        for lr in range(2):
            for lc in range(2):
                gj = J * 2 + lr
                gk = K * 2 + lc
                res[(lr, lc)] = { (gj, gk): 1 }
        return res

    def get_c_block(I, K):
        # Returns dict { (lr, lc): { (gi, gk): 1 } }
        res = {}
        for lr in range(2):
            for lc in range(2):
                gi = I * 2 + lr
                gk = K * 2 + lc
                res[(lr, lc)] = { (gi, gk): 1 }
        return res

    # Combine blocks
    def combine_a(blocks):
        # blocks: list of (I, J, coeff)
        res = { (0,0): {}, (0,1): {}, (1,0): {}, (1,1): {} }
        for I, J, coeff in blocks:
            b = get_a_block(I, J)
            for lr in range(2):
                for lc in range(2):
                    for (gi, gj), c in b[(lr, lc)].items():
                        res[(lr, lc)][(gi, gj)] = res[(lr, lc)].get((gi, gj), 0) + coeff * c
        return res

    def combine_b(blocks):
        # blocks: list of (J, K, coeff)
        res = { (0,0): {}, (0,1): {}, (1,0): {}, (1,1): {} }
        for J, K, coeff in blocks:
            b = get_b_block(J, K)
            for lr in range(2):
                for lc in range(2):
                    for (gj, gk), c in b[(lr, lc)].items():
                        res[(lr, lc)][(gj, gk)] = res[(lr, lc)].get((gj, gk), 0) + coeff * c
        return res

    def combine_c(blocks):
        # blocks: list of (I, K, coeff)
        res = { (0,0): {}, (0,1): {}, (1,0): {}, (1,1): {} }
        for I, K, coeff in blocks:
            b = get_c_block(I, K)
            for lr in range(2):
                for lc in range(2):
                    for (gi, gk), c in b[(lr, lc)].items():
                        res[(lr, lc)][(gi, gk)] = res[(lr, lc)].get((gi, gk), 0) + coeff * c
        return res

    # M1: A=(A00+A11), B=(B00+B11)
    # C contribs: C00+=1, C11+=1
    a1 = combine_a([(0,0,1), (1,1,1)])
    b1 = combine_b([(0,0,1), (1,1,1)])
    c1 = combine_c([(0,0,1), (1,1,1)])
    process_2x2_strassen(a1, b1, c1, col)
    col += 7
    
    # M2: A=(A10+A11), B=B00
    # C contribs: C10+=1, C11-=1
    a2 = combine_a([(1,0,1), (1,1,1)])
    b2 = combine_b([(0,0,1)])
    c2 = combine_c([(1,0,1), (1,1,-1)])
    process_2x2_strassen(a2, b2, c2, col)
    col += 7
    
    # M3: A=A00, B=(B01-B11)
    # C contribs: C01+=1, C11+=1
    a3 = combine_a([(0,0,1)])
    b3 = combine_b([(0,1,1), (1,1,-1)])
    c3 = combine_c([(0,1,1), (1,1,1)])
    process_2x2_strassen(a3, b3, c3, col)
    col += 7
    
    # M4: A=A11, B=(B10-B00)
    # C contribs: C00+=1, C10+=1
    a4 = combine_a([(1,1,1)])
    b4 = combine_b([(1,0,1), (0,0,-1)])
    c4 = combine_c([(0,0,1), (1,0,1)])
    process_2x2_strassen(a4, b4, c4, col)
    col += 7
    
    # M5: A=(A00+A01), B=B11
    # C contribs: C00-=1, C01+=1
    a5 = combine_a([(0,0,1), (0,1,1)])
    b5 = combine_b([(1,1,1)])
    c5 = combine_c([(0,0,-1), (0,1,1)])
    process_2x2_strassen(a5, b5, c5, col)
    col += 7
    
    # M6: A=(A10-A00), B=(B00+B01)
    # C contribs: C11+=1
    a6 = combine_a([(1,0,1), (0,0,-1)])
    b6 = combine_b([(0,0,1), (0,1,1)])
    c6 = combine_c([(1,1,1)])
    process_2x2_strassen(a6, b6, c6, col)
    col += 7
    
    # M7: A=(A01-A11), B=(B10+B11)
    # C contribs: C00+=1, C11+=1
    a7 = combine_a([(0,1,1), (1,1,-1)])
    b7 = combine_b([(1,0,1), (1,1,1)])
    c7 = combine_c([(0,0,1), (1,1,1)])
    process_2x2_strassen(a7, b7, c7, col)
    col += 7
    
    # Part 2: 4x4x3 Schoolbook
    # A is 4x4. B' is 4x3 (cols 4,5,6 of B). C' is 4x3 (cols 4,5,6 of C).
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k_global = 4 + k_local
                U[i * m + j][col] = 1
                V[j * p + k_global][col] = 1
                W[k_global * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
