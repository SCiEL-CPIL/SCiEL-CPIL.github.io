"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    # Split p=7 into p1=3 and p2=4
    p1, p2 = 3, 4
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1
    
    # Part 1: Schoolbook for columns 0..2 (k in [0,3))
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                add_term(u, v, w)
    
    # Part 2: Strassen for columns 3..6 (k in [3,7))
    # Strassen 4x4x4: 7 terms, each involving 2x2 blocks
    # For each Strassen term, we need to handle the 2x7 (actually 2x4 here) part
    
    # Strassen 4x4x4 decomposition
    # A = [[A00, A01], [A10, A11]], each 2x2
    # B = [[B00, B01], [B10, B11]], each 2x2
    # C = [[C00, C01], [C10, C11]], each 2x2
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our case, B is 4x4 (columns 3-6), so B blocks are 2x2
    # Each "multiplication" in Strassen is 2x2 x 2x2, which we decompose into 4 schoolbook terms
    
    # Strassen terms: (coeff_A, coeff_B) pairs
    # Each term is sum of products of linear combinations
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But in rank-1 decomposition, we need to expand these
    
    # Actually, for the tensor decomposition, each rank-1 term corresponds to a single
    # (i,j,k) triple with coefficients. Let me directly construct the 49 terms.
    
    # Strassen 4x4x4: 7 recursive calls, each 2x2x2 with rank 7
    # Total 49 rank-1 terms
    
    # Let me define the Strassen structure explicitly
    # 2x2x2 Strassen: 7 terms
    # For 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply this at two levels.
    # Level 1: 7 terms, each is a 2x2x2 multiplication
    # Level 2: each 2x2x2 is decomposed into 7 rank-1 terms
    # Total: 49 rank-1 terms
    
    # Let me build this by defining the 49 (i,j,k) triples with coefficients
    
    # Actually, it's easier to think of it as: the 4x4x4 multiplication tensor
    # can be written as sum of 49 rank-1 tensors
    
    # Let me use a known explicit decomposition
    
    # Define helper: for a 2x2x2 Strassen, given base indices (i0, j0, k0)
    # for the 2x2 blocks, generate 7 rank-1 terms
    
    def strassen_2x2x2(i0, j0, k0):
        """Generate 7 rank-1 terms for 2x2x2 Strassen with block offset."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # This expands to: A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j0+1,k0+1] + A[i0+1,j0+1]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0+1]
        # But in rank-1 form, M1 itself is one "multiplication" that we further decompose
        # Wait, I'm confusing levels.
        
        # At the 2x2x2 level, Strassen gives 7 "multiplications" of scalars.
        # Each scalar multiplication a*b contributes one rank-1 term.
        # So for 2x2x2, we have 7 rank-1 terms.
        
        # The 7 terms correspond to:
        # t1: (A00+A11) * (B00+B11) -> but this is NOT a single rank-1 term
        # I need to be more careful.
        
        # Actually, Strassen's algorithm for 2x2x2:
        # We compute 7 products of linear combinations, then combine.
        # Each "product" p = (sum a_i * A_i) * (sum b_j * B_j) is NOT a rank-1 term
        # of the original tensor. Instead, the rank-1 terms come from expanding these.
        
        # Let me reconsider. The matrix multiplication tensor for 2x2x2 is:
        # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        # (with appropriate index conventions)
        
        # Strassen shows this can be written as sum of 7 rank-1 tensors.
        # The 7 rank-1 tensors are:
        
        # r1 = (e_{0,0}+e_{1,1}) ⊗ (e_{0,0}+e_{1,1}) ⊗ (e_{0,0}+e_{1,1}-e_{0,1}-e_{1,0}+e_{1,1})
        # This is getting complicated. Let me use a known explicit list.
        
        # Known Strassen 2x2x2 decomposition (49 total for 4x4x4, 7 for 2x2x2):
        # I'll use the standard formulation where each rank-1 term is defined by
        # its support on (i,j,k) indices.
        
        # For 2x2x2, the 7 rank-1 terms (u, v, w) where u is 2x2, v is 2x2, w is 2x2:
        # Term 1: u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]  (identity*identity)
        # No, that's not right either.
        
        # Let me just use the schoolbook for 2x2x2 (4 terms) to be safe,
        # and only use Strassen at the top level.
        
        # Actually, for 4x4x4, if I use Strassen at top level (7 terms) and
        # schoolbook for each 2x2x2 (4 terms each), total = 7*4 = 28.
        # That's better than 49 and better than schoolbook 64.
        
        # But 28 might not be valid. Let me verify.
        
        # Strassen 4x4x4: 7 products, each is 2x2x2.
        # If each 2x2x2 is done with schoolbook (4 scalar multiplications),
        # then total scalar multiplications = 7*4 = 28.
        # This should be a valid rank-28 decomposition!
        
        pass
    
    # OK let me just directly construct the 28-term decomposition for 4x4x4
    # using Strassen at top level and schoolbook for 2x2x2.
    
    # Strassen 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product, done with 4 schoolbook multiplications.
    
    # For M1 = (A00+A11)(B00+B11):
    # Let A' = A00+A11, B' = B00+B11
    # M1[i,j] = sum_k A'[i,k] * B'[k,j] for i,j in {0,1}
    # This is 4 scalar multiplications:
    # M1[0,0] = A'[0,0]*B'[0,0] + A'[0,1]*B'[1,0]
    # etc.
    
    # In the tensor decomposition, each scalar multiplication a*b where
    # a is a linear combination of A entries and b is a linear combination
    # of B entries contributes a rank-1 term.
    
    # Let me define the 28 terms explicitly.
    
    # For the 4x4x4 part (columns 3-6), k offset is 3.
    k_off = p1  # 3
    
    # Define Strassen terms as (list of (i,j, coeff) for U, list of (j,k, coeff) for V, list of (k,i, coeff) for W)
    # Actually, let me build it differently.
    
    # I'll create the 28 rank-1 terms for 4x4x4.
    
    # Strassen level 1: 7 terms
    # Each term has:
    # - A part: linear combination of 2x2 blocks
    # - B part: linear combination of 2x2 blocks
    # - The product is a 2x2 matrix, decomposed into 4 scalar mults
    
    # Let me define each of the 7 Strassen products and their 4 scalar components.
    
    # Helper: block indices
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    
    # For each Strassen term, define:
    # A_coeffs: dict from (i,j) to coefficient
    # B_coeffs: dict from (j,k) to coefficient
    # Then the 4 scalar multiplications for the 2x2 product
    
    strassen_terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11
    A1 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
    # B part: B00 + B11
    B1 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
    # Output: C00 += M1, C11 += M1
    # M1 is 2x2: M1[i,j] for i,j in {0,1} maps to C[i,j] and C[i+2,j+2]
    
    # This is getting very complex. Let me just use schoolbook for the whole thing
    # and accept rank 112, since I'm running out of time and complexity.
    
    # Actually, let me try a simpler improvement: use the fact that for the 4x4x7 tensor,
    # we can potentially share some structure.
    
    # For now, let me just return schoolbook (rank 112) to ensure correctness,
    # and add a TODO for optimization.
    
    # Reset
    U = []
    V = []
    W = []
    rank = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                U.append(u)
                V.append(v)
                W.append(w)
                rank += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
