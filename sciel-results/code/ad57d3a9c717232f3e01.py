import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # We will build U, V, W as lists of lists or numpy arrays.
    # Final rank will be determined by the number of terms.
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x5 using Schoolbook (Rank 80)
    
    # Initialize lists for U, V, W
    # U: (16, rank), V: (16, rank), W: (16, rank)
    # We don't know rank yet, so we'll collect terms in lists.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    # --- Strassen for 4x4x4 ---
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    # We define the 7 Strassen steps for the top level.
    # Each step involves linear combinations of the 4x4 blocks.
    # Blocks are 2x2.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00, B01, B10, B11
    
    # Strassen coefficients for 2x2 block multiplication
    # We need to map the 2x2 base case to the 4x4 indices.
    
    # Base case 2x2 rank 7 terms
    # Let's define the 7 terms for 2x2 multiplication X*Y -> Z
    # X, Y, Z are 2x2.
    # Term 1: (x00+x11)(y00+y11)
    # Term 2: (x10+x11)y00
    # Term 3: x00(y10-y11)
    # Term 4: x11(y01-y00)
    # Term 5: (x00+x10)y11
    # Term 6: (x01-x00)(y00+y10)
    # Term 7: (x10-x11)(y01+y11)
    
    # We will iterate through the 7 top-level Strassen products.
    # For each product, we have A' and B' which are 2x2 matrices of 2x2 blocks.
    # Wait, A' is a 2x2 matrix where each entry is a 2x2 block of A.
    # So A' is effectively a 4x4 matrix, but structured as 2x2 blocks.
    # The multiplication A' * B' is a 2x2 matrix multiplication of these blocks.
    # We decompose this 2x2 block multiplication into 7 scalar (block) multiplications.
    # Each block multiplication is a 2x2 matrix multiplication.
    # We decompose each 2x2 matrix multiplication into 7 scalar multiplications.
    
    # So, for each of the 7 top-level products:
    #   We identify the linear combinations of A blocks and B blocks.
    #   For each of the 7 base-case Strassen terms:
    #     We identify the linear combination of entries of A' and entries of B'.
    #     This gives a vector of coefficients for U (entries of A) and V (entries of B).
    #     And the output W is determined by the position in the result block.
    
    # Let's define the top-level Strassen combinations.
    # A' and B' are 2x2 matrices of blocks.
    # A' = [[A00+A11, A01], [A10+A11, A00]] ? No, standard Strassen:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A10)B11
    # M6: (A01-A00)(B00+B10)
    # M7: (A10-A11)(B01+B11)
    
    # Let's represent a linear combination of blocks as a dict {(block_row, block_col): coeff}
    
    top_level_steps = [
        { # M1
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1}
        },
        { # M2
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1}
        },
        { # M3
            'A': {(0,0): 1},
            'B': {(1,0): 1, (1,1): -1}
        },
        { # M4
            'A': {(1,1): 1},
            'B': {(0,1): 1, (0,0): -1}
        },
        { # M5
            'A': {(0,0): 1, (1,0): 1},
            'B': {(1,1): 1}
        },
        { # M6
            'A': {(0,1): 1, (0,0): -1},
            'B': {(0,0): 1, (1,0): 1}
        },
        { # M7
            'A': {(1,0): 1, (1,1): -1},
            'B': {(0,1): 1, (1,1): 1}
        }
    ]
    
    # Base case 2x2 Strassen terms
    # Each term is defined by:
    # - Linear combo of X entries (2x2)
    # - Linear combo of Y entries (2x2)
    # - Output position in Z (2x2) and coefficient
    # Actually, the base case produces the full 2x2 result Z.
    # Z00 = M1 - M4 + M6 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M5 + M7
    
    # So for each top-level step, we get a 2x2 matrix of block products.
    # Let's call the block product for top-level step t and base-case term b as P_{t,b}.
    # P_{t,b} is a 2x2 matrix (the result of multiplying the linear combos).
    # This 2x2 matrix contributes to the final 4x4 result C.
    # The contribution is added to specific blocks of C.
    
    # We need to map the base-case Strassen terms to the U, V, W vectors.
    
    # Let's iterate.
    
    for step in top_level_steps:
        A_comb = step['A']
        B_comb = step['B']
        
        # For each of the 7 base-case Strassen terms
        # We need to define the 7 terms for 2x2 multiplication.
        # Let's define them explicitly.
        
        base_terms = [
            { # Term 1
                'X': {(0,0): 1, (1,1): 1},
                'Y': {(0,0): 1, (1,1): 1},
                'Z': {(0,0): 1, (1,1): 1} # This is not how it works.
            }
        ]
        
        # Actually, it's easier to just say:
        # The product of the linear combos is a 2x2 matrix.
        # Let X' be the 2x2 matrix of linear combos of A blocks.
        # Let Y' be the 2x2 matrix of linear combos of B blocks.
        # We compute Z' = X' Y'.
        # Z' is a 2x2 matrix of 2x2 blocks.
        # Each block of Z' is a 2x2 matrix (result of 2x2 mult).
        # We decompose each block of Z' using the base case.
        
        # Wait, the base case decomposes the *entire* 2x2 multiplication.
        # So for the top-level step, we have a 2x2 block multiplication.
        # We apply the base case Strassen algorithm to this 2x2 block multiplication.
        # The base case uses 7 products of *entries* of the blocks.
        # But the entries of the blocks are 2x2 matrices.
        # So each "entry product" is a 2x2 matrix multiplication.
        # We must decompose each of these 2x2 matrix multiplications using the base case again?
        # No. The base case is for 2x2 *scalar* matrices.
        # Here, the "scalars" are 2x2 matrices.
        # So we need to decompose the 2x2 matrix multiplication of the "scalars".
        # This is recursive.
        
        # Level 1: 4x4x4. Split into 2x2 blocks.
        # 7 products of 2x2 blocks.
        # Level 2: Each 2x2 block product is a 2x2 matrix multiplication.
        # Decompose each into 7 products of 2x2 *entries* (which are 1x1 scalars? No, 2x2 entries are 1x1 if we split 2x2 into 1x1? No, 2x2 entries are scalars).
        # Wait. 2x2 matrix entries are scalars.
        # So the base case 2x2 Strassen algorithm applies directly to the 2x2 blocks.
        # The 2x2 blocks are matrices of scalars.
        # So for each of the 7 top-level products, we perform a 2x2 matrix multiplication.
        # This 2x2 matrix multiplication is done using the rank-7 Strassen algorithm.
        # This adds 7 rank terms per top-level product.
        # Total 49 terms.
        
        # So, for each top-level step:
        #   We have X' (2x2 matrix of A-blocks) and Y' (2x2 matrix of B-blocks).
        #   We apply the 7 Strassen formulas for 2x2 matrices.
        #   Each formula involves products of entries of X' and Y'.
        #   An entry of X' is a linear combination of 2x2 blocks of A.
        #   An entry of Y' is a linear combination of 2x2 blocks of B.
        #   The product of these two entries is a product of two 2x2 matrices.
        #   We decompose this 2x2 matrix product into 7 scalar products.
        
        # So we need to iterate:
        # for top_step in 7:
        #   for base_step in 7:
        #     Identify the entry of X' and entry of Y' involved.
        #     These are 2x2 matrices (linear combos of A/B blocks).
        #     Decompose their product using the 7 scalar Strassen terms.
        #     For each scalar term, add to U, V, W.
        
        # This is getting complex to code generically.
        # Let's hardcode the 49 terms? Or write a helper.
        
        pass # Placeholder

    # Part 2: Schoolbook for 4x4x5
    # i in 0..3, j in 0..3, k in 4..8
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                u_vec = [0]*16
                v_vec = [0]*16
                w_vec = [0]*16
                u_vec[i*4+j] = 1
                v_vec[j*4+k] = 1
                w_vec[k*4+i] = 1
                add_term(u_vec, v_vec, w_vec)
                
    # Now we need to fill the Strassen terms.
    # Let's write a function to add a Strassen term for 4x4x4.
    
    return {"U": U_terms, "V": V_terms, "W": W_terms}
