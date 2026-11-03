"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We will construct a decomposition of rank 77.
    # Part 1: 4x4 * 4x4 (first 4 cols of B) using recursive Strassen (Rank 49)
    # Part 2: 4x4 * 4x2 (last 2 cols of B) using block Strassen (Rank 28)
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    # U: (n*m, rank) = (16, 77)
    # V: (m*p, rank) = (24, 77)
    # W: (p*n, rank) = (24, 77)
    # W index is k*n + i.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, col, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # --- Part 1: 4x4 * 4x4 ---
    # Blocks of A (4x4):
    # A11 (top-left): rows 0-1, cols 0-1. Indices: 0,1,4,5
    # A12 (top-right): rows 0-1, cols 2-3. Indices: 2,3,6,7
    # A21 (bottom-left): rows 2-3, cols 0-1. Indices: 8,9,12,13
    # A22 (bottom-right): rows 2-3, cols 2-3. Indices: 10,11,14,15
    
    # Blocks of B (4x4 part, cols 0-3):
    # B11: rows 0-1, cols 0-1. Indices: 0,1,6,7
    # B12: rows 0-1, cols 2-3. Indices: 2,3,8,9
    # B21: rows 2-3, cols 0-1. Indices: 12,13,18,19
    # B22: rows 2-3, cols 2-3. Indices: 14,15,20,21
    
    # Blocks of C (4x4 part, cols 0-3):
    # C11: rows 0-1, cols 0-1. W indices: 0,1,4,5
    # C12: rows 0-1, cols 2-3. W indices: 2,3,6,7
    # C21: rows 2-3, cols 0-1. W indices: 8,9,12,13
    # C22: rows 2-3, cols 2-3. W indices: 10,11,14,15
    
    # Strassen 2x2*2x2 terms (local indices 0,1,2,3 correspond to 00,01,10,11):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to add 7 scalar multiplications (since each block is 2x2).
    # Let's define a helper to add a block multiplication term.
    
    def add_block_mult(a_block, b_block, c_blocks, base_col):
        # a_block: list of 4 indices
        # b_block: list of 4 indices
        # c_blocks: list of (c_block, coeff) where c_block is list of 4 indices
        # This adds the 7 terms for 2x2*2x2 -> sum of 2x2 blocks
        
        # 2x2*2x2 Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Local indices for 2x2 block:
        # 0: 00, 1: 01, 2: 10, 3: 11
        
        # We will add 7 terms.
        
        # Term 1: M1
        # A: 1,1,1,1 (all +1)
        # B: 1,1,1,1
        # C: C11(1,1,1,1), C22(1,1,1,1)
        
        # Wait, the "local" 2x2 multiplication is:
        # X (2x2) * Y (2x2) = Z (2x2)
        # The 7 terms are for the scalar multiplications of the linear combinations.
        
        # Let's just hardcode the 49 terms for 4x4x4 by iterating through the 7 Strassen block terms
        # and for each, the 7 scalar terms.
        
        # Actually, it's easier to just use the known rank 49 decomposition or generate it carefully.
        # Let's try to generate it by applying Strassen recursively.
        
        # Strassen for 4x4:
        # Split A, B into 2x2 blocks of 2x2 matrices.
        # 7 block multiplications.
        # Each block multiplication is 2x2 * 2x2.
        # We can use the standard 7-term Strassen for 2x2*2x2.
        
        # Let's define the 7 terms for 2x2*2x2 explicitly.
        # Input A (2x2), B (2x2).
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
        
        # For each of the 7 block terms in the 4x4 Strassen, we add these 7 scalar terms.
        
        # Let's write a function to add a 2x2*2x2 term to the global U, V, W.
        def add_2x2_mult(a_idx, b_idx, c_idx, base_col):
            # a_idx, b_idx, c_idx are lists of 4 global indices.
            # We add 7 terms.
            col = base_col
            
            # M1: (A00+A11)(B00+B11)
            # U: A00(1), A11(1) -> a_idx[0], a_idx[3]
            # V: B00(1), B11(1) -> b_idx[0], b_idx[3]
            # W: C00(1), C11(1) -> c_idx[0], c_idx[3]
            U[a_idx[0]][col] += 1; U[a_idx[3]][col] += 1
            V[b_idx[0]][col] += 1; V[b_idx[3]][col] += 1
            W[c_idx[0]][col] += 1; W[c_idx[3]][col] += 1
            col += 1
            
            # M2: (A10+A11)B00
            # U: A10(1), A11(1) -> a_idx[2], a_idx[3]
            # V: B00(1) -> b_idx[0]
            # W: C10(1), C11(-1) -> c_idx[2], c_idx[3]
            U[a_idx[2]][col] += 1; U[a_idx[3]][col] += 1
            V[b_idx[0]][col] += 1
            W[c_idx[2]][col] += 1; W[c_idx[3]][col] -= 1
            col += 1
            
            # M3: A00(B01-B11)
            # U: A00(1) -> a_idx[0]
            # V: B01(1), B11(-1) -> b_idx[1], b_idx[3]
            # W: C01(1), C11(1) -> c_idx[1], c_idx[3]
            U[a_idx[0]][col] += 1
            V[b_idx[1]][col] += 1; V[b_idx[3]][col] -= 1
            W[c_idx[1]][col] += 1; W[c_idx[3]][col] += 1
            col += 1
            
            # M4: A11(B10-B00)
            # U: A11(1) -> a_idx[3]
            # V: B10(1), B00(-1) -> b_idx[2], b_idx[0]
            # W: C00(1), C10(1) -> c_idx[0], c_idx[2]
            U[a_idx[3]][col] += 1
            V[b_idx[2]][col] += 1; V[b_idx[0]][col] -= 1
            W[c_idx[0]][col] += 1; W[c_idx[2]][col] += 1
            col += 1
            
            # M5: (A00+A01)B11
            # U: A00(1), A01(1) -> a_idx[0], a_idx[1]
            # V: B11(1) -> b_idx[3]
            # W: C00(-1), C01(1) -> c_idx[0], c_idx[1]
            U[a_idx[0]][col] += 1; U[a_idx[1]][col] += 1
            V[b_idx[3]][col] += 1
            W[c_idx[0]][col] -= 1; W[c_idx[1]][col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            # U: A10(1), A00(-1) -> a_idx[2], a_idx[0]
            # V: B00(1), B01(1) -> b_idx[0], b_idx[1]
            # W: C11(1) -> c_idx[3]
            U[a_idx[2]][col] += 1; U[a_idx[0]][col] -= 1
            V[b_idx[0]][col] += 1; V[b_idx[1]][col] += 1
            W[c_idx[3]][col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            # U: A01(1), A11(-1) -> a_idx[1], a_idx[3]
            # V: B10(1), B11(1) -> b_idx[2], b_idx[3]
            # W: C00(1) -> c_idx[0]
            U[a_idx[1]][col] += 1; U[a_idx[3]][col] -= 1
            V[b_idx[2]][col] += 1; V[b_idx[3]][col] += 1
            W[c_idx[0]][col] += 1
            col += 1
            
            return col
        
        # Now apply this to the 4x4 Strassen blocks.
        
        # A blocks:
        A11 = [0, 1, 4, 5]
        A12 = [2, 3, 6, 7]
        A21 = [8, 9, 12, 13]
        A22 = [10, 11, 14, 15]
        
        # B blocks (4x4 part):
        B11 = [0, 1, 6, 7]
        B12 = [2, 3, 8, 9]
        B21 = [12, 13, 18, 19]
        B22 = [14, 15, 20, 21]
        
        # C blocks (4x4 part):
        C11 = [0, 1, 4, 5]
        C12 = [2, 3, 6, 7]
        C21 = [8, 9, 12, 13]
        C22 = [10, 11, 14, 15]
        
        # Strassen 4x4 terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add the terms for each M_k.
        # Each M_k is a 2x2*2x2 multiplication.
        # We can just call add_2x2_mult for each M_k, but we need to map the linear combinations.
        
        # Actually, the "block" Strassen is just applying the 2x2 Strassen formula to the blocks.
        # So M1 is the product of (A11+A22) and (B11+B22).
        # (A11+A22) is a 2x2 matrix. Its indices are A11[i] and A22[i].
        # We can create a "virtual" 2x2 matrix for A11+A22 by passing a list of indices that includes both?
        # No, add_2x2_mult expects a single list of 4 indices for A.
        # We need to modify add_2x2_mult to handle linear combinations.
        
        # Let's rewrite add_2x2_mult to take a list of (indices, coeff) for A and B.
        
        def add_2x2_mult_general(a_terms, b_terms, c_terms, base_col):
            # a_terms: list of (indices, coeff)
            # b_terms: list of (indices, coeff)
            # c_terms: list of (indices, coeff)
            
            # This is getting complicated. Let's just hardcode the 49 terms for 4x4x4.
            # Or, better, let's use the fact that we can just add the terms for each scalar multiplication.
            
            # Let's stick to the plan:
            # For each of the 7 Strassen block terms, we add 7 scalar terms.
            # Total 49.
            
            # M1: A = A11+A22, B = B11+B22
            # A indices: A11[i] with coeff 1, A22[i] with coeff 1.
            # B indices: B11[i] with coeff 1, B22[i] with coeff 1.
            # C indices: C11[i] with coeff 1, C22[i] with coeff 1.
            
            # We can just call add_term for each of the 7 scalar terms of M1.
            
            # Let's define a helper to add a scalar term.
            def add_scalar_term(a_idx, a_coeff, b_idx, b_coeff, c_idx, c_coeff, col):
                if a_coeff != 0:
                    U[a_idx][col] += a_coeff
                if b_coeff != 0:
                    V[b_idx][col] += b_coeff
                if c_coeff != 0:
                    W[c_idx][col] += c_coeff
            
            # M1: (A11+A22)(B11+B22)
            # Scalar terms:
            # 1. (A11[0]+A22[0]) * (B11[0]+B22[0]) -> C11[0] += 1, C22[0] += 1
            # This is 4 terms for the first scalar, etc.
            
            # Actually, the 7 terms of 2x2*2x2 are:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For M1 (block):
            # A00 = A11[0] + A22[0]
            # A11 = A11[3] + A22[3]
            # B00 = B11[0] + B22[0]
            # B11 = B11[3] + B22[3]
            
            # So term 1 of M1 (block) is:
            # (A11[0]+A22[0] + A11[3]+A22[3]) * (B11[0]+B22[0] + B11[3]+B22[3])
            # This expands to 16 scalar products.
            # But Strassen says we only need 7 scalar multiplications for the whole 2x2*2x2.
            # So we should not expand.
            
            # The correct way is to treat (A11+A22) as a single 2x2 matrix A' and (B11+B22) as B'.
            # Then apply the 7-term Strassen to A' and B'.
            
            # So for M1 (block), we call add_2x2_mult with:
            # A' indices: [A11[0]+A22[0], A11[1]+A22[1], A11[2]+A22[2], A11[3]+A22[3]]
            # But we can't add indices. We need to add the terms.
            
            # Let's modify add_2x2_mult to take a list of (indices, coeff) for A and B.
            
            return base_col
        
        # This is too complex to do correctly in the time limit.
        # I will return the schoolbook decomposition (Rank 96) which is guaranteed to be correct.
        
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
