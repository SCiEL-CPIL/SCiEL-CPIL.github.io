"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # 1. Initialize with Schoolbook Decomposition (Rank = n*m*p)
    # This is a valid solution. We will try to reduce the rank.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_factors = schoolbook()
    current_rank = n * m * p

    # Helper to check if a set of factors is valid
    # We use Fractions for exactness to avoid floating point issues with half-integers
    def is_valid(U, V, W):
        # U, V, W are lists of lists of Fractions
        # Check if all entries are half-integers (denominator 1 or 2)
        for row in U:
            for val in row:
                if val.denominator not in (1, 2):
                    return False
        for row in V:
            for val in row:
                if val.denominator not in (1, 2):
                    return False
        for row in W:
            for val in row:
                if val.denominator not in (1, 2):
                    return False
        
        # Check tensor decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Helper to convert to output format (floats are fine if exact half-integers, but ints are safer)
    def to_output(factors):
        U, V, W = factors['U'], factors['V'], factors['W']
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}

    # Local Search Strategy
    # State: U, V, W as lists of lists of Fractions
    # We start with the schoolbook solution.
    
    U_curr = [[Fraction(x) for x in row] for row in best_factors['U']]
    V_curr = [[Fraction(x) for x in row] for row in best_factors['V']]
    W_curr = [[Fraction(x) for x in row] for row in best_factors['W']]
    
    rank = len(U_curr[0])
    
    # We want to reduce rank.
    # Idea: Try to find two columns r1, r2 that can be merged or removed?
    # Actually, standard local search for tensor rank is hard.
    # Let's try a different approach: 
    # Randomly perturb the factors and use a solver to fix W, then round?
    # No, that was the old strategy.
    
    # Let's try "Column Merging" heuristic.
    # If we have rank R, can we find a set of R-1 factors?
    # This is generally NP-hard.
    
    # Alternative: Use the "Strassen-like" structure.
    # For 2x4 * 4x6, we can view it as two 2x4 * 4x3 blocks? No.
    # We can view it as 2x2 blocks?
    # 2x4 * 4x6.
    # Split M into two 2x2 blocks.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x6.
    # C = A1*B1 + A2*B2.
    # This doesn't reduce rank directly unless we use fast matrix mult for 2x2.
    # Rank of 2x2 matrix mult is 7 (Strassen).
    # So we can do:
    # C = A1*B1 + A2*B2
    # Each product is 2x2 * 2x6? No, B1 is 2x6.
    # So we are multiplying 2x2 by 2x6.
    # This can be done by 3 independent 2x2 * 2x2 multiplications?
    # B1 = [B11, B12, B13] (each 2x2).
    # A1 * B1 = A1*B11 + A1*B12 + A1*B13.
    # Each A1*B1j is 2x2 * 2x2.
    # We can use Strassen for each?
    # Total operations: 3 (for B1) + 3 (for B2) = 6 products of 2x2 matrices.
    # Each 2x2 product costs rank 7.
    # Total rank = 6 * 7 = 42.
    # This is better than 48!
    
    # Let's construct this decomposition explicitly.
    
    def strassen_2x2(A, B):
        # A, B are 2x2 matrices of Fractions
        # Returns 7 terms (u, v, w) such that C = sum u_i * v_i * w_i ?
        # No, Strassen gives 7 scalar multiplications.
        # The tensor decomposition for 2x2*2x2 has rank 7.
        # We need to map this to our global U, V, W.
        
        # Strassen formulas:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # We need to express these as sums of products U[i,j,r] V[j,k,r] W[k,i,r]
        # In the context of the full 2x4 * 4x6 problem:
        # We have split the 4x4 middle dimension into two 2x2 blocks?
        # No, we split the 4x6 right matrix into two 2x6 blocks vertically?
        # Let's re-evaluate the split.
        
        # A is 2x4. B is 4x6.
        # Let A = [A_left, A_right] where A_left, A_right are 2x2.
        # Let B = [B_top; B_bottom] where B_top, B_bottom are 2x6.
        # C = A_left * B_top + A_right * B_bottom.
        
        # Now we have two products: P1 = A_left * B_top (2x2 * 2x6) and P2 = A_right * B_bottom (2x2 * 2x6).
        # We can decompose each 2x2 * 2x6 product.
        # A 2x2 * 2x6 product can be seen as three 2x2 * 2x2 products (columns of B).
        # So P1 = A_left * [B1_col1, B1_col2, B1_col3]
        #       = A_left*B1_col1 + A_left*B1_col2 + A_left*B1_col3
        # Each term is 2x2 * 2x2.
        # So we have 3 such products for P1 and 3 for P2. Total 6 products of 2x2 matrices.
        
        # For each 2x2 * 2x2 product, we use the rank-7 Strassen decomposition.
        # Total rank = 6 * 7 = 42.
        
        # We need to construct the global U, V, W matrices for this.
        # Global dimensions:
        # U: (2*4, R) = (8, R)
        # V: (4*6, R) = (24, R)
        # W: (6*2, R) = (12, R)
        
        # Let's define the mapping.
        # We have 6 "blocks" of 2x2 matrix multiplication.
        # Block 1: A_left * B1_col1 (contributes to C[:, 0:2])
        # Block 2: A_left * B1_col2 (contributes to C[:, 2:4])
        # Block 3: A_left * B1_col3 (contributes to C[:, 4:6])
        # Block 4: A_right * B2_col1 (contributes to C[:, 0:2])
        # Block 5: A_right * B2_col2 (contributes to C[:, 2:4])
        # Block 6: A_right * B2_col3 (contributes to C[:, 4:6])
        
        # Wait, B is 4x6.
        # B_top is rows 0,1 of B. B_bottom is rows 2,3 of B.
        # B_top is 2x6. Let's split it into 3 columns of 2x2?
        # B_top = [B0, B1, B2] where B0 is cols 0-1, B1 is cols 2-3, B2 is cols 4-5.
        # Similarly B_bottom = [B3, B4, B5].
        
        # So:
        # C[:, 0:2] = A_left * B0 + A_right * B3
        # C[:, 2:4] = A_left * B1 + A_right * B4
        # C[:, 4:6] = A_left * B2 + A_right * B5
        
        # Each term like A_left * B0 is a 2x2 * 2x2 product.
        # We use Strassen for each.
        
        # Strassen Decomposition for one 2x2 * 2x2 product:
        # Inputs: A (2x2), B (2x2).
        # Output: C (2x2).
        # The 7 terms correspond to scalar multiplications.
        # In our tensor decomposition, each "scalar multiplication" m_k is represented by a rank-1 term in the global decomposition.
        # Actually, the Strassen algorithm computes 7 intermediate values m1..m7.
        # Then it combines them linearly to get C.
        # This corresponds to a rank-7 decomposition of the 2x2*2x2 tensor.
        
        # We need to map these 7 terms into the global U, V, W.
        
        # Let's define a helper to add a 2x2*2x2 Strassen block to the global factors.
        # Parameters:
        # - a_row_offset: which rows of A (global) correspond to this 2x2 A.
        # - a_col_offset: which cols of A (global) correspond to this 2x2 A. (A is 2x4, so 0 or 2)
        # - b_row_offset: which rows of B (global) correspond to this 2x2 B. (B is 4x6, so 0, 2, 4, 6, 8, 10? No, B is 4x6, rows 0-3. Cols 0-5.)
        #   Wait, B is 4x6.
        #   B_top rows 0,1. B_bottom rows 2,3.
        #   B0 is B_top cols 0,1 -> Global B rows 0,1 cols 0,1.
        #   B1 is B_top cols 2,3 -> Global B rows 0,1 cols 2,3.
        #   B2 is B_top cols 4,5 -> Global B rows 0,1 cols 4,5.
        #   B3 is B_bottom cols 0,1 -> Global B rows 2,3 cols 0,1.
        #   B4 is B_bottom cols 2,3 -> Global B rows 2,3 cols 2,3.
        #   B5 is B_bottom cols 4,5 -> Global B rows 2,3 cols 4,5.
        
        # - c_col_offset: which cols of C (global) this block contributes to.
        #   C is 2x6.
        #   Block 1 (B0) contributes to C cols 0,1.
        #   Block 2 (B1) contributes to C cols 2,3.
        #   Block 3 (B2) contributes to C cols 4,5.
        #   Block 4 (B3) contributes to C cols 0,1.
        #   Block 5 (B4) contributes to C cols 2,3.
        #   Block 6 (B5) contributes to C cols 4,5.
        
        # - a_is_left: True if using A_left (cols 0,1 of A), False if A_right (cols 2,3).
        
        def add_strassen_block(U, V, W, rank_start, a_cols, b_rows, c_cols):
            # a_cols: 0 or 2 (offset in A's columns)
            # b_rows: 0 or 2 (offset in B's rows)
            # c_cols: 0, 2, or 4 (offset in C's columns)
            
            # A is 2x4. A_left is cols 0,1. A_right is cols 2,3.
            # B is 4x6. B_top is rows 0,1. B_bottom is rows 2,3.
            
            # The 7 Strassen terms:
            # We define the coefficients for U, V, W for each of the 7 terms.
            # For term t (0..6):
            # U[i, j, t] is non-zero for i in {0,1} (rows of A) and j in a_cols, a_cols+1
            # V[j, k, t] is non-zero for j in b_rows, b_rows+1 and k in c_cols, c_cols+1
            # W[k, i, t] is non-zero for k in c_cols, c_cols+1 and i in {0,1}
            
            # Strassen coefficients for 2x2:
            # m1 = (a00+a11)*(b00+b11)
            # m2 = (a10+a11)*b00
            # m3 = a00*(b01-b11)
            # m4 = a11*(b10-b00)
            # m5 = (a00+a01)*b11
            # m6 = (a10-a00)*(b00+b01)
            # m7 = (a01-a11)*(b10+b11)
            
            # We need to map these to the tensor indices.
            # Let local A indices be (li, lj) in 0..1.
            # Let local B indices be (lj, lk) in 0..1.
            # Let local C indices be (lk, li) in 0..1.
            
            # Global indices:
            # U row: i (0..1), col: a_cols + lj
            # V row: b_rows + lj, col: c_cols + lk
            # W row: c_cols + lk, col: i
            
            # Coefficients for each term t:
            # We define U_coeff[li, lj], V_coeff[lj, lk], W_coeff[lk, li]
            
            terms = []
            
            # Term 0 (m1): (a00+a11)*(b00+b11)
            # U: a00 (1,1), a11 (1,1) -> U[0,0]=1, U[1,1]=1
            # V: b00 (1,1), b11 (1,1) -> V[0,0]=1, V[1,1]=1
            # W: 1
            terms.append({
                'U': { (0,0): 1, (1,1): 1 },
                'V': { (0,0): 1, (1,1): 1 },
                'W': { (0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1 } # Wait, W is scalar? No, W is part of the tensor.
                # The term is sum U V W.
                # If U and V are fixed, W must be 1 for the "1" coefficient?
                # No, the Strassen algorithm computes m1, then adds it to C00, C11, etc.
                # This means the final C is a linear combination of the 7 m's.
                # So W is not just 1. W encodes the linear combination coefficients.
                
                # Let's rethink.
                # The decomposition is T = sum_r U_r V_r W_r.
                # For the 2x2*2x2 tensor, the rank-7 decomposition is:
                # C00 = m1 + m4 - m5 + m7
                # C01 = m3 + m5
                # C10 = m2 + m4
                # C11 = m1 - m2 + m3 + m6
                
                # So for each m_k, we have a rank-1 tensor?
                # No, m_k is a scalar product of linear forms.
                # m1 = (a00+a11)(b00+b11).
                # This is a rank-1 tensor in the (A, B) space?
                # Yes.
                # But we need to map it to C.
                # The contribution of m1 to C is:
                # C00 += 1 * m1
                # C11 += 1 * m1
                # So W for m1 has non-zero entries at (0,0) and (1,1) with value 1.
            })
            
            # This is getting complicated to hardcode.
            # Let's generate the rank-7 decomposition programmatically.
            
            return rank_start + 7
        
        # Instead of hardcoding, let's use a known rank-7 decomposition for 2x2.
        # And map it.
        
        # Let's try a simpler structural decomposition first.
        # Rank 42 is a good target.
        
        # We will build U, V, W for the 42-rank solution.
        
        U_42 = [[0] * 42 for _ in range(8)]
        V_42 = [[0] * 42 for _ in range(24)]
        W_42 = [[0] * 42 for _ in range(12)]
        
        # Strassen coefficients for 2x2
        # We define the 7 terms as (U_vec, V_vec, W_vec)
        # U_vec is 4x1 (flattened 2x2), V_vec is 4x1, W_vec is 4x1.
        
        # Term 1: m1 = (a00+a11)(b00+b11)
        # U: [1, 0, 0, 1] (a00, a01, a10, a11) -> indices (0,0), (1,1)
        # V: [1, 0, 0, 1] (b00, b01, b10, b11) -> indices (0,0), (1,1)
        # W: [1, 0, 0, 1] (c00, c01, c10, c11) -> indices (0,0), (1,1)
        
        # Term 2: m2 = (a10+a11)b00
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, 0] (contributes to C10)
        
        # Term 3: m3 = a00(b01-b11)
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 1] (contributes to C01, C11)
        
        # Term 4: m4 = a11(b10-b00)
        # U: [0, 0, 0, 1]
        # V: [0, 0, 1, -1]
        # W: [1, 0, 1, 0] (contributes to C00, C10)
        
        # Term 5: m5 = (a00+a01)b11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: [-1, 0, 0, 1] (contributes to C00, C11)
        
        # Term 6: m6 = (a10-a00)(b00+b01)
        # U: [0, 0, 1, -1]
        # V: [1, 1, 0, 0]
        # W: [0, 0, 0, 1] (contributes to C11)
        
        # Term 7: m7 = (a01-a11)(b10+b11)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: [1, 0, 0, 0] (contributes to C00)
        
        strassen_terms = [
            {
                'U': [(0,0,1), (1,1,1)],
                'V': [(0,0,1), (1,1,1)],
                'W': [(0,0,1), (1,1,1)]
            },
            {
                'U': [(1,0,1), (1,1,1)],
                'V': [(0,0,1)],
                'W': [(1,0,1)]
            },
            {
                'U': [(0,0,1)],
                'V': [(0,1,1), (1,1,-1)],
                'W': [(0,1,1), (1,1,1)]
            },
            {
                'U': [(1,1,1)],
                'V': [(1,0,1), (0,0,-1)],
                'W': [(0,0,1), (1,0,1)]
            },
            {
                'U': [(0,0,1), (0,1,1)],
                'V': [(1,1,1)],
                'W': [(0,0,-1), (1,1,1)]
            },
            {
                'U': [(1,0,1), (0,0,-1)],
                'V': [(0,0,1), (0,1,1)],
                'W': [(1,1,1)]
            },
            {
                'U': [(0,1,1), (1,1,-1)],
                'V': [(1,0,1), (1,1,1)],
                'W': [(0,0,1)]
            }
        ]
        
        # Now map these 6 blocks.
        # Block 1: A_left (cols 0,1) * B0 (rows 0,1, cols 0,1) -> C cols 0,1
        # Block 2: A_left (cols 0,1) * B1 (rows 0,1, cols 2,3) -> C cols 2,3
        # Block 3: A_left (cols 0,1) * B2 (rows 0,1, cols 4,5) -> C cols 4,5
        # Block 4: A_right (cols 2,3) * B3 (rows 2,3, cols 0,1) -> C cols 0,1
        # Block 5: A_right (cols 2,3) * B4 (rows 2,3, cols 2,3) -> C cols 2,3
        # Block 6: A_right (cols 2,3) * B5 (rows 2,3, cols 4,5) -> C cols 4,5
        
        blocks = [
            (0, 0, 0), # A_col_offset, B_row_offset, C_col_offset
            (0, 0, 2),
            (0, 0, 4),
            (2, 2, 0),
            (2, 2, 2),
            (2, 2, 4)
        ]
        
        rank_offset = 0
        for a_off, b_off, c_off in blocks:
            for t in strassen_terms:
                r = rank_offset
                
                # Fill U
                for (li, lj, val) in t['U']:
                    # Global U index: row i=li, col a_off + lj
                    U_42[li][a_off + lj][r] = val
                
                # Fill V
                for (lj, lk, val) in t['V']:
                    # Global V index: row b_off + lj, col c_off + lk
                    V_42[b_off + lj][c_off + lk][r] = val
                
                # Fill W
                for (lk, li, val) in t['W']:
                    # Global W index: row c_off + lk, col li
                    W_42[c_off + lk][li][r] = val
                
                rank_offset += 1
        
        # Verify the 42-rank solution
        U_42_f = [[Fraction(x) for x in row] for row in U_42]
        V_42_f = [[Fraction(x) for x in row] for row in V_42]
        W_42_f = [[Fraction(x) for x in row] for row in W_42]
        
        if is_valid(U_42_f, V_42_f, W_42_f):
            if 42 < current_rank:
                best_factors = {"U": U_42_f, "V": V_42_f, "W": W_42_f}
                current_rank = 42
        
        # Now, can we do better than 42?
        # The theoretical lower bound for 2x4 * 4x6 is not 42.
        # It is at least max(2*4, 4*6, 2*6) = 24? No, that's for rank-1.
        # The rank of matrix multiplication tensor is at least the dimension of the output?
        # No.
        
        # Let's try to search for rank < 42 using local search from the 42 solution.
        # Or just return 42 if time is running out.
        
        if time.time() > deadline - 20:
            return to_output(best_factors)
        
        # Local Search: Try to merge columns in the 42-rank solution.
        # This is complex. Let's stick to 42 for now as it's a solid improvement over 48.
        
        # We can try to see if we can reduce further by using a different split.
        # But 42 is a known good bound for this size using Strassen.
        
        return to_output(best_factors)

# EVOLVE-BLOCK-END
