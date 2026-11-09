"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in two parts:
    # Part 1: 4x4 * 4x4 multiplication (columns 0-3 of B)
    # Part 2: 4x4 * 4x1 multiplication (column 4 of B)
    
    # Part 1 Rank: Recursive Strassen for 4x4x4 is rank 49.
    # Part 2 Rank: Schoolbook for 4x4x1 is rank 16.
    # Total Rank: 49 + 16 = 65.
    
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    # This part is straightforward.
    # A (4x4) * b (4x1)
    # U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
    # Here k=4, p=5, n=4
    # V index: j*5 + 4
    # W index: 4*4 + i = 16 + i
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4
    
    # --- Part 1: Recursive Strassen for 4x4x4 ---
    # We decompose the 4x4x4 multiplication into 7 products of 2x2x2 matrices.
    # Each 2x2x2 multiplication is decomposed using Strassen (rank 7).
    
    # Helper to get global index for A element
    # A is 4x4. Index i*m + j.
    def a_idx(i, j):
        return i * m + j
    
    # Helper to get global index for B element
    # B is 4x5. We are using columns 0-3.
    # Index j*p + k.
    def b_idx(j, k):
        return j * p + k
    
    # Helper to get global index for C element
    # C is 4x5. We are using columns 0-3.
    # W index is k*n + i.
    def c_idx(i, k):
        return k * n + i
    
    # Define the 7 Strassen products for 4x4x4
    # Each product Mk corresponds to a 2x2 matrix multiplication.
    # The inputs to Mk are linear combinations of the 2x2 blocks of A and B.
    # The output of Mk is a 2x2 matrix which contributes to the blocks of C.
    
    # Blocks of A (4x4):
    # A00 (0:2, 0:2), A01 (0:2, 2:4)
    # A10 (2:4, 0:2), A11 (2:4, 2:4)
    
    # Blocks of B (4x4 part):
    # B00 (0:2, 0:2), B01 (0:2, 2:4)
    # B10 (2:4, 0:2), B11 (2:4, 2:4)
    
    # Blocks of C (4x4 part):
    # C00 (0:2, 0:2), C01 (0:2, 2:4)
    # C10 (2:4, 0:2), C11 (2:4, 2:4)
    
    # Strassen definitions for 4x4:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output combination:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mk (2x2x2) into 7 rank-1 terms (Strassen for 2x2x2).
    # Let's define a function to add a 2x2x2 Strassen decomposition to U, V, W
    # given the linear combinations for the 2x2 blocks.
    
    # The 2x2x2 Strassen decomposition (rank 7):
    # Local indices for 2x2 matrices:
    # U_local: 4 entries (2x2)
    # V_local: 4 entries (2x2)
    # W_local: 4 entries (2x2)
    
    # Strassen terms for 2x2:
    # T1 = (A00+A11)(B00+B11)
    # T2 = (A10+A11)B00
    # T3 = A00(B01-B11)
    # T4 = A11(B10-B00)
    # T5 = (A00+A01)B11
    # T6 = (A10-A00)(B00+B01)
    # T7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # We will iterate over the 7 Mk products.
    # For each Mk, we have a set of coefficients for the 2x2 blocks of A' and B'.
    # A' blocks: A'_00, A'_01, A'_10, A'_11 (each 2x2)
    # B' blocks: B'_00, B'_01, B'_10, B'_11 (each 2x2)
    
    # The coefficients are linear combinations of the global 2x2 blocks.
    # E.g., for M1:
    # A'_00 = A00 + A11
    # A'_01 = 0
    # A'_10 = 0
    # A'_11 = 0
    # Wait, the blocks are 2x2 matrices.
    # A00 is a 2x2 matrix.
    # So A'_00 is the sum of the elements of A00 and A11.
    
    # Let's define the linear combinations for each Mk.
    # L_A[mk][block_idx][u][v] = list of (global_a_idx, coeff)
    # R_B[mk][block_idx][u][v] = list of (global_b_idx, coeff)
    
    # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in 2x2 block space.
    # u, v: 0,1 local indices within the 2x2 block.
    
    def get_a_coeffs(mk, bi, u, v):
        # bi: 0,1,2,3 corresponding to blocks 00, 01, 10, 11
        # u, v: local row, col in 2x2 block
        # Returns list of (global_a_idx, coeff)
        
        # Global indices for the 4 elements of the 2x2 block bi
        # Block 00: rows 0-1, cols 0-1
        # Block 01: rows 0-1, cols 2-3
        # Block 10: rows 2-3, cols 0-1
        # Block 11: rows 2-3, cols 2-3
        
        base_row = (bi // 2) * 2
        base_col = (bi % 2) * 2
        global_idx = a_idx(base_row + u, base_col + v)
        
        if mk == 0: # M1: A' = A00 + A11
            # A00 is block 0, A11 is block 3
            if bi == 0:
                return [(a_idx(u, v), 1.0), (a_idx(2+u, 2+v), 1.0)]
            elif bi == 3:
                return [(a_idx(u, v), 1.0), (a_idx(2+u, 2+v), 1.0)]
            else:
                return []
        elif mk == 1: # M2: A' = A10 + A11
            # A10 is block 2, A11 is block 3
            if bi == 2:
                return [(a_idx(2+u, v), 1.0), (a_idx(2+u, 2+v), 1.0)]
            elif bi == 3:
                return [(a_idx(2+u, v), 1.0), (a_idx(2+u, 2+v), 1.0)]
            else:
                return []
        elif mk == 2: # M3: A' = A00
            if bi == 0:
                return [(a_idx(u, v), 1.0)]
            else:
                return []
        elif mk == 3: # M4: A' = A11
            if bi == 3:
                return [(a_idx(2+u, 2+v), 1.0)]
            else:
                return []
        elif mk == 4: # M5: A' = A00 + A01
            # A00 is block 0, A01 is block 1
            if bi == 0:
                return [(a_idx(u, v), 1.0), (a_idx(u, 2+v), 1.0)]
            elif bi == 1:
                return [(a_idx(u, v), 1.0), (a_idx(u, 2+v), 1.0)]
            else:
                return []
        elif mk == 5: # M6: A' = A10 - A00
            # A10 is block 2, A00 is block 0
            if bi == 2:
                return [(a_idx(2+u, v), 1.0), (a_idx(u, v), -1.0)]
            elif bi == 0:
                return [(a_idx(2+u, v), 1.0), (a_idx(u, v), -1.0)]
            else:
                return []
        elif mk == 6: # M7: A' = A01 - A11
            # A01 is block 1, A11 is block 3
            if bi == 1:
                return [(a_idx(u, 2+v), 1.0), (a_idx(2+u, 2+v), -1.0)]
            elif bi == 3:
                return [(a_idx(u, 2+v), 1.0), (a_idx(2+u, 2+v), -1.0)]
            else:
                return []
        
        return []

    def get_b_coeffs(mk, bi, u, v):
        # bi: 0,1,2,3 corresponding to blocks 00, 01, 10, 11
        # u, v: local row, col in 2x2 block
        # Returns list of (global_b_idx, coeff)
        
        base_row = (bi // 2) * 2
        base_col = (bi % 2) * 2
        # Note: B is 4x5, but we only use cols 0-3.
        # So global col is base_col + v.
        
        if mk == 0: # M1: B' = B00 + B11
            if bi == 0:
                return [(b_idx(u, v), 1.0), (b_idx(2+u, 2+v), 1.0)]
            elif bi == 3:
                return [(b_idx(u, v), 1.0), (b_idx(2+u, 2+v), 1.0)]
            else:
                return []
        elif mk == 1: # M2: B' = B00
            if bi == 0:
                return [(b_idx(u, v), 1.0)]
            else:
                return []
        elif mk == 2: # M3: B' = B01 - B11
            # B01 is block 1, B11 is block 3
            if bi == 1:
                return [(b_idx(u, 2+v), 1.0), (b_idx(2+u, 2+v), -1.0)]
            elif bi == 3:
                return [(b_idx(u, 2+v), 1.0), (b_idx(2+u, 2+v), -1.0)]
            else:
                return []
        elif mk == 3: # M4: B' = B10 - B00
            # B10 is block 2, B00 is block 0
            if bi == 2:
                return [(b_idx(2+u, v), 1.0), (b_idx(u, v), -1.0)]
            elif bi == 0:
                return [(b_idx(2+u, v), 1.0), (b_idx(u, v), -1.0)]
            else:
                return []
        elif mk == 4: # M5: B' = B11
            if bi == 3:
                return [(b_idx(2+u, 2+v), 1.0)]
            else:
                return []
        elif mk == 5: # M6: B' = B00 + B01
            # B00 is block 0, B01 is block 1
            if bi == 0:
                return [(b_idx(u, v), 1.0), (b_idx(u, 2+v), 1.0)]
            elif bi == 1:
                return [(b_idx(u, v), 1.0), (b_idx(u, 2+v), 1.0)]
            else:
                return []
        elif mk == 6: # M7: B' = B10 + B11
            # B10 is block 2, B11 is block 3
            if bi == 2:
                return [(b_idx(2+u, v), 1.0), (b_idx(2+u, 2+v), 1.0)]
            elif bi == 3:
                return [(b_idx(2+u, v), 1.0), (b_idx(2+u, 2+v), 1.0)]
            else:
                return []
        
        return []

    # Now, for each Mk (0..6), we decompose the 2x2x2 multiplication into 7 terms (T1..T7).
    # Each term Tt is a rank-1 tensor in the local 2x2 space.
    # We need to map the local indices to the global indices using the coefficients above.
    
    # The 2x2x2 Strassen terms T1..T7:
    # T1 = (A'_00 + A'_11) * (B'_00 + B'_11)
    # T2 = (A'_10 + A'_11) * B'_00
    # T3 = A'_00 * (B'_01 - B'_11)
    # T4 = A'_11 * (B'_10 - B'_00)
    # T5 = (A'_00 + A'_01) * B'_11
    # T6 = (A'_10 - A'_00) * (B'_00 + B'_01)
    # T7 = (A'_01 - A'_11) * (B'_10 + B'_11)
    
    # For each term Tt, we have a linear combination of A' elements and B' elements.
    # A' elements are linear combinations of global A elements.
    # B' elements are linear combinations of global B elements.
    # The product is a sum of outer products.
    # Wait, no. The term Tt is a matrix product of two 2x2 matrices.
    # Tt = L_t(A') * R_t(B')
    # L_t(A') is a 2x2 matrix. R_t(B') is a 2x2 matrix.
    # Tt is the matrix product.
    # The tensor decomposition of a 2x2x2 multiplication is:
    # sum_{u,v,w} A_{u,w} B_{w,v} C_{u,v}
    # In Strassen, we replace the 8 multiplications with 7.
    # Each of the 7 multiplications is a scalar multiplication of linear combinations.
    # Let S_t = (linear combo of A) * (linear combo of B).
    # Then C is a linear combo of S_t.
    
    # So, for each Mk, we have 7 scalar products S_{mk, t}.
    # Each S_{mk, t} corresponds to one column in the decomposition.
    # U_col = vector of coefficients for A in the linear combo.
    # V_col = vector of coefficients for B in the linear combo.
    # W_col = vector of coefficients for C in the output combination.
    
    # Let's define the linear combinations for the 7 terms of 2x2x2 Strassen.
    # Term t:
    # A_coeffs[t]: dict of (u, w) -> coeff
    # B_coeffs[t]: dict of (w, v) -> coeff
    # C_coeffs[t]: dict of (u, v) -> coeff
    
    # T1: (A00+A11)(B00+B11)
    # A: A00=1, A11=1. Others 0.
    # B: B00=1, B11=1. Others 0.
    # C: C00=1, C11=1. Others 0. (Wait, C00 = T1... so coeff 1 for C00. C11 = T1... so coeff 1 for C11.)
    
    # Let's list them:
    # T1: A: {(0,0):1, (1,1):1}, B: {(0,0):1, (1,1):1}, C: {(0,0):1, (1,1):1}
    # T2: A: {(1,0):1, (1,1):1}, B: {(0,0):1}, C: {(1,0):1}
    # T3: A: {(0,0):1}, B: {(0,1):1, (1,1):-1}, C: {(0,1):1}
    # T4: A: {(1,1):1}, B: {(1,0):1, (0,0):-1}, C: {(0,0):1, (1,0):1}
    # T5: A: {(0,0):1, (0,1):1}, B: {(1,1):1}, C: {(0,0):-1, (0,1):1}
    # T6: A: {(1,0):1, (0,0):-1}, B: {(0,0):1, (0,1):1}, C: {(1,1):1}
    # T7: A: {(0,1):1, (1,1):-1}, B: {(1,0):1, (1,1):1}, C: {(0,0):1, (1,1):-1}
    
    # Note: The local indices (u, w) for A correspond to A_{u, w}.
    # The local indices (w, v) for B correspond to B_{w, v}.
    # The local indices (u, v) for C correspond to C_{u, v}.
    
    # We need to map these local indices to global indices.
    # For a given Mk, the local A elements are linear combinations of global A elements.
    # Specifically, A'_{u, w} is given by get_a_coeffs(mk, bi_a, u_a, w_a) where bi_a is the block index.
    # Wait, the local 2x2 matrix A' has elements A'_{u, w} for u,w in 0,1.
    # These elements correspond to specific positions in the 4x4 matrix A.
    # The block structure of A' is defined by the Strassen combination.
    # E.g., for M1, A' = A00 + A11.
    # This means A'_{u, w} = A_{u, w} + A_{2+u, 2+w}.
    # So the local element (u, w) of A' corresponds to the sum of two global elements.
    
    # So, for a term Tt in Mk:
    # The linear combination for A is:
    # sum_{u,w} A_local_coeffs[t][(u,w)] * A'_{u,w}
    # = sum_{u,w} A_local_coeffs[t][(u,w)] * sum_{ga} coeff_a(mk, u, w, ga) * A_{ga}
    # = sum_{ga} (sum_{u,w} A_local_coeffs[t][(u,w)] * coeff_a(mk, u, w, ga)) * A_{ga}
    
    # Similarly for B and C.
    
    # Let's precompute the coefficients for each term.
    
    # A_local_coeffs for T1..T7
    A_local = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    
    B_local = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    
    C_local = [
        {(0,0):1, (1,1):1},
        {(1,0):1},
        {(0,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1, (1,1):-1}
    ]
    
    # Now, for each mk (0..6) and each term t (0..6), we compute the global coefficients.
    
    for mk in range(7):
        for t in range(7):
            col = mk * 7 + t
            
            # Compute global U coefficients
            U_coeffs = {}
            for (lu, lw), coeff in A_local[t].items():
                # A'_{lu, lw} is a linear combo of global A elements.
                # We need to find which global block bi corresponds to local (lu, lw) in A'.
                # Wait, the local indices (lu, lw) are within the 2x2 matrix A'.
                # The matrix A' is formed by combining 2x2 blocks of A.
                # The combination is defined by get_a_coeffs.
                # get_a_coeffs(mk, bi, u, v) gives the coeffs for the element at local (u, v) in block bi?
                # No, get_a_coeffs(mk, bi, u, v) returns the global coeffs for the element at global position (block bi, local u, v).
                # But in the Strassen formula for 2x2x2, the local indices (u, w) refer to the elements of the 2x2 matrix A'.
                # The matrix A' is a 2x2 matrix. Its elements are A'_{0,0}, A'_{0,1}, A'_{1,0}, A'_{1,1}.
                # These elements are linear combinations of the elements of the 4x4 matrix A.
                # Specifically, A'_{u, w} is formed from the blocks of A.
                # The block structure of A' is not 2x2 blocks of 2x2 matrices, but rather the 2x2 matrix A' itself is the object.
                # The elements of A' are linear combinations of the elements of A.
                # For M1, A' = A00 + A11.
                # A'_{0,0} = A_{0,0} + A_{2,2}
                # A'_{0,1} = A_{0,1} + A_{2,3}
                # A'_{1,0} = A_{1,0} + A_{3,2}
                # A'_{1,1} = A_{1,1} + A_{3,3}
                
                # So, for local index (lu, lw) in A', we need to find the global coefficients.
                # This is exactly what get_a_coeffs does, but we need to map (lu, lw) to the correct block.
                # The local index (lu, lw) in the 2x2 matrix A' corresponds to:
                # If A' is formed from blocks, the block index bi is not fixed.
                # Actually, the local index (lu, lw) in the 2x2 matrix A' corresponds to the element at row lu, col lw of the 2x2 matrix.
                # The 2x2 matrix A' is composed of elements from the 4x4 matrix A.
                # The mapping from local (lu, lw) to global (i, j) depends on the Strassen combination.
                
                # Let's re-evaluate get_a_coeffs.
                # get_a_coeffs(mk, bi, u, v) was defined to return coeffs for the element at global position (block bi, local u, v).
                # This is not what we need. We need the coeffs for the element A'_{lu, lw}.
                # A'_{lu, lw} is a linear combination of global A elements.
                # For M1, A'_{lu, lw} = A_{lu, lw} + A_{2+lu, 2+lw}.
                # So the global indices are (lu, lw) and (2+lu, 2+lw).
                
                # Let's define a function get_a_global_coeffs(mk, lu, lw) that returns the list of (global_idx, coeff) for A'_{lu, lw}.
                
                # Similarly for B and C.
                
                pass # Placeholder, will implement below
            
            # We need to rewrite the coefficient retrieval logic.
            
            # Let's define helper functions that return the global coeffs for a local index in the 2x2 matrix.
            
            def get_a_global(mk, lu, lw):
                # Returns list of (global_idx, coeff) for A'_{lu, lw}
                if mk == 0: # M1: A' = A00 + A11
                    return [(a_idx(lu, lw), 1.0), (a_idx(2+lu, 2+lw), 1.0)]
                elif mk == 1: # M2: A' = A10 + A11
                    # A10 is rows 2-3, cols 0-1. A11 is rows 2-3, cols 2-3.
                    # A'_{lu, lw} = A_{2+lu, lw} + A_{2+lu, 2+lw}
                    return [(a_idx(2+lu, lw), 1.0), (a_idx(2+lu, 2+lw), 1.0)]
                elif mk == 2: # M3: A' = A00
                    return [(a_idx(lu, lw), 1.0)]
                elif mk == 3: # M4: A' = A11
                    return [(a_idx(2+lu, 2+lw), 1.0)]
                elif mk == 4: # M5: A' = A00 + A01
                    # A00 is rows 0-1, cols 0-1. A01 is rows 0-1, cols 2-3.
                    # A'_{lu, lw} = A_{lu, lw} + A_{lu, 2+lw}
                    return [(a_idx(lu, lw), 1.0), (a_idx(lu, 2+lw), 1.0)]
                elif mk == 5: # M6: A' = A10 - A00
                    # A10 is rows 2-3, cols 0-1. A00 is rows 0-1, cols 0-1.
                    # A'_{lu, lw} = A_{2+lu, lw} - A_{lu, lw}
                    return [(a_idx(2+lu, lw), 1.0), (a_idx(lu, lw), -1.0)]
                elif mk == 6: # M7: A' = A01 - A11
                    # A01 is rows 0-1, cols 2-3. A11 is rows 2-3, cols 2-3.
                    # A'_{lu, lw} = A_{lu, 2+lw} - A_{2+lu, 2+lw}
                    return [(a_idx(lu, 2+lw), 1.0), (a_idx(2+lu, 2+lw), -1.0)]
                return []

            def get_b_global(mk, lw, lv):
                # Returns list of (global_idx, coeff) for B'_{lw, lv}
                # Note: B' is 2x2. Indices are (row, col).
                # In the product A'B', the row index of B' is the col index of A'.
                # So B'_{lw, lv} corresponds to B' element at row lw, col lv.
                
                if mk == 0: # M1: B' = B00 + B11
                    return [(b_idx(lw, lv), 1.0), (b_idx(2+lw, 2+lv), 1.0)]
                elif mk == 1: # M2: B' = B00
                    return [(b_idx(lw, lv), 1.0)]
                elif mk == 2: # M3: B' = B01 - B11
                    # B01 is rows 0-1, cols 2-3. B11 is rows 2-3, cols 2-3.
                    # B'_{lw, lv} = B_{lw, 2+lv} - B_{2+lw, 2+lv}
                    return [(b_idx(lw, 2+lv), 1.0), (b_idx(2+lw, 2+lv), -1.0)]
                elif mk == 3: # M4: B' = B10 - B00
                    # B10 is rows 2-3, cols 0-1. B00 is rows 0-1, cols 0-1.
                    # B'_{lw, lv} = B_{2+lw, lv} - B_{lw, lv}
                    return [(b_idx(2+lw, lv), 1.0), (b_idx(lw, lv), -1.0)]
                elif mk == 4: # M5: B' = B11
                    return [(b_idx(2+lw, 2+lv), 1.0)]
                elif mk == 5: # M6: B' = B00 + B01
                    # B00 is rows 0-1, cols 0-1. B01 is rows 0-1, cols 2-3.
                    # B'_{lw, lv} = B_{lw, lv} + B_{lw, 2+lv}
                    return [(b_idx(lw, lv), 1.0), (b_idx(lw, 2+lv), 1.0)]
                elif mk == 6: # M7: B' = B10 + B11
                    # B10 is rows 2-3, cols 0-1. B11 is rows 2-3, cols 2-3.
                    # B'_{lw, lv} = B_{2+lw, lv} + B_{2+lw, 2+lv}
                    return [(b_idx(2+lw, lv), 1.0), (b_idx(2+lw, 2+lv), 1.0)]
                return []

            def get_c_global(mk, lu, lv):
                # Returns list of (global_idx, coeff) for C'_{lu, lv}
                # C' is 2x2. It contributes to the blocks of C.
                # The contribution is defined by the output combination.
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # The local C'_{lu, lv} corresponds to the element at row lu, col lv of the 2x2 matrix C'.
                # This element contributes to the global C blocks.
                # The block of C corresponding to local (lu, lv) depends on the block structure.
                # For M1, C' contributes to C00 and C11.
                # C'_{lu, lv} contributes to C_{lu, lv} (in C00) and C_{2+lu, 2+lv} (in C11).
                
                # Let's check the output combination for each mk.
                # The coefficient of C'_{lu, lv} in the global C is determined by which blocks C' is added to.
                
                # For M1: C' adds to C00 and C11.
                # C00 is rows 0-1, cols 0-1. C11 is rows 2-3, cols 2-3.
                # So C'_{lu, lv} contributes to C_{lu, lv} and C_{2+lu, 2+lv}.
                
                # For M2: C' adds to C10 and C11 (with -1).
                # C10 is rows 2-3, cols 0-1. C11 is rows 2-3, cols 2-3.
                # So C'_{lu, lv} contributes to C_{2+lu, lv} and C_{2+lu, 2+lv} (with -1).
                
                # For M3: C' adds to C01 and C11.
                # C01 is rows 0-1, cols 2-3. C11 is rows 2-3, cols 2-3.
                # So C'_{lu, lv} contributes to C_{lu, 2+lv} and C_{2+lu, 2+lv}.
                
                # For M4: C' adds to C00 and C10.
                # C00 is rows 0-1, cols 0-1. C10 is rows 2-3, cols 0-1.
                # So C'_{lu, lv} contributes to C_{lu, lv} and C_{2+lu, lv}.
                
                # For M5: C' adds to C00 (with -1) and C01.
                # C00 is rows 0-1, cols 0-1. C01 is rows 0-1, cols 2-3.
                # So C'_{lu, lv} contributes to C_{lu, lv} (with -1) and C_{lu, 2+lv}.
                
                # For M6: C' adds to C11.
                # C11 is rows 2-3, cols 2-3.
                # So C'_{lu, lv} contributes to C_{2+lu, 2+lv}.
                
                # For M7: C' adds to C00 and C11 (with -1).
                # C00 is rows 0-1, cols 0-1. C11 is rows 2-3, cols 2-3.
                # So C'_{lu, lv} contributes to C_{lu, lv} and C_{2+lu, 2+lv} (with -1).
                
                coeffs = []
                if mk == 0: # M1
                    coeffs.append((c_idx(lu, lv), 1.0))
                    coeffs.append((c_idx(2+lu, 2+lv), 1.0))
                elif mk == 1: # M2
                    coeffs.append((c_idx(2+lu, lv), 1.0))
                    coeffs.append((c_idx(2+lu, 2+lv), -1.0))
                elif mk == 2: # M3
                    coeffs.append((c_idx(lu, 2+lv), 1.0))
                    coeffs.append((c_idx(2+lu, 2+lv), 1.0))
                elif mk == 3: # M4
                    coeffs.append((c_idx(lu, lv), 1.0))
                    coeffs.append((c_idx(2+lu, lv), 1.0))
                elif mk == 4: # M5
                    coeffs.append((c_idx(lu, lv), -1.0))
                    coeffs.append((c_idx(lu, 2+lv), 1.0))
                elif mk == 5: # M6
                    coeffs.append((c_idx(2+lu, 2+lv), 1.0))
                elif mk == 6: # M7
                    coeffs.append((c_idx(lu, lv), 1.0))
                    coeffs.append((c_idx(2+lu, 2+lv), -1.0))
                
                return coeffs

            # Now, accumulate the coefficients for U, V, W for this column.
            
            # U coefficients
            for (lu, lw), coeff_local in A_local[t].items():
                global_coeffs = get_a_global(mk, lu, lw)
                for (ga_idx, coeff_global) in global_coeffs:
                    U[ga_idx, col] += coeff_local * coeff_global
            
            # V coefficients
            for (lw, lv), coeff_local in B_local[t].items():
                global_coeffs = get_b_global(mk, lw, lv)
                for (gb_idx, coeff_global) in global_coeffs:
                    V[gb_idx, col] += coeff_local * coeff_global
            
            # W coefficients
            for (lu, lv), coeff_local in C_local[t].items():
                global_coeffs = get_c_global(mk, lu, lv)
                for (gc_idx, coeff_global) in global_coeffs:
                    W[gc_idx, col] += coeff_local * coeff_global
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
