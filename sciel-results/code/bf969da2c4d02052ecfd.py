"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    DEADLINE = 150.0
    start = time.time()

    # Dimensions of the tensor T_{i,j,k} = A_{i,j} B_{j,k}
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # The tensor T is a linear map from the space of (A, B) to C?
    # No, it's a 3-way tensor.
    # We can view it as a matrix M of size (n*p) x (m*m) ? No.
    # T_{i, j, k}. 
    # Let's flatten indices.
    # Index 1: i (n)
    # Index 2: j (m)
    # Index 3: k (p)
    
    # We want to decompose T = sum_r u_r \otimes v_r \otimes w_r.
    # This is equivalent to finding a matrix factorization of a certain flattening.
    # Let's flatten T into a matrix T_mat of size (n*p) x (m*m) ?
    # If we group indices (i, k) for rows and (j) for cols? No, j is in the middle.
    # Standard trick: Matricize T as T_{(i, k), j}.
    # Row index: i * p + k. Col index: j.
    # Wait, T_{i, j, k} depends on A_{i, j} and B_{j, k}.
    # If we fix j, the slice T_{:, j, :} is a matrix of size n x p.
    # T_{i, j, k} = A_{i, j} B_{j, k}.
    # This is an outer product of column j of A and row j of B.
    # So T_{:, j, :} = A[:, j] \otimes B[j, :].
    
    # If we stack these matrices for j=0..m-1, we get a 3D tensor.
    # If we flatten the first and third dimensions (i, k) and keep j, we get a matrix?
    # No, the rank of the tensor is not simply the rank of this matricization.
    # The tensor rank is the minimum r such that T = sum_{s=1}^r u_s \otimes v_s \otimes w_s.
    
    # Let's try a different matricization.
    # Group (i, j) for rows (size n*m) and k for cols (size p).
    # T_{(i, j), k} = A_{i, j} B_{j, k}.
    # For a fixed k, the column is a vector of size n*m.
    # T_{(i, j), k} = sum_j A_{i, j} B_{j, k}.
    # This is just the matrix multiplication result C_{i, k} repeated for each j? No.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So T_{(i, j), k} is NOT just C. It depends on j explicitly in the index.
    # Wait, T_{i, j, k} IS the entry of the tensor.
    # The tensor is defined such that contracting U, V, W gives T.
    # T_{i, j, k} = sum_s U_{i*m+j, s} V_{j*p+k, s} W_{k*n+i, s}.
    
    # Let's use the "randomized rank finding" approach.
    # 1. Generate random vectors r (size n*p) and s (size m*m)?
    # Actually, let's just try to find the rank by checking the rank of a random projection.
    
    # A better approach for small dimensions:
    # The tensor T is in C^{n*m} \otimes C^{m*p} \otimes C^{p*n}.
    # We can form a matrix M of size (n*m * m*p) x (p*n) ? No.
    
    # Let's stick to the known lower bound.
    # Rank of 4x4x4 is 49 (Strassen).
    # Rank of 4x4x9 is likely 49 (for 4x4x4 part) + Rank(4x4x5).
    # Rank of 4x4x5?
    # Schoolbook is 80.
    # Can we do better?
    # 4x4x1 has rank 16 (schoolbook) or less?
    # Rank of 2x2x1 is 4.
    # 4x4x1 can be split into four 2x2x1. Rank 16.
    # 4x4x2?
    # 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1.
    # Rank = 49 + 49 + 16 = 114.
    
    # Is there a better decomposition?
    # Maybe 4x4x9 can be done with rank 112?
    # Let's try to verify if 114 is minimal or if we can squeeze more.
    
    # Strategy:
    # 1. Construct the 114 rank solution (Strassen + Schoolbook).
    # 2. Try to find a lower rank solution using a search?
    #    Search space is huge.
    #    Instead, let's just output the 114 solution. It's a known improvement over 144.
    #    The prompt asks for a "rank-minimal" decomposition.
    #    If 114 is not minimal, we might miss points, but 114 is better than 144.
    #    The current program outputs 144 (schoolbook) because the Strassen implementation was flawed/abandoned.
    #    I will implement the correct 114 rank solution.
    
    # Implementation of 114 rank solution:
    # Part 1: 4x4 * 4x4 (first 4 columns of B) using Strassen (Rank 49).
    # Part 2: 4x4 * 4x4 (next 4 columns of B) using Strassen (Rank 49).
    # Part 3: 4x4 * 4x1 (last 1 column of B) using Schoolbook (Rank 16).
    # Total Rank = 49 + 49 + 16 = 114.
    
    # Wait, the current program's "Strassen" helper was actually just doing 2x2 schoolbook or something?
    # No, the comments said it was trying to do Strassen but failed.
    # I need to write a correct Strassen decomposition for 4x4*4x4.
    
    # Strassen for 4x4:
    # Split A into 2x2 blocks: A11, A12, A21, A22.
    # Split B into 2x2 blocks: B11, B12, B21, B22.
    # 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each Mi is a 2x2 * 2x2 product.
    # We can use Strassen for 2x2*2x2 (Rank 7) or Schoolbook (Rank 8).
    # Using Strassen for the sub-blocks gives Rank 7 * 7 = 49.
    # Using Schoolbook for sub-blocks gives Rank 8 * 7 = 56.
    # 49 is better.
    
    # So I need a function `add_strassen_2x2` that adds 7 rank-1 terms to U, V, W for a 2x2*2x2 multiplication.
    # And a function `add_strassen_4x4` that uses `add_strassen_2x2`?
    # No, `add_strassen_4x4` will construct the 7 linear combinations of A and B blocks,
    # and then call `add_strassen_2x2` for each product?
    # Wait, the linear combinations are on the *entries* of the blocks.
    # The tensor decomposition is linear.
    # If we compute M1 = (A11+A22)(B11+B22), this is a 2x2 matrix.
    # The tensor for M1 is the tensor for 2x2*2x2 multiplication, but with inputs being linear combos.
    # The rank-1 terms for M1 will be:
    # sum_{s=1}^7 u_s \otimes v_s \otimes w_s
    # where u_s, v_s, w_s are the Strassen vectors for 2x2, but mapped to the global indices.
    # The mapping of u_s needs to reflect that the input is A11+A22.
    # So if u_s has a 1 at index (0,0) in the 2x2 local space, it corresponds to A11[0,0] in global space.
    # But wait, the "input" to the 2x2 multiplication for M1 is the matrix A11+A22.
    # So the U vector for the term should pick up A11[0,0] AND A22[0,0]?
    # No, the U matrix maps from the *original* A entries to the rank space.
    # So for the term corresponding to u_s[0,0] (which is A_local[0,0]), we need to add 1 to U at the global index for A11[0,0] AND 1 to U at the global index for A22[0,0].
    
    # So, `add_strassen_4x4` will:
    # 1. Define the 7 Strassen terms for 2x2 (u_s, v_s, w_s).
    # 2. For each of the 7 "M" products, define which global A blocks and B blocks are involved and with what signs.
    # 3. For each rank-1 term in the 2x2 Strassen decomposition, update U, V, W with the appropriate global indices and signs.
    
    # Let's implement this.
    
    def add_rank1_term(U, V, W, u_indices, v_indices, w_indices, sign):
        # u_indices: list of (global_idx, coeff)
        # v_indices: list of (global_idx, coeff)
        # w_indices: list of (global_idx, coeff)
        # This is not quite right.
        # A rank-1 term is u \otimes v \otimes w.
        # U[:, col] += u
        # V[:, col] += v
        # W[:, col] += w
        
        for idx, coeff in u_indices:
            U[idx][col] += sign * coeff
        for idx, coeff in v_indices:
            V[idx][col] += sign * coeff
        for idx, coeff in w_indices:
            W[idx][col] += sign * coeff

    # This is getting complicated to manage indices manually.
    # Let's use a helper to map local 2x2 indices to global indices.
    
    def add_strassen_2x2_block(U, V, W, i0, j0, k0, col_offset, sign=1):
        # Adds the 7 rank-1 terms for a 2x2*2x2 multiplication at offsets i0, j0, k0.
        # Returns the new col_offset.
        # Local indices:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Strassen vectors for 2x2:
        # u1 = [1, 0, 0, 1] (A00, A01, A10, A11) -> A00+A11
        # v1 = [1, 0, 0, 1] (B00, B01, B10, B11) -> B00+B11
        # w1 = [1, 0, 0, 1] (C00, C01, C10, C11) -> C00+C11
        
        # u2 = [0, 0, 1, 1] -> A10+A11
        # v2 = [1, 0, 0, 0] -> B00
        # w2 = [0, 0, 1, -1] -> C10-C11
        
        # u3 = [1, 0, 0, 0] -> A00
        # v3 = [0, 1, 0, -1] -> B01-B11
        # w3 = [0, 1, 0, 1] -> C01+C11
        
        # u4 = [0, 0, 0, 1] -> A11
        # v4 = [-1, 0, 1, 0] -> B10-B00
        # w4 = [1, 0, 1, 0] -> C00+C10
        
        # u5 = [1, 1, 0, 0] -> A00+A01
        # v5 = [0, 0, 0, 1] -> B11
        # w5 = [-1, 1, 0, 0] -> -C00+C01
        
        # u6 = [-1, 0, 1, 0] -> A10-A00
        # v6 = [1, 1, 0, 0] -> B00+B01
        # w6 = [0, 0, 0, 1] -> C11
        
        # u7 = [0, 1, 0, -1] -> A01-A11
        # v7 = [0, 0, 1, 1] -> B10+B11
        # w7 = [1, 0, 0, 0] -> C00
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = col_offset
        for u, v, w in terms:
            # Map local indices to global
            # A local (li, lj) -> global (i0+li, j0+lj) -> idx (i0+li)*m + (j0+lj)
            # B local (lj, lk) -> global (j0+lj, k0+lk) -> idx (j0+lj)*p + (k0+lk)
            # C local (lk, li) -> global (k0+lk, i0+li) -> idx (k0+lk)*n + (i0+li)
            
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    coeff = u[local_idx]
                    if coeff != 0:
                        g_i = i0 + li
                        g_j = j0 + lj
                        U[g_i * m + g_j][col] += sign * coeff
            
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    coeff = v[local_idx]
                    if coeff != 0:
                        g_j = j0 + lj
                        g_k = k0 + lk
                        V[g_j * p + g_k][col] += sign * coeff
            
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    coeff = w[local_idx]
                    if coeff != 0:
                        g_k = k0 + lk
                        g_i = i0 + li
                        W[g_k * n + g_i][col] += sign * coeff
            
            col += 1
        
        return col

    def add_strassen_4x4(U, V, W, i0, j0, k0, col_offset):
        # i0, j0, k0 are the top-left offsets for the 4x4 blocks.
        # We split A (4x4) into 2x2 blocks:
        # A11 at (i0, j0)
        # A12 at (i0, j0+2)
        # A21 at (i0+2, j0)
        # A22 at (i0+2, j0+2)
        
        # B (4x4) into 2x2 blocks:
        # B11 at (j0, k0)
        # B12 at (j0, k0+2)
        # B21 at (j0+2, k0)
        # B22 at (j0+2, k0+2)
        
        # C (4x4) into 2x2 blocks:
        # C11 at (i0, k0)
        # C12 at (i0, k0+2)
        # C21 at (i0+2, k0)
        # C22 at (i0+2, k0+2)
        
        # Strassen products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # This is tricky because the inputs to the 2x2 multiplications are sums/differences of 2x2 blocks.
        # My `add_strassen_2x2_block` assumes the inputs are single 2x2 blocks at specific offsets.
        # To handle sums, I need to call `add_strassen_2x2_block` multiple times with signs?
        # No, that would increase rank.
        # (A11+A22) * (B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This is 4 separate 2x2*2x2 products.
        # If I use Strassen for each, rank is 4*7=28.
        # But Strassen combines them into 7 products total for the 4x4.
        # The "7 products" are 2x2 matrices.
        # Each 2x2 matrix product is decomposed into 7 rank-1 terms.
        # So total rank is 7 * 7 = 49.
        
        # How to implement the linear combinations?
        # The U vector for a term in M1 needs to pick up A11 and A22.
        # My current `add_strassen_2x2_block` only picks up one block.
        # I need a more flexible function or to modify the U, V, W updates directly.
        
        # Let's write a generic `add_term` that takes lists of (idx, coeff) for U, V, W.
        
        col = col_offset
        
        # Define the 7 Strassen terms for 2x2 (u, v, w vectors)
        s_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        # For each of the 7 M products, define the A and B block contributions.
        # A_contrib: list of (block_offset, sign)
        # B_contrib: list of (block_offset, sign)
        # C_contrib: list of (block_offset, sign) -- for the W matrix
        
        # Block offsets (di, dj) for A, (dj, dk) for B, (dk, di) for C
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
        # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
        # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A: A11(+), A22(+)
        # B: B11(+), B22(+)
        # C: C11(+), C22(+)  (Wait, C11 gets M1, C22 gets M1? No.)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes to C11 and C22.
        
        products = [
            # (A_contrib, B_contrib, C_contrib)
            ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)]), # M1
            ([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,-1)]), # M2 -> C21(+), C22(-)
            ([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)]), # M3 -> C12(+), C22(+)
            ([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]), # M4 -> C11(+), C21(+)
            ([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)]), # M5 -> C11(-), C12(+)
            ([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)]), # M6 -> C22(+)
            ([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1)]) # M7 -> C11(+)
        ]
        
        # For each product, iterate through the 7 Strassen rank-1 terms.
        for p_idx, (a_contrib, b_contrib, c_contrib) in enumerate(products):
            u_s, v_s, w_s = s_terms[p_idx]
            
            # Build the full U, V, W vectors for this rank-1 term
            # U vector: size n*m
            # V vector: size m*p
            # W vector: size p*n
            
            # We will add this term to column `col`.
            
            # For U:
            # For each local index (li, lj) in u_s with coeff cu:
            #   For each (ai, aj, sa) in a_contrib:
            #      Global i = i0 + ai + li
            #      Global j = j0 + aj + lj
            #      U[global_i * m + global_j][col] += cu * sa
            
            for li in range(2):
                for lj in range(2):
                    cu = u_s[li * 2 + lj]
                    if cu == 0: continue
                    for ai, aj, sa in a_contrib:
                        gi = i0 + ai + li
                        gj = j0 + aj + lj
                        U[gi * m + gj][col] += cu * sa
            
            # For V:
            # Local index (lj, lk) in v_s with coeff cv:
            #   For each (bj, bk, sb) in b_contrib:
            #      Global j = j0 + bj + lj
            #      Global k = k0 + bk + lk
            #      V[global_j * p + global_k][col] += cv * sb
            
            for lj in range(2):
                for lk in range(2):
                    cv = v_s[lj * 2 + lk]
                    if cv == 0: continue
                    for bj, bk, sb in b_contrib:
                        gj = j0 + bj + lj
                        gk = k0 + bk + lk
                        V[gj * p + gk][col] += cv * sb
            
            # For W:
            # Local index (lk, li) in w_s with coeff cw:
            #   For each (ci, ck, sc) in c_contrib:
            #      Global i = i0 + ci + li
            #      Global k = k0 + ck + lk
            #      W[global_k * n + global_i][col] += cw * sc
            
            for lk in range(2):
                for li in range(2):
                    cw = w_s[lk * 2 + li]
                    if cw == 0: continue
                    for ci, ck, sc in c_contrib:
                        gi = i0 + ci + li
                        gk = k0 + ck + lk
                        W[gk * n + gi][col] += cw * sc
            
            col += 1
            
        return col

    def add_schoolbook_4x1(U, V, W, i0, j0, k0, col_offset):
        # 4x4 * 4x1
        # Rank 16.
        col = col_offset
        for i in range(4):
            for j in range(4):
                U[(i0+i)*m + (j0+j)][col] = 1
                V[(j0+j)*p + (k0+0)][col] = 1
                W[(k0+0)*n + (i0+i)][col] = 1
                col += 1
        return col

    # Total rank: 49 + 49 + 16 = 114
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # B is 4x9.
    # Split into B1 (4x4), B2 (4x4), B3 (4x1).
    # A is 4x4.
    
    # A * B1 (cols 0-3 of B)
    col = add_strassen_4x4(U, V, W, 0, 0, 0, col)
    
    # A * B2 (cols 4-7 of B)
    col = add_strassen_4x4(U, V, W, 0, 0, 4, col)
    
    # A * B3 (col 8 of B)
    col = add_schoolbook_4x1(U, V, W, 0, 0, 8, col)
    
    # Check time
    if time.time() > DEADLINE - 5:
         return {"U": U, "V": V, "W": W}
         
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
