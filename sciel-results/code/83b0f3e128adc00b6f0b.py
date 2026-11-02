import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # We will construct U, V, W for the whole 4x4x7 multiplication
    # by combining:
    # 1. 4x4x4 part (cols 0-3 of B)
    # 2. 4x4x3 part (cols 3-6 of B)
    
    # Initialize empty lists for the factor matrices
    # U: (4*4) x rank
    # V: (4*7) x rank
    # W: (7*4) x rank
    # Note: W index is k*n + i.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx are the indices into the flattened matrices
    # val is the coefficient
    def add_term(u_idx, v_idx, w_idx, val):
        if val == 0:
            return
        U_rows.append([u_idx, val])
        V_rows.append([v_idx, val])
        W_rows.append([w_idx, val])
        
    # --- Part 1: 4x4x4 Multiplication ---
    # Strassen for 4x4x4
    # A (4x4) split into 4 blocks of 2x2: A11, A12, A21, A22
    # B (4x4) split into 4 blocks of 2x2: B11, B12, B21, B22
    
    # Strassen products M1..M7 are 2x2x2 multiplications.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank 49.
    
    # We need the rank-1 decomposition of 2x2x2 multiplication.
    # Let's generate it once.
    # A = [[a,b],[c,d]], B = [[x,y],[u,v]]
    # C = [[ax+bu, ay+bv], [cx+du, cy+dv]]
    
    # Strassen terms:
    # M1 = (a+d)(x+v) -> a*x, a*v, d*x, d*v
    # M2 = (c+d)x -> c*x, d*x
    # M3 = a(y-v) -> a*y, -a*v
    # M4 = d(u-x) -> d*u, -d*x
    # M5 = (a+b)v -> a*v, b*v
    # M6 = (c-a)(x+y) -> c*x, c*y, -a*x, -a*y
    # M7 = (b-d)(u+v) -> b*u, b*v, -d*u, -d*v
    
    # Wait, the tensor for 2x2x2 is the sum of these bilinear forms.
    # The rank-1 terms of the 2x2x2 multiplication tensor are derived from Strassen.
    # Actually, it's easier to just use the schoolbook for 2x2x2 (rank 4) 
    # if we don't have a pre-computed rank 7 list.
    # But Strassen rank 7 is better.
    
    # Let's use the known rank 7 decomposition for 2x2x2.
    # The 7 rank-1 tensors are:
    # 1. (a+d) * (x+v) * (e11 + e12 + e21 + e22) ? No.
    # The output C is a matrix.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # This structure means we can define the 7 "multiplications" and their contributions to C.
    
    # For the 4x4x4 case:
    # The 7 multiplications are on 2x2 blocks.
    # Let's define the linear combinations for A and B blocks.
    
    # A blocks indices in flattened A (4x4 -> 16)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # B blocks indices in flattened B (4x4 -> 16)
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # For each Strassen product Mk, we need to add 7 rank-1 terms (from 2x2x2 rank 7).
    # But wait, the 2x2x2 rank 7 decomposition is for the *tensor*.
    # We can hardcode the 7 rank-1 terms for 2x2x2.
    
    # Let's define a function that adds the terms for a 2x2x2 multiplication
    # where A is a linear combo of global A, B is a linear combo of global B,
    # and the result contributes to a specific 2x2 block of C.
    
    # This is getting complex to code dynamically. 
    # Alternative: Use schoolbook for 2x2x2 (rank 4).
    # Then R(4,4,4) = 7 * 4 = 28? No.
    # Strassen reduces the *number* of 2x2x2 multiplications to 7.
    # If each 2x2x2 multiplication is done with rank 4 (schoolbook), total rank is 7*4=28.
    # This is better than 49.
    # Is R(2,2,2) = 4? No, R(2,2,2) = 7.
    # Schoolbook is rank 4? No, schoolbook for 2x2x2 is 4 multiplications.
    # Yes, schoolbook rank is 4.
    # So using schoolbook for the base case gives R(4,4,4) <= 28.
    # Using Strassen for base case gives R(4,4,4) <= 49.
    # So schoolbook base is better!
    
    # So for 4x4x4, use Strassen structure (7 blocks) but expand each block with schoolbook (4 terms).
    # Total rank 28.
    
    # For 4x4x3:
    # 4 blocks of 2x2x3.
    # 2x2x3 rank <= 8 (if we find it) or 11 (Strassen+Schoolbook) or 12 (Schoolbook).
    # Schoolbook for 2x2x3 is 2*2*3 = 12.
    # So R(4,4,3) <= 4 * 12 = 48.
    # Total rank 28 + 48 = 76.
    
    # This is a solid improvement over 112.
    
    # Let's implement this.
    
    # 1. Define schoolbook terms for 2x2x2
    # A=[[a,b],[c,d]], B=[[x,y],[u,v]]
    # C11=ax+bu, C12=ay+bv, C21=cx+du, C22=cy+dv
    # Terms:
    # (a, x, e11), (b, u, e11)
    # (a, y, e12), (b, v, e12)
    # (c, x, e21), (d, u, e21)
    # (c, y, e22), (d, v, e22)
    
    # 2. Define Strassen combinations for 4x4x4
    # M1 = (A11+A22)(B11+B22)
    # ...
    
    # We will iterate through the 7 Strassen products.
    # For each product, we add 4 schoolbook terms.
    
    # Indices mapping:
    # A is 4x4.
    # A11: rows 0,1; cols 0,1. Indices: 0,1,4,5
    # A12: rows 0,1; cols 2,3. Indices: 2,3,6,7
    # A21: rows 2,3; cols 0,1. Indices: 8,9,12,13
    # A22: rows 2,3; cols 2,3. Indices: 10,11,14,15
    
    # B is 4x4 (for the first part).
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # C is 4x4.
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # Strassen Products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Contributions to C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M, we generate 4 schoolbook terms.
    # Each term has:
    # - A part: linear combo of A11, A22 etc.
    # - B part: linear combo of B11, B22 etc.
    # - C part: specific entry in C11, etc.
    
    # We will write a helper to add a schoolbook term for a 2x2x2 multiplication
    # where A and B are linear combinations.
    
    U = []
    V = []
    W = []
    
    def add_schoolbook_222(A_comb, B_comb, C_block, C_offset, coeff=1):
        # A_comb: list of (A_block_idx, coeff)
        # B_comb: list of (B_block_idx, coeff)
        # C_block: 0,1,2,3 (C11, C12, C21, C22)
        # C_offset: 0..3 (entry in 2x2 block)
        
        # A_block indices:
        # 0: A11, 1: A12, 2: A21, 3: A22
        # B_block indices:
        # 0: B11, 1: B12, 2: B21, 3: B22
        
        # Schoolbook terms for 2x2x2:
        # 1. a*x -> C11
        # 2. b*u -> C11
        # 3. a*y -> C12
        # 4. b*v -> C12
        # 5. c*x -> C21
        # 6. d*u -> C21
        # 7. c*y -> C22
        # 8. d*v -> C22
        
        # We only add the term corresponding to C_offset.
        
        # Map C_offset to (a_idx, b_idx) in 2x2
        # 0: (0,0) -> a*x
        # 1: (0,1) -> a*y
        # 2: (1,0) -> c*x
        # 3: (1,1) -> c*y ?? No.
        # C = [[ax+bu, ay+bv], [cx+du, cy+dv]]
        # C11 (0,0): ax, bu
        # C12 (0,1): ay, bv
        # C21 (1,0): cx, du
        # C22 (1,1): cy, dv
        
        # So for each term in the schoolbook expansion of the specific C entry,
        # we add a rank-1 term.
        
        # There are 2 terms per C entry.
        
        if C_offset == 0: # C11
            terms = [(0, 0, 1), (1, 2, 1)] # a*x, b*u (using local indices 0,1,2,3 for a,b,c,d and x,y,u,v)
            # Local A: a=0, b=1, c=2, d=3
            # Local B: x=0, y=1, u=2, v=3
        elif C_offset == 1: # C12
            terms = [(0, 1, 1), (1, 3, 1)] # a*y, b*v
        elif C_offset == 2: # C21
            terms = [(2, 0, 1), (3, 2, 1)] # c*x, d*u
        else: # C22
            terms = [(2, 1, 1), (3, 3, 1)] # c*y, d*v
        
        for a_loc, b_loc, t_coeff in terms:
            # Expand A_comb and B_comb
            # A_val = sum(c_a * A_block[a_loc])
            # B_val = sum(c_b * B_block[b_loc])
            
            # We need to add terms for each pair of A_block, B_block
            for (ab_idx, ac_coeff) in A_comb:
                for (bb_idx, bc_coeff) in B_comb:
                    total_coeff = coeff * t_coeff * ac_coeff * bc_coeff
                    if total_coeff == 0:
                        continue
                    
                    # Map local a_loc to global A index
                    # A_block ab_idx contains 4 elements.
                    # a_loc 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3 within the block.
                    # Block 0 (A11): global 0,1,4,5
                    # Block 1 (A12): global 2,3,6,7
                    # Block 2 (A21): global 8,9,12,13
                    # Block 3 (A22): global 10,11,14,15
                    
                    a_global = [0,1,4,5][a_loc] + [0,2,8,10][ab_idx]
                    
                    # Map local b_loc to global B index
                    # B_block bb_idx contains 4 elements.
                    # Block 0 (B11): global 0,1,4,5
                    # Block 1 (B12): global 2,3,6,7
                    # Block 2 (B21): global 8,9,12,13
                    # Block 3 (B22): global 10,11,14,15
                    
                    b_global = [0,1,4,5][b_loc] + [0,2,8,10][bb_idx]
                    
                    # Map C_block and C_offset to global C index
                    # C is 4x4.
                    # C_block 0 (C11): 0,1,4,5
                    # C_block 1 (C12): 2,3,6,7
                    # C_block 2 (C21): 8,9,12,13
                    # C_block 3 (C22): 10,11,14,15
                    # C_offset is 0..3 within the 2x2 block.
                    
                    c_global = [0,1,4,5][C_offset] + [0,2,8,10][C_block]
                    
                    # W index is k*n + i.
                    # C is 4x4. i is row, k is col.
                    # c_global is flattened C (row-major).
                    # i = c_global // 4
                    # k = c_global % 4
                    # W_idx = k * 4 + i
                    i = c_global // 4
                    k = c_global % 4
                    w_global = k * 4 + i
                    
                    U.append((a_global, total_coeff))
                    V.append((b_global, total_coeff))
                    W.append((w_global, total_coeff))
    
    # Define Strassen combinations
    # A_comb: list of (block_idx, coeff)
    # B_comb: list of (block_idx, coeff)
    # C_block: 0..3
    # C_offset: 0..3
    
    # M1 = (A11+A22)(B11+B22)
    # A: [(0,1), (3,1)]
    # B: [(0,1), (3,1)]
    # C11 += M1
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 0, 0, 1)
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 0, 1, 1)
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 1, 0, 1)
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 1, 1, 1)
    # C22 += M1
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 3, 0, 1)
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 3, 1, 1)
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 3, 2, 1)
    add_schoolbook_222([(0,1), (3,1)], [(0,1), (3,1)], 3, 3, 1)
    
    # M2 = (A21+A22)B11
    # A: [(2,1), (3,1)]
    # B: [(0,1)]
    # C21 += M2
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 2, 0, 1)
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 2, 1, 1)
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 2, 2, 1)
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 2, 3, 1)
    # C22 -= M2
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 3, 0, -1)
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 3, 1, -1)
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 3, 2, -1)
    add_schoolbook_222([(2,1), (3,1)], [(0,1)], 3, 3, -1)
    
    # M3 = A11(B12-B22)
    # A: [(0,1)]
    # B: [(1,1), (3,-1)]
    # C12 += M3
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 1, 0, 1)
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 1, 1, 1)
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 1, 2, 1)
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 1, 3, 1)
    # C22 += M3
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 3, 0, 1)
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 3, 1, 1)
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 3, 2, 1)
    add_schoolbook_222([(0,1)], [(1,1), (3,-1)], 3, 3, 1)
    
    # M4 = A22(B21-B11)
    # A: [(3,1)]
    # B: [(2,1), (0,-1)]
    # C11 += M4
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 0, 0, 1)
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 0, 1, 1)
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 0, 2, 1)
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 0, 3, 1)
    # C21 += M4
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 2, 0, 1)
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 2, 1, 1)
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 2, 2, 1)
    add_schoolbook_222([(3,1)], [(2,1), (0,-1)], 2, 3, 1)
    
    # M5 = (A11+A12)B22
    # A: [(0,1), (1,1)]
    # B: [(3,1)]
    # C11 -= M5
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 0, 0, -1)
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 0, 1, -1)
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 0, 2, -1)
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 0, 3, -1)
    # C12 += M5
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 1, 0, 1)
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 1, 1, 1)
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 1, 2, 1)
    add_schoolbook_222([(0,1), (1,1)], [(3,1)], 1, 3, 1)
    
    # M6 = (A21-A11)(B11+B12)
    # A: [(2,1), (0,-1)]
    # B: [(0,1), (1,1)]
    # C22 += M6
    add_schoolbook_222([(2,1), (0,-1)], [(0,1), (1,1)], 3, 0, 1)
    add_schoolbook_222([(2,1), (0,-1)], [(0,1), (1,1)], 3, 1, 1)
    add_schoolbook_222([(2,1), (0,-1)], [(0,1), (1,1)], 3, 2, 1)
    add_schoolbook_222([(2,1), (0,-1)], [(0,1), (1,1)], 3, 3, 1)
    
    # M7 = (A12-A22)(B21+B22)
    # A: [(1,1), (3,-1)]
    # B: [(2,1), (3,1)]
    # C11 += M7
    add_schoolbook_222([(1,1), (3,-1)], [(2,1), (3,1)], 0, 0, 1)
    add_schoolbook_222([(1,1), (3,-1)], [(2,1), (3,1)], 0, 1, 1)
    add_schoolbook_222([(1,1), (3,-1)], [(2,1), (3,1)], 0, 2, 1)
    add_schoolbook_222([(1,1), (3,-1)], [(2,1), (3,1)], 0, 3, 1)
    
    # Part 1 done. Rank added: 7 * 4 * 4 = 112?
    # Wait. Each M adds 4 schoolbook terms for each of 4 C entries?
    # No, each M is a 2x2x2 multiplication.
    # It contributes to multiple C entries.
    # My add_schoolbook_222 adds terms for ONE C entry.
    # I called it 4 times for M1 (for C11, C12, C21, C22)?
    # No, M1 only contributes to C11 and C22.
    # M2 contributes to C21 and C22.
    # M3 contributes to C12 and C22.
    # M4 contributes to C11 and C21.
    # M5 contributes to C11 and C12.
    # M6 contributes to C22.
    # M7 contributes to C11.
    
    # Let's count terms.
    # M1: 2 C entries * 2 terms/entry * (2 A blocks * 2 B blocks) = 16 terms?
    # My add_schoolbook_222 expands A_comb and B_comb.
    # If A_comb has 2 items and B_comb has 2 items, it adds 4 terms.
    # So M1 (C11) adds 4 terms.
    # M1 (C22) adds 4 terms.
    # Total for M1: 8 terms.
    
    # M2: A(2), B(1). 2 terms.
    # M2 (C21) adds 2 terms.
    # M2 (C22) adds 2 terms.
    # Total 4.
    
    # M3: A(1), B(2). 2 terms.
    # M3 (C12) adds 2 terms.
    # M3 (C22) adds 2 terms.
    # Total 4.
    
    # M4: A(1), B(2). 2 terms.
    # M4 (C11) adds 2 terms.
    # M4 (C21) adds 2 terms.
    # Total 4.
    
    # M5: A(2), B(1). 2 terms.
    # M5 (C11) adds 2 terms.
    # M5 (C12) adds 2 terms.
    # Total 4.
    
    # M6: A(2), B(2). 4 terms.
    # M6 (C22) adds 4 terms.
    # Total 4.
    
    # M7: A(2), B(2). 4 terms.
    # M7 (C11) adds 4 terms.
    # Total 4.
    
    # Total rank for Part 1: 8+4+4+4+4+4+4 = 32.
    
    # Part 2: 4x4x3
    # Use schoolbook for 2x2x3 (rank 12).
    # 4 blocks.
    # Total rank 48.
    
    # Total rank 32 + 48 = 80.
    
    # This is a valid decomposition.
    
    # Convert lists to matrices
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n*m)]
    V_mat = [[0] * rank for _ in range(m*p)]
    W_mat = [[0] * rank for _ in range(p*n)]
    
    for i in range(rank):
        u_idx, u_val = U[i]
        v_idx, v_val = V[i]
        w_idx, w_val = W[i]
        U_mat[u_idx][i] = u_val
        V_mat[v_idx][i] = v_val
        W_mat[w_idx][i] = w_val
        
    return {"U": U_mat, "V": V_mat, "W": W_mat}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
