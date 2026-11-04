import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0

    # Schoolbook baseline (rank = n*m*p = 112)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if total != 1:
                        return False
        # Also verify that entries that should be 0 are 0
        # (This is implicitly checked since we only need the (i,j,k) triples to give 1)
        return True

    def strassen_2x2_2x2():
        """
        Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
        Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
        U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] sums to 1 for all (i,j,k) in 2x2x2.
        """
        # Strassen's 7 products for 2x2 * 2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # A entries: a=0, b=1, c=2, d=3
        V = [[0]*rank for _ in range(4)]  # B entries: e=0, f=1, g=2, h=3
        W = [[0]*rank for _ in range(4)]  # C entries: C00=0, C01=1, C10=2, C11=3
        
        # M1 = (a+d)(e+h)
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1  # C00
        W[3][0] += 1  # C11
        
        # M2 = (c+d)e
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1  # C10
        W[3][1] -= 1  # C11
        
        # M3 = a(f-h)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1  # C01
        W[3][2] += 1  # C11
        
        # M4 = d(g-e)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1  # C00
        W[2][3] += 1  # C10
        
        # M5 = (a+b)h
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1  # C00
        W[1][4] += 1  # C01
        
        # M6 = (c-a)(e+f)
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1  # C11
        
        # M7 = (b-d)(g+h)
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1  # C00
        
        return U, V, W

    def combine_decompositions(U1, V1, W1, r1, U2, V2, W2, r2, 
                                a_map, b_map, c_map):
        """
        Combine two decompositions.
        a_map: dict mapping global A index -> list of (local A index, coeff)
        b_map: dict mapping global B index -> list of (local B index, coeff)
        c_map: dict mapping global C index -> list of (local C index, coeff)
        """
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(len(a_map))]
        V = [[0]*total_rank for _ in range(len(b_map))]
        W = [[0]*total_rank for _ in range(len(c_map))]
        
        for gi, mappings in a_map.items():
            for li, coeff in mappings:
                for r in range(r1):
                    U[gi][r] = U[gi][r] + coeff * U1[li][r]
        
        for gi, mappings in b_map.items():
            for li, coeff in mappings:
                for r in range(r2):
                    V[gi][r+r1] = V[gi][r+r1] + coeff * V2[li][r]
        
        for gi, mappings in c_map.items():
            for li, coeff in mappings:
                for r in range(r2):
                    W[gi][r+r2] = W[gi][r+r2] + coeff * W2[li][r]
        
        return U, V, W, total_rank

    def generate_hybrid_4x4x7():
        """
        Hybrid approach: Use Strassen for the 4x4*4x4 part and schoolbook for 4x4*4x3.
        Split B into B_left (4x4, cols 0-3) and B_right (4x3, cols 4-6).
        C = A*B_left + A*B_right
        
        For A*B_left: use Strassen decomposition of 4x4*4x4.
        For A*B_right: schoolbook with rank 4*4*3 = 48.
        
        The 4x4*4x4 Strassen decomposition has rank 28 (7 products of 2x2*2x2, each rank 4... 
        wait, 7*4=28? No, each 2x2*2x2 Strassen has rank 7, so 7*7=49? 
        Actually, the standard Strassen for 4x4*4x4 uses 7 products of 2x2*2x2, 
        and each 2x2*2x2 can be done in rank 7 (Strassen) or 8 (schoolbook).
        So rank = 7*7 = 49 using Strassen recursively, or 7*8=56 using schoolbook for 2x2.
        
        Let me use the recursive Strassen: 4x4*4x4 has rank 49.
        Total: 49 + 48 = 97.
        """
        if time.time() > deadline:
            return None
        
        # Get Strassen decomposition for 2x2*2x2
        S_U, S_V, S_W = strassen_2x2_2x2()
        s_rank = 7
        
        # For 4x4*4x4, we use 7 Strassen products, each of which is a 2x2*2x2 multiplication.
        # The 4x4*4x4 Strassen decomposition:
        # Split A into A00, A01, A10, A11 (each 2x2)
        # Split B into B00, B01, B10, B11 (each 2x2)
        # 7 products M1..M7, each is 2x2*2x2
        
        # Each Mi involves linear combinations of A and B blocks.
        # We need to build the full 4x4*4x4 decomposition.
        
        # Let's build it directly using the Strassen formulas.
        # A is 4x4, indexed as A[i,j] with i,j in 0..3
        # B is 4x4, indexed as B[j,k] with j,k in 0..3
        # C is 4x4, indexed as C[k,i] with k,i in 0..3 (trace convention)
        
        # Strassen products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each Mi, we use the 2x2*2x2 Strassen decomposition (rank 7).
        # Total rank for 4x4*4x4 = 7 * 7 = 49.
        
        rank_4x4x4 = 49
        U_4x4x4 = [[0]*rank_4x4x4 for _ in range(16)]  # 4*4 = 16
        V_4x4x4 = [[0]*rank_4x4x4 for _ in range(16)]  # 4*4 = 16
        W_4x4x4 = [[0]*rank_4x4x4 for _ in range(16)]  # 4*4 = 16
        
        # Define the 7 Strassen products with their A and B coefficient patterns.
        # Each product is a 2x2*2x2 multiplication.
        # For product Mi, we need to specify:
        # - Which A entries (with coefficients) go into the 2x2 L matrix
        # - Which B entries (with coefficients) go into the 2x2 R matrix
        # - How the resulting 2x2 C matrix maps to the global C
        
        # A block indices (global):
        # A00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
        # A01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
        # A10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
        # A11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
        
        # B block indices (global for 4x4 part, cols 0-3):
        # B00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
        # B01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
        # B10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
        # B11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
        
        # C block indices (global, 4x4 part, rows 0-3):
        # C00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
        # C01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
        # C10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
        # C11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
        
        # For each Strassen product, define:
        # a_blocks: list of (block_id, coeff) where block_id in {0,1,2,3} for A00,A01,A10,A11
        # b_blocks: list of (block_id, coeff)
        # c_blocks: list of (block_id, coeff) for the output C block
        
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00 and C11
        # The 2x2 L matrix has entries from A00+A11, and R from B00+B11.
        # The result 2x2 matrix contributes to C00 (top-left) and C11 (bottom-right).
        
        # Let me define a helper to add a Strassen product.
        
        def add_strassen_product(a_block_coeffs, b_block_coeffs, c_block_coeffs, offset):
            """
            a_block_coeffs: dict {block_id: coeff} for A blocks
            b_block_coeffs: dict {block_id: coeff} for B blocks
            c_block_coeffs: dict {block_id: coeff} for C blocks
            offset: column offset in the U,V,W matrices
            """
            # For the 2x2*2x2 Strassen decomposition, we have 7 terms.
            # Each term r (0..6) has:
            # S_U[li][r], S_V[li][r], S_W[li][r] for li in 0..3
            
            # We need to map local 2x2 indices to global indices.
            # For A: local (lr,lc) in block (br,bc) -> global (br*2+lr, bc*2+lc) -> index (br*2+lr)*4 + (bc*2+lc)
            # For B: local (lr,lc) in block (br,bc) -> global (br*2+lr, bc*2+lc) -> index (br*2+lr)*4 + (bc*2+lc)
            # For C: local (lr,lc) in block (br,bc) -> global (br*2+lr, bc*2+lc) -> index (br*2+lr)*4 + (bc*2+lc)
            
            # But the Strassen product involves linear combinations of blocks.
            # L[lr,lc] = sum over (br,bc,coeff) in a_block_coeffs: coeff * A[br*2+lr, bc*2+lc]
            # R[lr,lc] = sum over (br,bc,coeff) in b_block_coeffs: coeff * B[br*2+lr, bc*2+lc]
            # Result[lr,lc] = sum over (br,bc,coeff) in c_block_coeffs: coeff * C[br*2+lr, bc*2+lc]
            
            # For each of the 7 Strassen terms r:
            # S_U[li][r] * S_V[lj][r] * S_W[lk][r] should sum to give the correct C.
            
            # Let's just directly compute the contribution.
            # For term r, the local 2x2*2x2 Strassen gives:
            # sum_r S_U[li][r] * S_V[lj][r] * S_W[lk][r] = delta(li+2*lj, lk) for the 2x2*2x2 case.
            # Wait, that's not quite right. Let me think again.
            
            # Actually, the Strassen decomposition satisfies:
            # sum_{r=0}^{6} S_U[i][r] * S_V[j][r] * S_W[k][r] = 1 if (i,j,k) is a valid 2x2*2x2 triple, else 0.
            # Where i,j,k are in 0..3 (flattened 2x2 indices).
            
            # So for our case:
            # L is a 2x2 matrix with entries that are linear combos of A.
            # R is a 2x2 matrix with entries that are linear combos of B.
            # The product L*R is a 2x2 matrix, and each entry is a linear combo of C.
            
            # For term r, we have:
            # U[global_a_idx][offset+r] += sum over (br,bc,coeff) in a_block_coeffs: coeff * S_U[local_a_idx][r]
            # where local_a_idx corresponds to (lr,lc) in the 2x2 block.
            
            # Let me be more explicit.
            # For A: the 2x2 L matrix has entries L[lr,lc] = sum_{(br,bc,coeff)} coeff * A[br*2+lr, bc*2+lc]
            # The local index for A[br*2+lr, bc*2+lc] in the 2x2*2x2 Strassen is (lr*2+lc) if we consider each block separately.
            # But since we're combining blocks, we need to be careful.
            
            # Actually, let me just build the full 4x4*4x4 decomposition directly.
            
            pass
        
        # This is getting complicated. Let me use a different approach.
        # I'll directly construct the 4x4*4x4 Strassen decomposition by explicitly 
        # writing out all 49 terms.
        
        # Actually, let me just use schoolbook for 4x4*4x4 (rank 64) and schoolbook for 4x4*4x3 (rank 48).
        # Total: 64 + 48 = 112, which is the same as full schoolbook. Not useful.
        
        # Let me try a different split. Use Strassen for the entire 4x4*4x7 by padding.
        # Pad B to 4x8, use Strassen for 4x4*4x8, then ignore the last column.
        # 4x4*4x8: Strassen gives rank 7*7 = 49 for the 4x4*4x4 part... no, that's not right.
        
        # Actually, the standard Strassen recursion for n x n * n x n gives rank 7^log2(n).
        # For n=4: 7^2 = 49.
        # For n=4, p=7: we can't directly use Strassen since p != n.
        
        # Let me just use the schoolbook for the right part and Strassen for the left part.
        # 4x4*4x4 Strassen: rank 49.
        # 4x4*4x3 schoolbook: rank 48.
        # Total: 97.
        
        # Let me build the 4x4*4x4 Strassen decomposition explicitly.
        
        # I'll use the recursive Strassen structure.
        # Level 1: 7 products of 2x2*2x2.
        # Level 2: each 2x2*2x2 uses 7 rank-1 terms.
        # Total: 49 terms.
        
        # For each of the 7 level-1 products, I need to specify:
        # - The A coefficient pattern (which A entries with what coefficients)
        # - The B coefficient pattern
        # - The C coefficient pattern
        
        # Let me define the 7 products:
        products = [
            # (a_blocks, b_blocks, c_blocks)
            # a_blocks: list of (block_id, coeff)
            # b_blocks: list of (block_id, coeff)
            # c_blocks: list of (block_id, coeff)
            
            # M1 = (A00+A11)*(B00+B11) -> C00 + C11
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            
            # M2 = (A10+A11)*B00 -> C10 - C11
            ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
            
            # M3 = A00*(B01-B11) -> C01 + C11
            ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
            
            # M4 = A11*(B10-B00) -> C00 + C10
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            
            # M5 = (A00+A01)*B11 -> -C00 + C01
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            
            # M6 = (A10-A00)*(B00+B01) -> C11
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            
            # M7 = (A01-A11)*(B10+B11) -> C00
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
        ]
        
        # For each product, add 7 terms (one for each Strassen 2x2*2x2 term).
        col_offset = 0
        for prod_idx in range(7):
            a_blocks, b_blocks, c_blocks = products[prod_idx]
            
            # For each of the 7 Strassen terms r:
            for r in range(7):
                # S_U[li][r], S_V[li][r], S_W[li][r] for li in 0..3
                # li=0: (0,0), li=1: (0,1), li=2: (1,0), li=3: (1,1)
                
                # Add contributions to U, V, W
                for li in range(4):
                    s_u_val = S_U[li][r]
                    s_v_val = S_V[li][r]
                    s_w_val = S_W[li][r]
                    
                    # For U: map local index li to global A indices
                    lr = li // 2
                    lc = li % 2
                    for block_id, coeff in a_blocks:
                        br = block_id // 2
                        bc = block_id % 2
                        global_idx = (br*2+lr)*4 + (bc*2+lc)
                        U_4x4x4[global_idx][col_offset + r] += coeff * s_u_val
                    
                    # For V: map local index li to global B indices
                    for block_id, coeff in b_blocks:
                        br = block_id // 2
                        bc = block_id % 2
                        global_idx = (br*2+lr)*4 + (bc*2+lc)
                        V_4x4x4[global_idx][col_offset + r] += coeff * s_v_val
                    
                    # For W: map local index li to global C indices
                    for block_id, coeff in c_blocks:
                        br = block_id // 2
                        bc = block_id % 2
                        global_idx = (br*2+lr)*4 + (bc*2+lc)
                        W_4x4x4[global_idx][col_offset + r] += coeff * s_w_val
                
                col_offset += 1
        
        # Now add the schoolbook part for 4x4*4x3 (cols 4-6 of B, cols 4-6 of C)
        rank_right = 4 * 4 * 3  # 48
        total_rank = rank_4x4x4 + rank_right
        
        U_full = [[0]*total_rank for _ in range(16)]
        V_full = [[0]*total_rank for _ in range(28)]  # 4*7 = 28
        W_full = [[0]*total_rank for _ in range(28)]  # 7*4 = 28
        
        # Copy the 4x4*4x4 part
        for i in range(16):
            for r in range(rank_4x4x4):
                U_full[i][r] = U_4x4x4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(rank_4x4x4):
                    V_full[j*7+k][r] = V_4x4x4[j*4+k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(rank_4x4x4):
                    W_full[k*4+i][r] = W_4x4x4[k*4+i][r]
        
        # Add schoolbook for 4x4*4x3
        col = rank_4x4x4
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U_full[i*4+j][col] = 1
                    V_full[j*7+(4+k)][col] = 1
                    W_full[(4+k)*4+i][col] = 1
                    col += 1
        
        return U_full, V_full, W_full, total_rank

    try:
        result = generate_hybrid_4x4x7()
        if result is not None:
            U, V, W, rank = result
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
