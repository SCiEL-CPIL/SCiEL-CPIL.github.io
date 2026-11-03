import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We will try to build a Strassen-based decomposition.
    # The idea: 
    # A is 4x4, B is 4x9.
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 blocks (vertically): B = [[B11, B12, B13], [B21, B22, B23]]
    # where each Bjk is 2x3.
    # C = A*B is 4x9: C = [[C11, C12, C13], [C21, C22, C23]] where each Cjk is 2x3.
    
    # Strassen formulas for 2x2 block multiplication:
    # M1 = (A11+A22)*(B11+B21+B22+B23) -- wait, B blocks are 2x3, so B11+B21 is 2x3? No.
    # Let's be precise.
    # A11, A12, A21, A22 are 2x2.
    # B11, B12, B13 are 2x3 (top half of B, columns 0-2, 3-5, 6-8)
    # B21, B22, B23 are 2x3 (bottom half of B, columns 0-2, 3-5, 6-8)
    
    # Standard Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For 2x2 * 2x(2x3) = 2x(2x3):
    # We can group the 3 column blocks of B.
    # Let B = [B1, B2, B3] where each Bj is 2x3.
    # Then M1 = (A11+A22) * (B1 + B2 + B3)  [since B11+B22+B13+B23 = B1+B2+B3]
    # M2 = (A21+A22) * B1
    # M3 = A11 * (B2 - B3)  [B12-B22 = B2-B3? No, B12 is top-mid, B22 is bot-mid. So B12-B22 is not a full column block.]
    
    # This approach is tricky because Strassen assumes square matrices or compatible block structures.
    # Let's try a different split.
    # Pad B to 4x12? No, that increases rank.
    
    # Alternative: Use Strassen for 4x4 * 4x4 (rank 49) and then handle the extra 5 columns of B separately?
    # B is 4x9 = 4x4 + 4x5.
    # C = A*(B1 + B2) = A*B1 + A*B2.
    # A*B1: 4x4 * 4x4. Rank 49 (Strassen).
    # A*B2: 4x4 * 4x5. Rank?
    # For 4x4 * 4x5, we can use schoolbook: 4*4*5 = 80.
    # Total rank: 49 + 80 = 129.
    # This is better than 144.
    
    # Let's implement this.
    # Part 1: Strassen for 4x4 * 4x4.
    # Part 2: Schoolbook for 4x4 * 4x5.
    
    # We need to carefully map the indices.
    # U: 16 x rank
    # V: 36 x rank
    # W: 36 x rank
    
    # Indices:
    # U[i*m+j] = U[4i+j]
    # V[j*p+k] = V[9j+k]
    # W[k*n+i] = W[4k+i]
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))
    
    # --- Part 1: Strassen 4x4 * 4x4 ---
    # A is 4x4, B1 is 4x4 (first 4 columns of B).
    # Split A into 2x2 blocks: A11, A12, A21, A22.
    # Split B1 into 2x2 blocks: B11, B12, B21, B22.
    
    # Strassen products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mx is a 2x2 matrix product. We can use Strassen again (7 terms) or schoolbook (8 terms).
    # Let's use Strassen for the inner 2x2*2x2 to get 7*7=49 terms.
    
    # We need a helper to add a 2x2*2x2 Strassen decomposition.
    # It takes the base indices for A, B, C.
    
    def add_strassen_2x2(base_u_i, base_u_j, base_v_j, base_v_k, base_w_k, base_w_i):
        """
        Adds Strassen terms for a 2x2 * 2x2 multiplication.
        A is at U[base_u_i*4 + base_u_j : base_u_i*4 + base_u_j+4]
        B is at V[base_v_j*9 + base_v_k : base_v_j*9 + base_v_k+4] -- wait, V is 4x9, so V[j*9+k].
        For a 2x2 block of B, the rows are j, j+1 and columns are k, k+1.
        So V indices are (j*9+k), (j*9+k+1), ((j+1)*9+k), ((j+1)*9+k+1).
        Similarly for W.
        """
        # Local indices for 2x2:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Global U index for A[i,j]: (base_u_i + i)*4 + (base_u_j + j)
        # Global V index for B[i,j]: (base_v_j + i)*9 + (base_v_k + j)
        # Global W index for C[i,j]: (base_w_k + j)*4 + (base_w_i + i)
        
        def u_idx(i, j):
            return (base_u_i + i) * 4 + (base_u_j + j)
        
        def v_idx(i, j):
            return (base_v_j + i) * 9 + (base_v_k + j)
        
        def w_idx(i, j):
            return (base_w_k + j) * 4 + (base_w_i + i)
        
        # Strassen terms for 2x2*2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to expand these into elementary terms.
        # Each M is a product of two 2x2 matrices (or sums/differences).
        # We will define the U and V vectors for each M, then add the product terms.
        
        # M1: U = A00+A11, V = B00+B11
        # U1 = e00 + e11, V1 = e00 + e11
        # Terms: (0,0)*(0,0), (0,0)*(1,1), (1,1)*(0,0), (1,1)*(1,1)
        # But we must only keep terms where the j index matches.
        # (0,0) U has j=0, (0,0) V has j=0. Match.
        # (0,0) U has j=0, (1,1) V has j=1. No match.
        # (1,1) U has j=1, (0,0) V has j=0. No match.
        # (1,1) U has j=1, (1,1) V has j=1. Match.
        
        # So M1 contributes:
        # A00*B00 -> C00, C11
        # A11*B11 -> C00, C11
        
        # Let's write a generic function to add a product of two sums.
        def add_product(u_terms, v_terms, w_terms, sign):
            # u_terms: list of (i, j, coeff)
            # v_terms: list of (i, j, coeff)
            # w_terms: list of (i, j, coeff)
            # We add sign * (sum u) * (sum v) * (sum w)
            for ui, uj, uc in u_terms:
                for vi, vj, vc in v_terms:
                    if uj == vi:  # j index must match
                        for wi, wj, wc in w_terms:
                            coeff = sign * uc * vc * wc
                            add_term(u_idx(ui, uj), v_idx(vi, vj), w_idx(wi, wj), coeff)
        
        # Define the 7 products
        # M1
        u1 = [(0,0,1), (1,1,1)]
        v1 = [(0,0,1), (1,1,1)]
        w1 = [(0,0,1), (1,1,1)]  # C00, C11
        
        # M2
        u2 = [(1,0,1), (1,1,1)]
        v2 = [(0,0,1)]
        w2 = [(1,0,1), (1,1,1)]  # C10, C11
        
        # M3
        u3 = [(0,0,1)]
        v3 = [(1,0,1), (1,1,-1)]
        w3 = [(0,1,1), (1,1,1)]  # C01, C11
        
        # M4
        u4 = [(1,1,1)]
        v4 = [(0,1,1), (0,0,-1)]
        w4 = [(0,0,1), (1,0,1)]  # C00, C10
        
        # M5
        u5 = [(0,0,1), (0,1,1)]
        v5 = [(1,1,1)]
        w5 = [(0,0,-1), (0,1,1)]  # C00, C01
        
        # M6
        u6 = [(1,0,1), (0,0,-1)]
        v6 = [(0,0,1), (0,1,1)]
        w6 = [(1,1,1)]  # C11
        
        # M7
        u7 = [(0,1,1), (1,1,-1)]
        v7 = [(1,0,1), (1,1,1)]
        w7 = [(0,0,-1), (1,1,1)]  # C00, C11
        
        # Combine into C
        # C00: M1 + M4 - M5 - M7
        add_product(u1, v1, w1, 1)
        add_product(u4, v4, w4, 1)
        add_product(u5, v5, w5, -1)
        add_product(u7, v7, w7, -1)
        
        # C01: M5 + M3
        add_product(u5, v5, w5, 1)
        add_product(u3, v3, w3, 1)
        
        # C10: M2 + M4
        add_product(u2, v2, w2, 1)
        add_product(u4, v4, w4, 1)
        
        # C11: M1 - M2 + M3 + M6
        add_product(u1, v1, w1, 1)
        add_product(u2, v2, w2, -1)
        add_product(u3, v3, w3, 1)
        add_product(u6, v6, w6, 1)
    
    # Apply Strassen to the 4x4 * 4x4 part.
    # A blocks:
    # A11: i=0,1; j=0,1 -> base_u_i=0, base_u_j=0
    # A12: i=0,1; j=2,3 -> base_u_i=0, base_u_j=2
    # A21: i=2,3; j=0,1 -> base_u_i=2, base_u_j=0
    # A22: i=2,3; j=2,3 -> base_u_i=2, base_u_j=2
    
    # B1 blocks (first 4 columns of B):
    # B11: j=0,1; k=0,1 -> base_v_j=0, base_v_k=0
    # B12: j=0,1; k=2,3 -> base_v_j=0, base_v_k=2
    # B21: j=2,3; k=0,1 -> base_v_j=2, base_v_k=0
    # B22: j=2,3; k=2,3 -> base_v_j=2, base_v_k=2
    
    # C blocks:
    # C11: i=0,1; k=0,1 -> base_w_i=0, base_w_k=0
    # C12: i=0,1; k=2,3 -> base_w_i=0, base_w_k=2
    # C21: i=2,3; k=0,1 -> base_w_i=2, base_w_k=0
    # C22: i=2,3; k=2,3 -> base_w_i=2, base_w_k=2
    
    # We need to call add_strassen_2x2 for each of the 7 Strassen products at the 4x4 level.
    # But add_strassen_2x2 expects a single 2x2*2x2 block.
    # The 4x4 Strassen products are sums of 2x2 blocks.
    # So we need to modify add_strassen_2x2 to handle sums of blocks.
    
    # Let's redefine add_strassen_2x2 to take lists of blocks.
    
    def add_strassen_2x2_blocks(u_blocks, v_blocks, w_blocks, sign):
        """
        u_blocks: list of (base_i, base_j, coeff)
        v_blocks: list of (base_j, base_k, coeff)
        w_blocks: list of (base_k, base_i, coeff)
        """
        # Expand each block into local terms
        u_terms = []
        for bi, bj, bc in u_blocks:
            for i in range(2):
                for j in range(2):
                    u_terms.append((i, j, bc))
        
        v_terms = []
        for bj, bk, bc in v_blocks:
            for i in range(2):
                for j in range(2):
                    v_terms.append((i, j, bc))
        
        w_terms = []
        for bk, bi, bc in w_blocks:
            for i in range(2):
                for j in range(2):
                    w_terms.append((i, j, bc))
        
        # Now we need to map local (i,j) to global indices.
        # This is complex because the blocks are at different positions.
        # We need to track the global offset for each term.
        
        # Let's change the approach: just add the terms directly.
        # For each u_block, v_block, w_block:
        #   For each local (i,j) in u_block:
        #     For each local (i',j') in v_block:
        #       If j == i':
        #         For each local (i'',j'') in w_block:
        #           add_term(global_u, global_v, global_w, coeff)
        
        for bi_u, bj_u, bc_u in u_blocks:
            for bi_v, bj_v, bc_v in v_blocks:
                # The j index of u must match the i index of v?
                # No, u is A[i,j], v is B[j,k]. So j_u == i_v.
                # In local coordinates, i_u, j_u and i_v, j_v.
                # We need j_u == i_v.
                for i_u in range(2):
                    for j_u in range(2):
                        for i_v in range(2):
                            for j_v in range(2):
                                if j_u == i_v:
                                    for bi_w, bj_w, bc_w in w_blocks:
                                        for i_w in range(2):
                                            for j_w in range(2):
                                                # Check consistency: j_v should be j_w? No.
                                                # The product is A[i_u,j_u] * B[i_v,j_v] * C[i_w,j_w].
                                                # We need j_u == i_v and j_v == j_w? No.
                                                # The tensor is T[i,j,k] = U[i,j] V[j,k] W[k,i].
                                                # So j_u == i_v and j_v == k_w? No, W is [k,i].
                                                # So j_v (k of B) must equal k of W, which is j_w? No, W index is [k,i].
                                                # Let's stick to the global indices.
                                                
                                                gu = (bi_u + i_u) * 4 + (bj_u + j_u)
                                                gv = (bi_v + i_v) * 9 + (bj_v + j_v)
                                                gw = (bj_w + j_w) * 4 + (bi_w + i_w)
                                                
                                                coeff = sign * bc_u * bc_v * bc_w
                                                add_term(gu, gv, gw, coeff)
    
    # Now define the 7 Strassen products at the 4x4 level.
    # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
    # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
    # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
    
    # M1 = (A11+A22)*(B11+B22)
    # U: A11(1), A22(1)
    # V: B11(1), B22(1)
    # W: C11(1), C22(1)
    add_strassen_2x2_blocks([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], 1)
    
    # M2 = (A21+A22)*B11
    # U: A21(1), A22(1)
    # V: B11(1)
    # W: C21(1), C22(1)
    add_strassen_2x2_blocks([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,1)], 1)
    
    # M3 = A11*(B12-B22)
    # U: A11(1)
    # V: B12(1), B22(-1)
    # W: C12(1), C22(1)
    add_strassen_2x2_blocks([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)], 1)
    
    # M4 = A22*(B21-B11)
    # U: A22(1)
    # V: B21(1), B11(-1)
    # W: C11(1), C21(1)
    add_strassen_2x2_blocks([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)], 1)
    
    # M5 = (A11+A12)*B22
    # U: A11(1), A12(1)
    # V: B22(1)
    # W: C11(-1), C12(1)
    add_strassen_2x2_blocks([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)], 1)
    
    # M6 = (A21-A11)*(B11+B12)
    # U: A21(1), A11(-1)
    # V: B11(1), B12(1)
    # W: C22(1)
    add_strassen_2x2_blocks([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)], 1)
    
    # M7 = (A12-A22)*(B21+B22)
    # U: A12(1), A22(-1)
    # V: B21(1), B22(1)
    # W: C11(-1), C22(1)
    add_strassen_2x2_blocks([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,-1), (2,2,1)], 1)
    
    # --- Part 2: Schoolbook for 4x4 * 4x5 ---
    # B2 is the last 5 columns of B: k=4,5,6,7,8.
    # For each i in 0..3, j in 0..3, k in 4..8:
    #   add_term(U[4i+j], V[9j+k], W[4k+i], 1)
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                add_term(4*i+j, 9*j+k, 4*k+i, 1)
    
    # Now build the matrices
    if not terms:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col in range(rank):
            k = col % p
            tmp = col // p
            j = tmp % m
            i = tmp // m
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    # Determine rank
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
