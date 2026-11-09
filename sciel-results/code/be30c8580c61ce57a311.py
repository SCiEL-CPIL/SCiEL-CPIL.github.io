import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS # 4, 4, 5
    
    # Rank 72 decomposition
    # 56 terms for 4x4x4 (Strassen on 2x2 blocks)
    # 16 terms for 4x4x1 (Schoolbook)
    
    rank = 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx, col] += val
        for idx, val in v_dict.items():
            V[idx, col] += val
        for idx, val in w_dict.items():
            W[idx, col] += val
        col += 1
        
    # Part 1: 4x4x4 using Strassen on 2x2 blocks
    # A blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B blocks: B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
    # Note: B indices are k*5+j.
    # B00: k in 0,1; j in 0,1 -> idx 0,1,5,6
    # B01: k in 0,1; j in 2,3 -> idx 2,3,7,8
    # B10: k in 2,3; j in 0,1 -> idx 10,11,15,16
    # B11: k in 2,3; j in 2,3 -> idx 12,13,17,18
    
    # We need to compute 8 block products.
    # Each block product is 2x2x2.
    # We will use the 7 Strassen products for each.
    
    # Strassen coefficients for 2x2x2:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We need to map these to the global indices.
    
    # Define block index ranges
    # A block (bi, bj) where bi, bj in {0,1}
    # A indices: i in [2*bi, 2*bi+1], k in [2*bj, 2*bj+1]
    # U index: i*4+k
    
    # B block (bk, bj) where bk, bj in {0,1}
    # B indices: k in [2*bk, 2*bk+1], j in [2*bj, 2*bj+1]
    # V index: k*5+j
    
    # C block (bi, bj)
    # W index: j*4+i
    
    # We iterate over the 8 block products needed for C.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each pair (A_block, B_block), we compute the product using Strassen.
    # But Strassen computes the full 2x2 product.
    # So for A00*B00, we use Strassen on A00 and B00.
    # The result is a 2x2 matrix which contributes to C00 (top-left block of C).
    
    # We will generate 56 terms.
    
    # Strassen terms definition
    # Each term is (u_coeff, v_coeff, w_coeff) for the 2x2 blocks.
    # u_coeff is 2x2, v_coeff is 2x2, w_coeff is 2x2?
    # No, the term is a scalar product of linear forms.
    # u is a vector of length 4 (flattened 2x2 A block).
    # v is a vector of length 4 (flattened 2x2 B block).
    # w is a vector of length 4 (flattened 2x2 C block).
    
    # The 7 Strassen forms:
    # 1. u = [1,0,0,1] (A00+A11), v = [1,0,0,1] (B00+B11), w = [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]? No.
    # The w vector indicates which entries of C this term contributes to.
    # Actually, the decomposition is $C = \sum M_c W_c$.
    # $M_c$ is a scalar.
    # $W_c$ is a 2x2 matrix.
    
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M_c, we have a 2x2 matrix of coefficients for C.
    
    # M1: C00+=1, C11+=1. W = [[1,0],[0,1]]
    # M2: C10+=1. W = [[0,0],[1,0]]
    # M3: C01+=1. W = [[0,1],[0,0]]
    # M4: C00+=1, C10+=1. W = [[1,0],[1,0]]
    # M5: C00-=1, C01+=1. W = [[-1,1],[0,0]]
    # M6: C11+=1. W = [[0,0],[0,1]]
    # M7: C00+=1. W = [[1,0],[0,0]]
    
    # And the U and V vectors for the 2x2 blocks.
    # M1: U = [1,0,0,1] (A00+A11), V = [1,0,0,1] (B00+B11)
    # M2: U = [0,0,1,1] (A10+A11), V = [1,0,0,0] (B00)
    # M3: U = [1,0,0,0] (A00), V = [0,1,0,-1] (B01-B11)
    # M4: U = [0,0,0,1] (A11), V = [0,0,1,-1] (B10-B00) -> Wait, B10 is [1,0] in local?
    # Let's use local indices 0,1,2,3 for 2x2.
    # 0:00, 1:01, 2:10, 3:11.
    # B00 is local 0. B01 is local 1. B10 is local 2. B11 is local 3.
    # M4: U = [0,0,0,1] (A11), V = [ -1, 0, 1, 0] (B10 - B00).
    # M5: U = [1,1,0,0] (A00+A01), V = [0,0,0,1] (B11).
    # M6: U = [-1,0,1,0] (A10-A00), V = [1,1,0,0] (B00+B01).
    # M7: U = [0,1,0,-1] (A01-A11), V = [0,0,1,1] (B10+B11).
    
    # Now we map local indices to global indices.
    
    strassen_u = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    strassen_v = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    strassen_w = [
        [[1,0],[0,1]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]]
    ]
    
    # Define the 8 block products
    # (a_block, b_block, c_block)
    # a_block: (bi, bj)
    # b_block: (bk, bj)
    # c_block: (bi, bj)
    
    block_products = [
        ((0,0), (0,0), (0,0)), # A00*B00 -> C00
        ((0,1), (1,0), (0,0)), # A01*B10 -> C00
        ((0,0), (0,1), (0,1)), # A00*B01 -> C01
        ((0,1), (1,1), (0,1)), # A01*B11 -> C01
        ((1,0), (0,0), (1,0)), # A10*B00 -> C10
        ((1,1), (1,0), (1,0)), # A11*B10 -> C10
        ((1,0), (0,1), (1,1)), # A10*B01 -> C11
        ((1,1), (1,1), (1,1)), # A11*B11 -> C11
    ]
    
    for (ab, bb, cb) in block_products:
        # ab = (ai, ak)
        # bb = (bk, bj)
        # cb = (ci, cj)
        
        # Global indices for A block
        # i in [2*ai, 2*ai+1], k in [2*ak, 2*ak+1]
        # Local index l = li*2 + lk
        # Global index = (2*ai+li)*4 + (2*ak+lk)
        
        # Global indices for B block
        # k in [2*bk, 2*bk+1], j in [2*bj, 2*bj+1]
        # Local index l = lk*2 + lj
        # Global index = (2*bk+lk)*5 + (2*bj+lj)
        
        # Global indices for C block
        # i in [2*ci, 2*ci+1], j in [2*cj, 2*cj+1]
        # Local index l = lj*2 + li ?
        # W index is j*4+i.
        # So for local (li, lj), global i = 2*ci+li, global j = 2*cj+lj.
        # Global index = (2*cj+lj)*4 + (2*ci+li).
        
        for c in range(7):
            u_vec = strassen_u[c]
            v_vec = strassen_v[c]
            w_mat = strassen_w[c]
            
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for li in range(2):
                for lk in range(2):
                    l = li*2 + lk
                    val = u_vec[l]
                    if val != 0:
                        gi = 2*ab[0] + li
                        gk = 2*ab[1] + lk
                        g_idx = gi*4 + gk
                        u_dict[g_idx] = u_dict.get(g_idx, 0) + val
            
            for lk in range(2):
                for lj in range(2):
                    l = lk*2 + lj
                    val = v_vec[l]
                    if val != 0:
                        gk = 2*bb[0] + lk
                        gj = 2*bb[1] + lj
                        g_idx = gk*5 + gj
                        v_dict[g_idx] = v_dict.get(g_idx, 0) + val
            
            for li in range(2):
                for lj in range(2):
                    l = lj*2 + li # Note: w_mat is indexed by [row, col] = [i, j]?
                    # Strassen w_mat is 2x2.
                    # w_mat[li, lj] corresponds to C[li, lj].
                    # In our W index, j is the first multiplier.
                    # So we need to be careful.
                    # C[li, lj] -> i=li, j=lj.
                    # W index = j*4 + i.
                    val = w_mat[li][lj]
                    if val != 0:
                        gi = 2*cb[0] + li
                        gj = 2*cb[1] + lj
                        g_idx = gj*4 + gi
                        w_dict[g_idx] = w_dict.get(g_idx, 0) + val
            
            add_term(u_dict, v_dict, w_dict)
            
    # Part 2: 4x4x1 (column 4)
    # 16 terms
    for i in range(4):
        for k in range(4):
            u_dict = {i*4+k: 1}
            v_dict = {k*5+4: 1}
            w_dict = {4*4+i: 1}
            add_term(u_dict, v_dict, w_dict)
            
    return {"U": U, "V": V, "W": W}
