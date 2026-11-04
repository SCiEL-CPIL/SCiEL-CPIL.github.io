import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # We will build a tensor decomposition using Strassen's algorithm for the 4x4x4 part
    # and schoolbook for the remaining 3 columns.
    # Total rank will be 49 + 48 = 97.
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Helper to add a term with linear combinations
    # u_coeffs: list of (i, j, coeff)
    # v_coeffs: list of (j, k, coeff)
    # w_coeffs: list of (k, i, coeff)
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        u = [0] * (n * m)
        for i, j, c in u_coeffs:
            u[i * m + j] += c
        v = [0] * (m * p)
        for j, k, c in v_coeffs:
            v[j * p + k] += c
        w = [0] * (p * n)
        for k, i, c in w_coeffs:
            w[k * n + i] += c
        terms.append((u, v, w))
    
    # Strassen's algorithm for 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose 4x4x4 using two levels of Strassen.
    # Level 1: 4x4x4 -> 7 multiplications of 2x2x2
    # Level 2: Each 2x2x2 -> 7 multiplications of 1x1x1
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Let's define the 7 Strassen terms for 2x2x2.
    # Each term is (u, v, w) where u, v, w are vectors of length 4 (for 2x2).
    # u[i*2+j], v[j*2+k], w[k*2+i]
    
    def strassen_2x2_terms():
        # Returns list of 7 tuples (u_coeffs, v_coeffs, w_coeffs)
        # where coeffs are lists of (idx, coeff) for the 4 elements
        s_terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        # u: A00=1, A11=1 -> [1, 0, 0, 1]
        # v: B00=1, B11=1 -> [1, 0, 0, 1]
        # w: C00 += 1, C11 += 1 -> [1, 0, 0, 1]
        s_terms.append(([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]))
        
        # M2 = (A10 + A11)B00
        # u: A10=1, A11=1 -> [0, 0, 1, 1]
        # v: B00=1 -> [1, 0, 0, 0]
        # w: C10 += 1 -> [0, 0, 1, 0]
        s_terms.append(([ (2,1), (3,1) ], [ (0,1) ], [ (2,1) ]))
        
        # M3 = A00(B10 - B11)
        # u: A00=1 -> [1, 0, 0, 0]
        # v: B10=1, B11=-1 -> [0, 0, 1, -1]
        # w: C01 += 1, C11 += 1 -> [0, 1, 0, 1]
        s_terms.append(([ (0,1) ], [ (2,1), (3,-1) ], [ (1,1), (3,1) ]))
        
        # M4 = A11(B01 - B00)
        # u: A11=1 -> [0, 0, 0, 1]
        # v: B01=1, B00=-1 -> [1, 1, 0, 0] -> wait, B01 is (0,1), B00 is (0,0)
        # v[0*2+1] = 1, v[0*2+0] = -1 -> [ -1, 1, 0, 0]
        # w: C00 += 1, C10 += 1 -> [1, 0, 1, 0]
        s_terms.append(([ (3,1) ], [ (0,-1), (1,1) ], [ (0,1), (2,1) ]))
        
        # M5 = (A00 + A01)B11
        # u: A00=1, A01=1 -> [1, 1, 0, 0]
        # v: B11=1 -> [0, 0, 0, 1]
        # w: C00 -= 1, C01 += 1 -> [-1, 1, 0, 0]
        s_terms.append(([ (0,1), (1,1) ], [ (3,1) ], [ (0,-1), (1,1) ]))
        
        # M6 = (A10 - A00)(B00 + B01)
        # u: A10=1, A00=-1 -> [ -1, 0, 1, 0]
        # v: B00=1, B01=1 -> [1, 1, 0, 0]
        # w: C11 += 1 -> [0, 0, 0, 1]
        s_terms.append(([ (0,-1), (2,1) ], [ (0,1), (1,1) ], [ (3,1) ]))
        
        # M7 = (A01 - A11)(B10 + B11)
        # u: A01=1, A11=-1 -> [0, 1, 0, -1]
        # v: B10=1, B11=1 -> [0, 0, 1, 1]
        # w: C00 += 1 -> [1, 0, 0, 0]
        s_terms.append(([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (0,1) ]))
        
        return s_terms
    
    strassen_2x2 = strassen_2x2_terms()
    
    # Now, for 4x4x4, we split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    # C = [[C00, C01], [C10, C11]] where each block is 2x2.
    
    # The 7 Strassen terms for the block level are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each Mi is a 2x2 matrix product.
    # We expand each Mi using the 7 terms from strassen_2x2.
    
    # We need to map the block indices to the full 4x4 indices.
    # Block (bi, bj) in A corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
    # So element (i, j) in block (bi, bj) is at global index (2*bi + i, 2*bj + j).
    
    # Let's define the 7 block-level terms.
    # Each term is (u_blocks, v_blocks, w_blocks) where each is a dict mapping (bi, bj) to coeff.
    # But it's easier to think in terms of the 7 strassen_2x2 terms applied to the blocks.
    
    # Actually, we can just iterate over the 7 block-level Strassen terms.
    # For each, we have a linear combination of blocks in A, B, and C.
    # Let's define them explicitly.
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # u_blocks: {(0,0): 1, (1,1): 1}
    # v_blocks: {(0,0): 1, (1,1): 1}
    # w_blocks: {(0,0): 1, (1,1): 1}  # C00 += M1, C11 += M1
    
    # Term 2: M2 = (A10 + A11)B00
    # u_blocks: {(1,0): 1, (1,1): 1}
    # v_blocks: {(0,0): 1}
    # w_blocks: {(1,0): 1}  # C10 += M2
    
    # Term 3: M3 = A00(B10 - B11)
    # u_blocks: {(0,0): 1}
    # v_blocks: {(1,0): 1, (1,1): -1}
    # w_blocks: {(0,1): 1, (1,1): 1}  # C01 += M3, C11 += M3
    
    # Term 4: M4 = A11(B01 - B00)
    # u_blocks: {(1,1): 1}
    # v_blocks: {(0,1): 1, (0,0): -1}
    # w_blocks: {(0,0): 1, (1,0): 1}  # C00 += M4, C10 += M4
    
    # Term 5: M5 = (A00 + A01)B11
    # u_blocks: {(0,0): 1, (0,1): 1}
    # v_blocks: {(1,1): 1}
    # w_blocks: {(0,0): -1, (0,1): 1}  # C00 -= M5, C01 += M5
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    # u_blocks: {(1,0): 1, (0,0): -1}
    # v_blocks: {(0,0): 1, (0,1): 1}
    # w_blocks: {(1,1): 1}  # C11 += M6
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    # u_blocks: {(0,1): 1, (1,1): -1}
    # v_blocks: {(1,0): 1, (1,1): 1}
    # w_blocks: {(0,0): 1}  # C00 += M7
    
    block_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Now, for each block term, we expand it into 7 scalar terms using strassen_2x2.
    # For a given block term (u_blocks, v_blocks, w_blocks):
    # We form u_global, v_global, w_global by combining the contributions from each block.
    # But wait, the Strassen algorithm for 2x2 matrices is applied to the blocks.
    # So for each of the 7 block-level terms, we need to compute the product of two 2x2 matrices.
    # This product is decomposed into 7 scalar multiplications.
    # So for each block term, we generate 7 scalar terms.
    
    # Let's do this carefully.
    # For a block term (u_blocks, v_blocks, w_blocks):
    # The u part is a linear combination of 2x2 blocks. Let's call it U_block.
    # The v part is a linear combination of 2x2 blocks. Let's call it V_block.
    # The w part is a linear combination of 2x2 blocks. Let's call it W_block.
    # We need to decompose U_block * V_block = W_block.
    # We use the 7 terms from strassen_2x2.
    # For each of the 7 terms (u_coeffs, v_coeffs, w_coeffs) in strassen_2x2:
    # We form a global u, v, w by:
    # u_global[i*4+j] = sum over (bi, bj) in u_blocks of u_coeffs[bi*2+i, bj*2+j] * u_blocks[(bi, bj)]
    # Wait, u_coeffs is for a 2x2 matrix, so indices are 0..3.
    # u_coeffs[0] corresponds to (0,0), u_coeffs[1] to (0,1), etc.
    # So for a block (bi, bj) with coeff c, and a local index (li, lj) with coeff cu:
    # The global index is (2*bi + li, 2*bj + lj).
    # The contribution to u_global is c * cu.
    
    for u_blocks, v_blocks, w_blocks in block_terms:
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            # u_coeffs is a list of (local_idx, coeff)
            # We need to expand this to global indices.
            u_global_coeffs = []
            for (li, lj), cu in u_coeffs:
                # Wait, u_coeffs in strassen_2x2 is a list of (idx, coeff) where idx is 0..3.
                # Let's convert idx to (li, lj).
                li = idx // 2
                lj = idx % 2
                # Now, for each block (bi, bj) in u_blocks:
                for (bi, bj), cb in u_blocks.items():
                    global_i = 2*bi + li
                    global_j = 2*bj + lj
                    u_global_coeffs.append((global_i, global_j, cu * cb))
            
            v_global_coeffs = []
            for (li, lj), cv in v_coeffs:
                li = idx // 2  # Bug: idx is not defined here
                # Fix:
                li = (li // 2) if isinstance(li, int) else 0
                # Actually, v_coeffs is a list of (idx, coeff).
                # Let's re-parse.
                pass
            
            # Let's redo this more carefully.
            u_global_coeffs = []
            for item in u_coeffs:
                idx_u, cu = item
                li = idx_u // 2
                lj = idx_u % 2
                for (bi, bj), cb in u_blocks.items():
                    global_i = 2*bi + li
                    global_j = 2*bj + lj
                    u_global_coeffs.append((global_i, global_j, cu * cb))
            
            v_global_coeffs = []
            for item in v_coeffs:
                idx_v, cv = item
                li = idx_v // 2
                lj = idx_v % 2
                for (bi, bj), cb in v_blocks.items():
                    global_i = 2*bi + li
                    global_j = 2*bj + lj
                    v_global_coeffs.append((global_i, global_j, cv * cb))
            
            w_global_coeffs = []
            for item in w_coeffs:
                idx_w, cw = item
                li = idx_w // 2
                lj = idx_w % 2
                for (bi, bj), cb in w_blocks.items():
                    global_i = 2*bi + li
                    global_j = 2*bj + lj
                    w_global_coeffs.append((global_i, global_j, cw * cb))
            
            add_term(u_global_coeffs, v_global_coeffs, w_global_coeffs)
    
    # Now, add schoolbook terms for the remaining 3 columns (k=4,5,6).
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}

