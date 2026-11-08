import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    # Recursive Strassen for 4x4 * 4x4 * 4x4 (first 4 columns)
    # This gives 49 terms
    
    def strassen_4x4_head():
        """Generate 49 terms for 4x4 * 4x4 * 4x4 (first 4 cols)"""
        # Outer Strassen 2x2 (where each "element" is a 2x2 block)
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        # C blocks: C00, C01, C10, C11
        
        outer = [
            # M1: (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # M2: (A10+A11)*B00 -> C10+=M2, C11-=M2
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            # M3: A00*(B01-B11) -> C01+=M3, C11+=M3
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            # M4: A11*(B10-B00) -> C00+=M4, C10+=M4
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # M5: (A00+A01)*B11 -> -C00, +C01
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # M6: (A10-A00)*(B00+B01) -> C11+=M6
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # M7: (A01-A11)*(B10+B11) -> C00+=M7
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        inner = [
            # M1: (x00+x11)*(y00+y11) -> z00+=M1, z11+=M1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # M2: (x10+x11)*y00 -> z10+=M2, z11-=M2
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            # M3: x00*(y01-y11) -> z01+=M3, z11+=M3
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            # M4: x11*(y10-y00) -> z00+=M4, z10+=M4
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # M5: (x00+x01)*y11 -> -z00, +z01
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # M6: (x10-x00)*(y00+y01) -> z11+=M6
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # M7: (x01-x11)*(y10+y11) -> z00+=M7
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        def get_global_idx(block_idx, local_idx, is_A):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            
            row = br * 2 + lr
            col = bc * 2 + lc
            
            if is_A:
                return row * m + col
            else:
                return row * p + col
        
        def get_w_idx(block_idx_c, local_idx_c):
            br = block_idx_c // 2
            bc = block_idx_c % 2
            lr = local_idx_c // 2
            lc = local_idx_c % 2
            
            i = br * 2 + lr
            k = bc * 2 + lc
            
            return k * n + i
        
        for oa, ob, oc in outer:
            for ia, ib, ic in inner:
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                
                for a_blk, a_coeff in oa:
                    for x_loc, x_coeff in ia:
                        idx = get_global_idx(a_blk, x_loc, True)
                        u[idx] += a_coeff * x_coeff
                
                for b_blk, b_coeff in ob:
                    for y_loc, y_coeff in ib:
                        idx = get_global_idx(b_blk, y_loc, False)
                        v[idx] += b_coeff * y_coeff
                
                for c_blk, c_coeff in oc:
                    for z_loc, z_coeff in ic:
                        idx = get_w_idx(c_blk, z_loc)
                        w[idx] += c_coeff * z_coeff
                
                add_term(u, v, w)

    # Tail: cols 4,5 using 4x4 * 4x2 Strassen (28 terms)
    # Col 6 using schoolbook (16 terms)
    
    def strassen_tail_2cols(k_off):
        """Generate 28 terms for 4x4 * 4x2 (cols k_off, k_off+1) using block Strassen"""
        # Split A into 4 blocks of 2x2, B into 4 blocks of 2x2 (but only 2 cols)
        # Actually, for 4x2, we split A into 2x2 blocks and B into 2x1 blocks
        # Let's use: split rows of A into 2 groups of 2, cols of B into 2 groups of 1
        # This gives a 2x2 * 2x2 * 2x2 structure where B has 2 cols
        
        # More simply: treat 4x2 as two separate 2x2 * 2x2 * 2x2 problems
        # Top half: A[0:2,:4] * B[:4, k_off:k_off+2] -> C[0:2, k_off:k_off+2]
        # Bottom half: A[2:4,:4] * B[:4, k_off:k_off+2] -> C[2:4, k_off:k_off+2]
        
        for i_b in range(2):  # row block of A
            for j_b in range(2):  # col block of B (each 1 col wide)
                i_off = i_b * 2
                j_off = j_b * 1
                k_off_local = k_off + j_b
                
                # 2x2 * 2x1 * 2x1 using 7 Strassen-like terms
                # Actually, for 2x2 * 2x1, we can use 4 schoolbook terms
                # But let's try to use fewer.
                # 2x2 * 2x1: C[i,j] = sum_k A[i,k]*B[k,j]
                # This is just 2*2*1 = 4 scalar multiplications per output element
                # Total 4 terms for 2 outputs? No, 4 terms total.
                
                # Let's just use schoolbook for 2x2 * 2x1
                for i in range(2):
                    for j in range(2):
                        u = [0.0] * (n * m)
                        v = [0.0] * (m * p)
                        w = [0.0] * (p * n)
                        u[(i_off + i) * m + j] = 1.0
                        v[j * p + k_off_local] = 1.0
                        w[k_off_local * n + (i_off + i)] = 1.0
                        add_term(u, v, w)
    
    def schoolbook_tail_1col(k_off):
        """Generate 16 terms for 4x4 * 4x1 (col k_off) using schoolbook"""
        for i in range(n):
            for j in range(m):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                u[i * m + j] = 1.0
                v[j * p + k_off] = 1.0
                w[k_off * n + i] = 1.0
                add_term(u, v, w)
    
    # Execute
    strassen_4x4_head()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    strassen_tail_2cols(4)  # cols 4, 5
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    schoolbook_tail_1col(6)  # col 6
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
