import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 terms definitions
    # Each term is a tuple (u_vec, v_vec, w_vec) for 2x2 matrices
    # Indices 0..3 for 2x2 matrices (row-major)
    # u: input A (2x2), v: input B (2x2), w: output C (2x2)
    
    def strassen_2x2_terms():
        terms = []
        # M0 = (a+d)(e+h)
        # a=0, b=1, c=2, d=3
        # e=0, f=1, g=2, h=3
        # u: a,d -> 1. v: e,h -> 1. w: p,s -> 1, -1? 
        # p = M0 - M3 + M4 + M6
        # q = M2 + M3
        # r = M1 + M4
        # s = M0 - M1 + M2 + M5
        
        # Term 0: M0 = (a+d)(e+h)
        u0 = [1, 0, 0, 1]
        v0 = [1, 0, 0, 1]
        w0 = [1, 0, 0, 1] # p=1, q=0, r=0, s=1
        terms.append((u0, v0, w0))
        
        # Term 1: M1 = (c+d)e
        u1 = [0, 0, 1, 1]
        v1 = [1, 0, 0, 0]
        w1 = [0, 0, 1, -1] # r=1, s=-1
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = a(f-h)
        u2 = [1, 0, 0, 0]
        v2 = [0, 1, 0, -1]
        w2 = [0, 1, 0, 1] # q=1, s=1
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = d(g-f)
        u3 = [0, 0, 0, 1]
        v3 = [0, -1, 1, 0]
        w3 = [-1, 1, 0, 0] # p=-1, q=1
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = (a+b)h
        u4 = [1, 1, 0, 0]
        v4 = [0, 0, 0, 1]
        w4 = [1, 0, 1, 0] # p=1, r=1
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (c-a)(e+f)
        u5 = [-1, 0, 1, 0]
        v5 = [1, 1, 0, 0]
        w5 = [0, 0, 0, 1] # s=1
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (b-d)(g+h)
        u6 = [0, 1, 0, -1]
        v6 = [0, 0, 1, 1]
        w6 = [1, 0, 0, 0] # p=1
        terms.append((u6, v6, w6))
        
        return terms

    base_terms = strassen_2x2_terms()
    
    # Block terms for 4x4
    # M_m = X_m Y_m
    # X_m, Y_m are 2x2 matrices of blocks
    # We need to map the 2x2 Strassen terms to the 4x4 indices
    
    # For each block term m, define the mapping from local 2x2 indices to global 4x4 indices
    # local index l in {0,1,2,3} corresponds to block element (r,c) where r=l//2, c=l%2
    # The block element is a sum of global elements
    
    # M0: X = A00 + A11, Y = B00 + B11
    # X_rc = A_rc + A_{r+2, c+2}
    # Mapping for X: local l -> global indices
    # l=0 (0,0): A[0,0], A[2,2]
    # l=1 (0,1): A[0,1], A[2,3]
    # l=2 (1,0): A[1,0], A[3,2]
    # l=3 (1,1): A[1,1], A[3,3]
    
    # We need to define for each of the 7 block terms:
    # 1. The mapping for X (u vector expansion)
    # 2. The mapping for Y (v vector expansion)
    # 3. The mapping for the result P (w vector expansion) and its position in C
    
    # Let's define the 7 block terms explicitly
    # term_idx: (X_map, Y_map, W_map, W_coeff)
    # X_map: list of 4 lists of (global_idx, coeff) for each local index
    # Actually, X is a 2x2 matrix. Each entry X_rc is a linear combo of A blocks.
    # For Strassen, X is a specific 2x2 matrix.
    
    # M0: X = [[A00+A11, 0], [0, A00+A11]]? No.
    # M0 = (A00+A11)(B00+B11).
    # X is the matrix (A00+A11). It is 2x2.
    # Y is the matrix (B00+B11). It is 2x2.
    # So X_rc = A_rc + A_{r+2,c+2}.
    
    # Let's create a helper to expand a local 2x2 vector to global 4x4 vector
    def expand_local_to_global(local_vec, block_type, sign=1):
        # block_type: 'sum' (A00+A11), 'diff' (A00-A11), etc.
        # This is getting complex. Let's hardcode the 7 block term structures.
        
        pass

    # Simpler approach:
    # Define the 7 block terms as pairs of (X_desc, Y_desc)
    # X_desc is a function or data structure describing the 2x2 matrix X in terms of A blocks.
    
    # Block Terms:
    # 0: X = A00+A11, Y = B00+B11
    # 1: X = A10+A11, Y = B00
    # 2: X = A00, Y = B01-B11
    # 3: X = A11, Y = B10-B00
    # 4: X = A00+A01, Y = B11
    # 5: X = A10-A00, Y = B00+B01
    # 6: X = A01-A11, Y = B10+B11
    
    # Output C:
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M3
    # C10 = M1 + M4
    # C11 = M0 - M1 + M2 + M5
    
    # For each block term m, we have 7 scalar terms.
    # Total 49 terms.
    
    # We will iterate m in 0..6, s in 0..6.
    # For each (m, s), we get u_local, v_local, w_local from base_terms[s].
    # We expand u_local to U_global, v_local to V_global, w_local to W_global.
    
    # Expansion logic:
    # u_local[k] is coefficient for X_{r,c} where r=k//2, c=k%2.
    # X_{r,c} is a sum of A elements.
    # We need to know for each m, what X_{r,c} is.
    
    # Let's define X_maps and Y_maps for the 7 block terms.
    # Each map is a 2x2 list of lists of (global_idx, coeff).
    
    def get_block_maps(m):
        # Returns X_map, Y_map, W_block_pos, W_coeff
        # X_map[r][c] = list of (global_idx, coeff)
        
        # Helper to create map for a block A_rc
        def block_map(r, c, coeff=1):
            # Returns list of (global_idx, coeff) for the 2x2 block at r,c
            # global_idx = (2*r + dr)*4 + (2*c + dc)
            res = []
            for dr in range(2):
                for dc in range(2):
                    gi = (2*r + dr)*4 + (2*c + dc)
                    res.append((gi, coeff))
            return res
        
        # M0: X = A00+A11
        if m == 0:
            X = [
                [block_map(0,0) + block_map(1,1)], # X00
                [block_map(0,0) + block_map(1,1)], # X01 ? No.
                # X is a 2x2 matrix.
                # X = A00 + A11.
                # X00 = A00_00 + A11_00
                # X01 = A00_01 + A11_01
                # X10 = A00_10 + A11_10
                # X11 = A00_11 + A11_11
                # So X_map[r][c] should be block_map(0,r,c?) No.
                # X is the matrix A00+A11.
                # So X_map[0][0] corresponds to element (0,0) of X, which is A00[0][0] + A11[0][0].
            ]
            # Let's build X_map properly
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            
            # X = A00 + A11
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = block_map(0, r, 1) + block_map(0, c, 1) # Wrong indices
                    # block_map takes block row, block col.
                    # Element (r,c) of X is A00[r][c] + A11[r][c].
                    # A00 is block (0,0). Element (r,c) is at global (2*0+r, 2*0+c).
                    # A11 is block (1,1). Element (r,c) is at global (2*1+r, 2*1+c).
                    X_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
            
            # Y = B00 + B11
            for r in range(2):
                for c in range(2):
                    Y_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
            
            # W contribution: M0 contributes to C00 and C11.
            # C00 is block (0,0). C11 is block (1,1).
            # w_local[k] is coeff for P_{r,c}.
            # P is M0.
            # M0 is added to C00 and C11.
            # So W_global will have contributions to blocks (0,0) and (1,1).
            # Coeff for C00 is +1. Coeff for C11 is +1.
            W_pos = [(0, 0, 1), (1, 1, 1)]
            
            return X_map, Y_map, W_pos

        # M1: X = A10+A11, Y = B00
        elif m == 1:
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    # X = A10 + A11
                    # X[r][c] = A10[r][c] + A11[r][c]
                    X_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
                    # Y = B00
                    Y_map[r][c] = [((2*0+r)*4 + (2*0+c), 1)]
            
            # M1 contributes to C10 (+1) and C11 (-1)
            W_pos = [(1, 0, 1), (1, 1, -1)]
            return X_map, Y_map, W_pos

        # M2: X = A00, Y = B01-B11
        elif m == 2:
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*0+r)*4 + (2*0+c), 1)]
                    # Y = B01 - B11
                    Y_map[r][c] = [((2*0+r)*4 + (2*1+c), 1), ((2*1+r)*4 + (2*1+c), -1)]
            
            # M2 contributes to C01 (+1) and C11 (+1)
            W_pos = [(0, 1, 1), (1, 1, 1)]
            return X_map, Y_map, W_pos

        # M3: X = A11, Y = B10-B00
        elif m == 3:
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*1+r)*4 + (2*1+c), 1)]
                    # Y = B10 - B00
                    Y_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*0+c), -1)]
            
            # M3 contributes to C00 (-1) and C01 (+1)
            W_pos = [(0, 0, -1), (0, 1, 1)]
            return X_map, Y_map, W_pos

        # M4: X = A00+A01, Y = B11
        elif m == 4:
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    # X = A00 + A01
                    X_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*1+c), 1)]
                    # Y = B11
                    Y_map[r][c] = [((2*1+r)*4 + (2*1+c), 1)]
            
            # M4 contributes to C00 (+1) and C10 (+1)
            W_pos = [(0, 0, 1), (1, 0, 1)]
            return X_map, Y_map, W_pos

        # M5: X = A10-A00, Y = B00+B01
        elif m == 5:
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    # X = A10 - A00
                    X_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*0+c), -1)]
                    # Y = B00 + B01
                    Y_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*1+c), 1)]
            
            # M5 contributes to C11 (+1)
            W_pos = [(1, 1, 1)]
            return X_map, Y_map, W_pos

        # M6: X = A01-A11, Y = B10+B11
        elif m == 6:
            X_map = [[None for _ in range(2)] for _ in range(2)]
            Y_map = [[None for _ in range(2)] for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    # X = A01 - A11
                    X_map[r][c] = [((2*0+r)*4 + (2*1+c), 1), ((2*1+r)*4 + (2*1+c), -1)]
                    # Y = B10 + B11
                    Y_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
            
            # M6 contributes to C00 (+1)
            W_pos = [(0, 0, 1)]
            return X_map, Y_map, W_pos

    for m in range(7):
        X_map, Y_map, W_pos = get_block_maps(m)
        
        for s in range(7):
            term_idx = m * 7 + s
            u_local, v_local, w_local = base_terms[s]
            
            # Expand U
            for k in range(4):
                if u_local[k] != 0:
                    r = k // 2
                    c = k % 2
                    coeff = u_local[k]
                    for gi, gc in X_map[r][c]:
                        U[gi][term_idx] += coeff * gc
                        
            # Expand V
            for k in range(4):
                if v_local[k] != 0:
                    r = k // 2
                    c = k % 2
                    coeff = v_local[k]
                    for gi, gc in Y_map[r][c]:
                        V[gi][term_idx] += coeff * gc
                        
            # Expand W
            for k in range(4):
                if w_local[k] != 0:
                    r = k // 2
                    c = k % 2
                    coeff = w_local[k]
                    # w_local[k] is coeff for P_{r,c}
                    # P is M_m
                    # M_m contributes to C at positions in W_pos
                    for br, bc, bc_coeff in W_pos:
                        # P_{r,c} is at global (2*br+r, 2*bc+c) in C
                        # W index is k*n + i = (2*bc+c)*4 + (2*br+r)
                        # Wait, W index is k*n + i.
                        # C_{i, k} is at W[k*4 + i].
                        # Here C row is 2*br+r, col is 2*bc+c.
                        # So i = 2*br+r, k = 2*bc+c.
                        # W index = (2*bc+c)*4 + (2*br+r)
                        wi = (2*bc + c) * 4 + (2*br + r)
                        W[wi][term_idx] += coeff * bc_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
