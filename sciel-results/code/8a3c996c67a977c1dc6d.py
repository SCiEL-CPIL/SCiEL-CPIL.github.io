import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using Strassen's algorithm applied to 2x2 block matrices.
    """
    deadline = time.time() + 120
    
    n = 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * n)]
    V = [[0.0] * rank for _ in range(n * n)]
    W = [[0.0] * rank for _ in range(n * n)]
    
    # Helper to add a term
    # u_dict: {flat_idx: coeff} for A
    # v_dict: {flat_idx: coeff} for B
    # w_dict: {flat_idx: coeff} for C (index k*n+i)
    def add_term(r, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    # Define the 7 Strassen block products for 4x4 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # We will define the 7 products M1..M7
    # Each Mk = Xk * Yk
    # Xk is a 2x2 matrix of linear forms in A
    # Yk is a 2x2 matrix of linear forms in B
    
    # Block indexing:
    # A00: A[0,0], A[0,1], A[1,0], A[1,1] -> flat 0,1,4,5
    # A01: A[0,2], A[0,3], A[1,2], A[1,3] -> flat 2,3,6,7
    # A10: A[2,0], A[2,1], A[3,0], A[3,1] -> flat 8,9,12,13
    # A11: A[2,2], A[2,3], A[3,2], A[3,3] -> flat 10,11,14,15
    
    # Let's represent a 2x2 matrix of linear forms as a list of 4 dicts
    # [0,0], [0,1], [1,0], [1,1]
    
    def make_block(block_r, block_c, sign=1.0):
        """Create dict for a 2x2 block at (block_r, block_c) of 4x4 matrix"""
        d = {}
        for i in range(2):
            for j in range(2):
                row = block_r * 2 + i
                col = block_c * 2 + j
                idx = row * 4 + col
                d[idx] = sign
        return d

    def add_blocks(d1, d2, sign2=1.0):
        """Add sign2*d2 to d1"""
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0.0) + sign2 * v
        return res

    # Define X and Y for each of the 7 Strassen products
    # S1: (A00+A11)(B00+B11)
    X1 = [
        add_blocks(make_block(0,0), make_block(1,1)), # 0,0
        add_blocks(make_block(0,0), make_block(1,1), 0), # 0,1 - wait, A00 is top-left.
        # A00 is rows 0-1, cols 0-1.
        # A11 is rows 2-3, cols 2-3.
        # X1[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
        # X1[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
        # X1[1,0] = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
        # X1[1,1] = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
    ]
    
    # Let's write a helper to get the linear form for Xk[r,s]
    # Xk is sum of blocks.
    # Block (br, bc) has entries A[2*br+i, 2*bc+j]
    
    def get_X_entry(br, bc, r, s, sign=1.0):
        """Get linear form for entry (r,s) of block (br,bc)"""
        row = 2*br + r
        col = 2*bc + s
        return {row*4+col: sign}

    # S1: X = A00 + A11
    # X[r,s] = A[2*0+r, 2*0+s] + A[2*1+r, 2*1+s]
    X1 = [
        [get_X_entry(0,0,0,0,1) | get_X_entry(1,1,0,0,1)],
        [get_X_entry(0,0,0,1,1) | get_X_entry(1,1,0,1,1)],
        [get_X_entry(0,0,1,0,1) | get_X_entry(1,1,1,0,1)],
        [get_X_entry(0,0,1,1,1) | get_X_entry(1,1,1,1,1)]
    ]
    # Y1 = B00 + B11
    Y1 = [
        [get_X_entry(0,0,0,0,1) | get_X_entry(1,1,0,0,1)],
        [get_X_entry(0,0,0,1,1) | get_X_entry(1,1,0,1,1)],
        [get_X_entry(0,0,1,0,1) | get_X_entry(1,1,1,0,1)],
        [get_X_entry(0,0,1,1,1) | get_X_entry(1,1,1,1,1)]
    ]
    # C contributions for S1: C00 (+1), C11 (+1)
    # C00 is block (0,0), C11 is block (1,1)
    # We need to map the 2x2 result Z to the 4x4 C.
    # Z[0,0] -> C[0,0], Z[0,1] -> C[0,1], Z[1,0] -> C[1,0], Z[1,1] -> C[1,1]
    # So if S1 contributes +1 to C00, then Z[0,0] contributes to C[0,0] with +1.
    # If S1 contributes +1 to C11, then Z[1,1] contributes to C[3,3] with +1?
    # No, C11 is the block at (1,1). So Z[0,0] of C11 block is C[2,2].
    # The Strassen formula is C00 = M1 + ...
    # So M1 is the full 2x2 block.
    # M1[0,0] is C[0,0]. M1[0,1] is C[0,1]. M1[1,0] is C[1,0]. M1[1,1] is C[1,1].
    # So for S1, the output block is C00 and C11.
    # Wait, M1 is added to C00 and C11.
    # So M1[0,0] -> C[0,0], M1[0,1] -> C[0,1]... NO.
    # C00 is the top-left 2x2 block.
    # So M1[0,0] is C[0,0], M1[0,1] is C[0,1] (which is in C01 block? No, C00 is 2x2).
    # C00 contains C[0,0], C[0,1], C[1,0], C[1,1].
    # So M1[0,0] -> C[0,0], M1[0,1] -> C[0,1], M1[1,0] -> C[1,0], M1[1,1] -> C[1,1].
    # And M1 is also added to C11.
    # C11 contains C[2,2], C[2,3], C[3,2], C[3,3].
    # So M1[0,0] -> C[2,2], M1[0,1] -> C[2,3], M1[1,0] -> C[3,2], M1[1,1] -> C[3,3].
    
    # This interpretation is correct.
    
    # Let's define the 7 products generally.
    # Product k has Xk, Yk, and a list of (block_pos, sign) for C.
    
    products = []
    
    # S1
    X = [[get_X_entry(0,0,r,s,1) | get_X_entry(1,1,r,s,1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(0,0,r,s,1) | get_X_entry(1,1,r,s,1) for s in range(2)] for r in range(2)]
    C_map = { (0,0): 1, (1,1): 1 }
    products.append((X, Y, C_map))
    
    # S2: (A10+A11)B00
    X = [[get_X_entry(1,0,r,s,1) | get_X_entry(1,1,r,s,1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(0,0,r,s,1) for s in range(2)] for r in range(2)]
    C_map = { (1,0): 1, (1,1): -1 }
    products.append((X, Y, C_map))
    
    # S3: A00(B01-B11)
    X = [[get_X_entry(0,0,r,s,1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(0,1,r,s,1) | get_X_entry(1,1,r,s,-1) for s in range(2)] for r in range(2)]
    C_map = { (0,1): 1, (1,1): 1 }
    products.append((X, Y, C_map))
    
    # S4: A11(B10-B00)
    X = [[get_X_entry(1,1,r,s,1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(1,0,r,s,1) | get_X_entry(0,0,r,s,-1) for s in range(2)] for r in range(2)]
    C_map = { (0,0): 1, (1,0): 1 }
    products.append((X, Y, C_map))
    
    # S5: (A00+A01)B11
    X = [[get_X_entry(0,0,r,s,1) | get_X_entry(0,1,r,s,1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(1,1,r,s,1) for s in range(2)] for r in range(2)]
    C_map = { (0,0): -1, (0,1): 1 }
    products.append((X, Y, C_map))
    
    # S6: (A10-A00)(B00+B01)
    X = [[get_X_entry(1,0,r,s,1) | get_X_entry(0,0,r,s,-1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(0,0,r,s,1) | get_X_entry(0,1,r,s,1) for s in range(2)] for r in range(2)]
    C_map = { (1,1): 1 }
    products.append((X, Y, C_map))
    
    # S7: (A01-A11)(B10+B11)
    X = [[get_X_entry(0,1,r,s,1) | get_X_entry(1,1,r,s,-1) for s in range(2)] for r in range(2)]
    Y = [[get_X_entry(1,0,r,s,1) | get_X_entry(1,1,r,s,1) for s in range(2)] for r in range(2)]
    C_map = { (0,0): 1 }
    products.append((X, Y, C_map))
    
    # Now apply 2x2 Strassen to each of the 7 products.
    # 2x2 Strassen terms:
    # z1 = (x00+x11)(y00+y11) -> Z00:1, Z11:1
    # z2 = (x10+x11)y00       -> Z10:1, Z11:-1
    # z3 = x00(y01-y11)       -> Z01:1, Z11:1
    # z4 = x11(y10-y00)       -> Z00:1, Z10:1
    # z5 = (x00+x01)y11       -> Z00:-1, Z01:1
    # z6 = (x10-x00)(y00+y01) -> Z11:1
    # z7 = (x01-x11)(y10+y11) -> Z00:1
    
    strassen_terms = [
        { 'X': [(0,0,1), (1,1,1)], 'Y': [(0,0,1), (1,1,1)], 'Z': {(0,0):1, (1,1):1} },
        { 'X': [(1,0,1), (1,1,1)], 'Y': [(0,0,1)], 'Z': {(1,0):1, (1,1):-1} },
        { 'X': [(0,0,1)], 'Y': [(0,1,1), (1,1,-1)], 'Z': {(0,1):1, (1,1):1} },
        { 'X': [(1,1,1)], 'Y': [(1,0,1), (0,0,-1)], 'Z': {(0,0):1, (1,0):1} },
        { 'X': [(0,0,1), (0,1,1)], 'Y': [(1,1,1)], 'Z': {(0,0):-1, (0,1):1} },
        { 'X': [(1,0,1), (0,0,-1)], 'Y': [(0,0,1), (0,1,1)], 'Z': {(1,1):1} },
        { 'X': [(0,1,1), (1,1,-1)], 'Y': [(1,0,1), (1,1,1)], 'Z': {(0,0):1} }
    ]
    
    r = 0
    for Xk, Yk, Ck in products:
        for st in strassen_terms:
            # Construct U, V, W for this term
            U_dict = {}
            V_dict = {}
            W_dict = {}
            
            # U from X
            for (rr, ss, coeff) in st['X']:
                # Xk[rr][ss] is a dict
                for idx, val in Xk[rr][ss].items():
                    U_dict[idx] = U_dict.get(idx, 0.0) + coeff * val
            
            # V from Y
            for (rr, ss, coeff) in st['Y']:
                for idx, val in Yk[rr][ss].items():
                    V_dict[idx] = V_dict.get(idx, 0.0) + coeff * val
            
            # W from Z contributions mapped to C
            for (zr, zs), zcoeff in st['Z'].items():
                for (cb_r, cb_c), ccoeff in Ck.items():
                    # Z[zr, zs] is in block (cb_r, cb_c)
                    # So it corresponds to C[2*cb_r + zr, 2*cb_c + zs]
                    row = 2*cb_r + zr
                    col = 2*cb_c + zs
                    c_idx = col * 4 + row # W index is k*n+i -> col*4+row
                    W_dict[c_idx] = W_dict.get(c_idx, 0.0) + zcoeff * ccoeff
            
            add_term(r, U_dict, V_dict, W_dict)
            r += 1
            
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    return {"U": U, "V": V, "W": W}
