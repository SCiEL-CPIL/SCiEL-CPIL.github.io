import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define outer Strassen steps
    # X_k and Y_k are 2x2 matrices of linear forms in A and B blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We represent a linear form in A blocks as a dict {block_idx: coeff}
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    # Similarly for B.
    
    # Outer definitions
    # k=0: X=A0+A3, Y=B0+B3
    # k=1: X=A2+A3, Y=B0
    # k=2: X=A0, Y=B1-B3
    # k=3: X=A3, Y=B2-B0
    # k=4: X=A0+A1, Y=B3
    # k=5: X=A2-A0, Y=B0+B1
    # k=6: X=A1-A3, Y=B2+B3
    
    outer_X = [
        {0: 1, 3: 1}, # k=0
        {2: 1, 3: 1}, # k=1
        {0: 1},       # k=2
        {3: 1},       # k=3
        {0: 1, 1: 1}, # k=4
        {2: 1, 0: -1},# k=5
        {1: 1, 3: -1} # k=6
    ]
    
    outer_Y = [
        {0: 1, 3: 1}, # k=0
        {0: 1},       # k=1
        {1: 1, 3: -1},# k=2
        {2: 1, 0: -1},# k=3
        {3: 1},       # k=4
        {0: 1, 1: 1}, # k=5
        {2: 1, 3: 1}  # k=6
    ]
    
    # Inner Strassen steps
    # l=0: U=x00+x11, V=y00+y11
    # l=1: U=x10+x11, V=y00
    # l=2: U=x00, V=y01-y11
    # l=3: U=x11, V=y10-y00
    # l=4: U=x00+x01, V=y11
    # l=5: U=x10-x00, V=y00+y01
    # l=6: U=x01-x11, V=y10+y11
    
    # We represent x and y linear forms as dicts {rs_idx: coeff}
    # rs_idx: 0=x00, 1=x01, 2=x10, 3=x11
    
    inner_U = [
        {0: 1, 3: 1}, # l=0
        {2: 1, 3: 1}, # l=1
        {0: 1},       # l=2
        {3: 1},       # l=3
        {0: 1, 1: 1}, # l=4
        {2: 1, 0: -1},# l=5
        {1: 1, 3: -1} # l=6
    ]
    
    inner_V = [
        {0: 1, 3: 1}, # l=0
        {0: 1},       # l=1
        {1: 1, 3: -1},# l=2
        {2: 1, 0: -1},# l=3
        {3: 1},       # l=4
        {0: 1, 1: 1}, # l=5
        {2: 1, 3: 1}  # l=6
    ]
    
    # Inner reconstruction coefficients
    # Z00 = L0 + L3 - L4 + L6
    # Z01 = L2 + L4
    # Z10 = L1 + L3
    # Z11 = L0 - L1 + L2 + L5
    
    inner_recon = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # Z00
        {2: 1, 4: 1},              # Z01
        {1: 1, 3: 1},              # Z10
        {0: 1, 1: -1, 2: 1, 5: 1}  # Z11
    ]
    
    # Outer reconstruction coefficients
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    outer_recon = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},              # C01
        {1: 1, 3: 1},              # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # Helper to get A entry index from block and inner index
    # block ab -> i in Group a, j in Group b
    # inner rs -> offset r, s
    def a_idx(block, r, s):
        a = block // 2
        b = block % 2
        return (2*a + r) * 4 + (2*b + s)
        
    def b_idx(block, r, s):
        b = block // 2
        c = block % 2
        return (2*b + r) * 4 + (2*c + s)
        
    def c_idx(block, r, s):
        a = block // 2
        c = block % 2
        return (2*c + s) * 4 + (2*a + r) # Note: W[k*n+i, r] -> W is indexed by output index.
        # Output index is i*n + k? No, W shape is (p*n, rank).
        # Task says W[k*n+i, r].
        # Let's check the convention.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Usually C[i,k] = sum U[i,j] V[j,k].
        # Here U is (n*m, rank), V is (m*p, rank).
        # U index: i*m + j? No, U is (n*m, rank).
        # Let's assume standard row-major for A (n x m), B (m x p), C (n x p).
        # A[i, j], B[j, k], C[i, k].
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
        # So W index is k*n + i.
        
    # Let's re-verify indices.
    # n=4, m=4, p=4.
    # U: 16 x 49. Index i*m + j.
    # V: 16 x 49. Index j*p + k.
    # W: 16 x 49. Index k*n + i.
    
    # In my block notation:
    # A[i, j]. i is row, j is col.
    # B[j, k]. j is row, k is col.
    # C[i, k]. i is row, k is col.
    
    # My a_idx function:
    # block ab: a is row group, b is col group.
    # r, s are offsets.
    # i = 2a + r, j = 2b + s.
    # U index = i * 4 + j.
    
    # My b_idx function:
    # block bc: b is row group, c is col group.
    # r, s are offsets.
    # j = 2b + r, k = 2c + s.
    # V index = j * 4 + k.
    
    # My c_idx function:
    # block ac: a is row group, c is col group.
    # r, s are offsets.
    # i = 2a + r, k = 2c + s.
    # W index = k * 4 + i.
    
    # Let's rewrite the index functions clearly.
    
    def get_u_idx(a_block, r, s):
        a = a_block // 2
        b = a_block % 2
        i = 2 * a + r
        j = 2 * b + s
        return i * 4 + j
        
    def get_v_idx(b_block, r, s):
        b = b_block // 2
        c = b_block % 2
        j = 2 * b + r
        k = 2 * c + s
        return j * 4 + k
        
    def get_w_idx(c_block, r, s):
        a = c_block // 2
        c = c_block % 2
        i = 2 * a + r
        k = 2 * c + s
        return k * 4 + i
        
    for k in range(7):
        for l in range(7):
            c_idx_val = k * 7 + l
            
            # Get linear forms for U and V
            # u is linear form in x_rs
            # v is linear form in y_rs
            
            u_form = inner_U[l]
            v_form = inner_V[l]
            
            # Expand u_form in terms of A blocks
            # x_rs is a linear form in A blocks given by outer_X[k]
            # outer_X[k] is a dict {block: coeff}
            # But x_rs is a specific entry of X_k.
            # X_k is a 2x2 matrix.
            # How do we map x_rs to A blocks?
            # In Strassen, X_k is a 2x2 matrix of blocks.
            # So X_k[0,0] is a block, X_k[0,1] is a block, etc.
            # My outer_X[k] definition was for the whole matrix?
            # No, I defined outer_X[k] as a dict of blocks.
            # This implies X_k is a single block? No.
            
            # Let's re-read my outer definitions.
            # k=0: X=A0+A3. This means X_k = A0 + A3.
            # But A0 and A3 are 2x2 blocks.
            # So X_k is a 2x2 matrix where every entry is the same?
            # No, Strassen steps operate on the blocks as scalars.
            # So X_k is a 2x2 matrix of *scalars* (which are block values).
            # Wait, the Strassen algorithm for 2x2 matrices A, B:
            # M1 = (A11 + A22)(B11 + B22)
            # Here A11 is a scalar.
            # In the recursive case, A11 is a 2x2 matrix.
            # So M1 is a product of two 2x2 matrices.
            # The "scalar" multiplication is actually a matrix multiplication.
            # So X_k is not a 2x2 matrix of blocks.
            # X_k is a *single* 2x2 matrix (the sum/difference of blocks).
            
            # Let's re-evaluate.
            # For k=0: X_0 = A_00 + A_11.
            # A_00 is a 2x2 matrix. A_11 is a 2x2 matrix.
            # X_0 is a 2x2 matrix.
            # Y_0 = B_00 + B_11.
            # Y_0 is a 2x2 matrix.
            # M_0 = X_0 Y_0.
            
            # So, for a fixed k, X_k is a 2x2 matrix.
            # Its entries are linear combinations of A's entries.
            # Specifically, X_k[rs] is a linear form in A's entries.
            
            # How is X_k defined?
            # X_k = C_1 A_{ab} + C_2 A_{cd} + ...
            # This means X_k[rs] = C_1 A_{ab}[rs] + C_2 A_{cd}[rs] + ...
            
            # So my outer_X[k] dict {block: coeff} is correct.
            # It tells us which blocks are added to form X_k.
            
            # So, to get the linear form for x_rs (where rs is 0..3):
            # We sum over all blocks in outer_X[k].
            # For each block ab with coeff c:
            #   x_rs += c * A[ab][rs]
            
            # Similarly for y_rs.
            
            # So, for a given l:
            # u = sum_{rs} u_form[rs] * x_rs
            # u = sum_{rs} u_form[rs] * (sum_{ab} outer_X[k][ab] * A[ab][rs])
            # u = sum_{ab} (sum_{rs} u_form[rs] * outer_X[k][ab]) * A[ab][rs]
            
            # This means the coefficient of A[ab][rs] in u is:
            # coeff_u[ab][rs] = u_form[rs] * outer_X[k].get(ab, 0)
            
            # Similarly for v.
            
            # Now we can fill U and V.
            
            # U index for A[ab][rs] is get_u_idx(ab, r, s).
            # V index for B[bc][rs] is get_v_idx(bc, r, s).
            
            # Wait, the indices r, s in x_rs and y_rs correspond to the same positions in the 2x2 matrix.
            # Yes.
            
            # Let's iterate over all blocks and inner indices.
            
            for ab in range(4):
                coeff_x = outer_X[k].get(ab, 0)
                if coeff_x == 0: continue
                
                for rs in range(4):
                    coeff_u = coeff_x * inner_U[l].get(rs, 0)
                    if coeff_u != 0:
                        r = rs // 2
                        s = rs % 2
                        u_idx = get_u_idx(ab, r, s)
                        U[u_idx][c_idx_val] += coeff_u
                        
            for bc in range(4):
                coeff_y = outer_Y[k].get(bc, 0)
                if coeff_y == 0: continue
                
                for rs in range(4):
                    coeff_v = coeff_y * inner_V[l].get(rs, 0)
                    if coeff_v != 0:
                        r = rs // 2
                        s = rs % 2
                        v_idx = get_v_idx(bc, r, s)
                        V[v_idx][c_idx_val] += coeff_v
                        
            # Now fill W.
            # The product uv contributes to C[ac][rs] with coefficient:
            # TotalCoeff = outer_recon[ac].get(k, 0) * inner_recon[rs].get(l, 0)
            
            for ac in range(4):
                coeff_c = outer_recon[ac].get(k, 0)
                if coeff_c == 0: continue
                
                for rs in range(4):
                    coeff_inner = inner_recon[rs].get(l, 0)
                    if coeff_inner == 0: continue
                    
                    total_coeff = coeff_c * coeff_inner
                    if total_coeff != 0:
                        r = rs // 2
                        s = rs % 2
                        w_idx = get_w_idx(ac, r, s)
                        W[w_idx][c_idx_val] += total_coeff
                        
    return {"U": U, "V": V, "W": W}
