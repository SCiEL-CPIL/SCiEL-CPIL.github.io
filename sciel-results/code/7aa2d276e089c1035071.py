import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # U, V, W as numpy arrays for easy manipulation
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2 definitions
    # Outer terms (t=0..6)
    # X_t[i][j] is a list of (row, col, coeff) for A
    # Y_t[i][j] is a list of (row, col, coeff) for B
    # C_t[i][j] is a list of (row, col, coeff) for C (output)
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Helper to get global indices
    def a_idx(br, bc, lr, lc):
        return (br*2 + lr)*m + (bc*2 + lc)
        
    def b_idx(br, bc, lr, lc):
        return (br*2 + lr)*p + (bc*2 + lc)
        
    def w_idx(br, bc, lr, lc):
        # W[k*n+i, r] -> index is r*p + (k*n+i)? 
        # Prompt: W[k*n+i, r] where C = A*B, C[i,j] = sum_k A[i,k]B[k,j].
        # Output is C[i,j]. 
        # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
        # This usually means the tensor T_{i,j,k} = sum_t U_{i,k}^t V_{j,l}^t ... wait.
        # Standard tensor decomposition for matmul:
        # C[i,j] = sum_k A[i,k] B[k,j]
        # Tensor T_{i,j,k} = 1 if ... no, T is the multiplication tensor.
        # T_{i,j,k} is the coefficient of A_{i,k} B_{k,j} in C_{i,j}.
        # So T_{i,j,k} = 1 if the indices match? No.
        # The tensor is defined such that C = (A \otimes B) \cdot T.
        # Actually, the decomposition is T = sum_t u_t \otimes v_t \otimes w_t.
        # Where u_t is in R^{n*m} (flattened A), v_t is in R^{m*p} (flattened B), w_t is in R^{p*n} (flattened C).
        # The flattening convention matters.
        # Seed code:
        # U: (n*m, rank). Index: a_r * m + a_c.
        # V: (m*p, rank). Index: b_r * p + b_c.
        # W: (p*n, rank). Index: col * n + row. (w_idx returned col*n + row).
        # This implies W is flattened column-major for the C matrix?
        # C[row, col]. Index = col * n + row.
        # Let's stick to the seed's convention to ensure validity.
        return (bc*2 + lc) * n + (br*2 + lr)

    # Define the 7 outer terms
    # Each term is a tuple (X, Y, C_targets)
    # X, Y are 2x2 lists of lists of (br, bc, lr, lc, coeff)
    # C_targets is list of (br, bc, lr, lc, coeff)
    
    def block_combos(blocks):
        # blocks: list of (br, bc, coeff)
        # Returns list of (br, bc, lr, lc, coeff) for all 4 local positions
        res = []
        for br, bc, c in blocks:
            for lr in range(2):
                for lc in range(2):
                    res.append((br, bc, lr, lc, c))
        return res

    # M1: X = A00+A11, Y = B00+B11, C: C00+=1, C11+=1
    X0 = block_combos([(0,0,1), (1,1,1)])
    Y0 = block_combos([(0,0,1), (1,1,1)])
    C0 = [(0,0,0,0,1), (1,1,0,0,1)] # C00 and C11 top-left? No, C00 is block 00, C11 is block 11.
    # Wait, C00 is the top-left 2x2 block. C11 is bottom-right.
    # The "coeff" in C_targets applies to the whole block?
    # In Strassen, M1 contributes to C00 and C11.
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 adds 1 to C00 (all 4 entries) and 1 to C11 (all 4 entries).
    
    # Let's refine C_targets to be specific indices.
    # For a block (br, bc), the local indices are 0,0; 0,1; 1,0; 1,1.
    # If M1 contributes to C00, it contributes to C[0,0], C[0,1], C[1,0], C[1,1].
    
    def expand_block_to_indices(br, bc, coeff):
        return [(br, bc, lr, lc, coeff) for lr in range(2) for lc in range(2)]

    # Term 0 (M1): X=A00+A11, Y=B00+B11
    X_t = block_combos([(0,0,1), (1,1,1)])
    Y_t = block_combos([(0,0,1), (1,1,1)])
    C_t = expand_block_to_indices(0,0,1) + expand_block_to_indices(1,1,1)
    
    # Term 1 (M2): X=A10+A11, Y=B00
    X_t = block_combos([(1,0,1), (1,1,1)])
    Y_t = block_combos([(0,0,1)])
    C_t = expand_block_to_indices(1,0,1) + expand_block_to_indices(1,1,-1) # C10 += 1, C11 -= 1
    
    # Term 2 (M3): X=A00, Y=B01-B11
    X_t = block_combos([(0,0,1)])
    Y_t = block_combos([(0,1,1), (1,1,-1)])
    C_t = expand_block_to_indices(0,1,1) + expand_block_to_indices(1,1,1) # C01 += 1, C11 += 1
    
    # Term 3 (M4): X=A11, Y=B10-B00
    X_t = block_combos([(1,1,1)])
    Y_t = block_combos([(1,0,1), (0,0,-1)])
    C_t = expand_block_to_indices(0,0,1) + expand_block_to_indices(1,0,1) # C00 += 1, C10 += 1
    
    # Term 4 (M5): X=A00+A01, Y=B11
    X_t = block_combos([(0,0,1), (0,1,1)])
    Y_t = block_combos([(1,1,1)])
    C_t = expand_block_to_indices(0,0,-1) + expand_block_to_indices(0,1,1) # C00 -= 1, C01 += 1
    
    # Term 5 (M6): X=A10-A00, Y=B00+B01
    X_t = block_combos([(1,0,1), (0,0,-1)])
    Y_t = block_combos([(0,0,1), (0,1,1)])
    C_t = expand_block_to_indices(1,1,1) # C11 += 1
    
    # Term 6 (M7): X=A01-A11, Y=B10+B11
    X_t = block_combos([(0,1,1), (1,1,-1)])
    Y_t = block_combos([(1,0,1), (1,1,1)])
    C_t = expand_block_to_indices(0,0,1) # C00 += 1

    outer_terms = [
        (X_t, Y_t, C_t), # M1
        (X_t, Y_t, C_t), # M2 - Wait, I reused variables. Need to store them.
    ]
    
    # Let's store them properly in a list.
    terms = []
    
    # M1
    terms.append((
        block_combos([(0,0,1), (1,1,1)]),
        block_combos([(0,0,1), (1,1,1)]),
        expand_block_to_indices(0,0,1) + expand_block_to_indices(1,1,1)
    ))
    
    # M2
    terms.append((
        block_combos([(1,0,1), (1,1,1)]),
        block_combos([(0,0,1)]),
        expand_block_to_indices(1,0,1) + expand_block_to_indices(1,1,-1)
    ))
    
    # M3
    terms.append((
        block_combos([(0,0,1)]),
        block_combos([(0,1,1), (1,1,-1)]),
        expand_block_to_indices(0,1,1) + expand_block_to_indices(1,1,1)
    ))
    
    # M4
    terms.append((
        block_combos([(1,1,1)]),
        block_combos([(1,0,1), (0,0,-1)]),
        expand_block_to_indices(0,0,1) + expand_block_to_indices(1,0,1)
    ))
    
    # M5
    terms.append((
        block_combos([(0,0,1), (0,1,1)]),
        block_combos([(1,1,1)]),
        expand_block_to_indices(0,0,-1) + expand_block_to_indices(0,1,1)
    ))
    
    # M6
    terms.append((
        block_combos([(1,0,1), (0,0,-1)]),
        block_combos([(0,0,1), (0,1,1)]),
        expand_block_to_indices(1,1,1)
    ))
    
    # M7
    terms.append((
        block_combos([(0,1,1), (1,1,-1)]),
        block_combos([(1,0,1), (1,1,1)]),
        expand_block_to_indices(0,0,1)
    ))

    # Inner Strassen Terms
    # For each outer term (X, Y, C), we apply 7 inner terms.
    # X is a list of (br, bc, lr, lc, coeff).
    # We need to form linear combinations of these for the inner Strassen.
    
    # Inner M1: (X00+X11)(Y00+Y11)
    # X00 is the part of X where lr=0, lc=0.
    # X11 is the part of X where lr=1, lc=1.
    
    def get_local_part(combos, lr, lc):
        return [c for c in combos if c[2] == lr and c[3] == lc]
        
    def add_combos(c1, c2):
        # Combine two lists of (br, bc, lr, lc, coeff)
        # Sum coefficients for identical (br, bc, lr, lc)
        d = {}
        for br, bc, lr, lc, c in c1:
            key = (br, bc, lr, lc)
            d[key] = d.get(key, 0) + c
        for br, bc, lr, lc, c in c2:
            key = (br, bc, lr, lc)
            d[key] = d.get(key, 0) + c
        return [(br, bc, lr, lc, c) for (br, bc, lr, lc), c in d.items() if c != 0]

    def sub_combos(c1, c2):
        return add_combos(c1, [(br, bc, lr, lc, -c) for br, bc, lr, lc, c in c2])

    col = 0
    for t_idx in range(7):
        X_outer, Y_outer, C_outer = terms[t_idx]
        
        # Inner M1: (X00+X11)(Y00+Y11)
        X_in = add_combos(get_local_part(X_outer, 0, 0), get_local_part(X_outer, 1, 1))
        Y_in = add_combos(get_local_part(Y_outer, 0, 0), get_local_part(Y_outer, 1, 1))
        # C contribution: Z00 += 1, Z11 += 1
        # Z00 is local (0,0) of C_outer? No.
        # The inner Strassen computes Z = X*Y.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        # So for Inner M1, it contributes +1 to Z00 and +1 to Z11.
        # Z00 corresponds to local indices (0,0) in the block C.
        # Z11 corresponds to local indices (1,1) in the block C.
        
        # We need to map the local C indices (0,0) and (1,1) to the global C indices defined in C_outer.
        # C_outer is a list of (br, bc, lr, lc, coeff).
        # The "lr, lc" in C_outer are the local indices within the 2x2 block of C.
        # So for Inner M1, we add the coefficients from C_outer where local index is (0,0) and (1,1).
        
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (0, 0):
                C_in.append((br, bc, lr, lc, c))
            elif (lr, lc) == (1, 1):
                C_in.append((br, bc, lr, lc, c))
        
        # Fill U, V, W for this rank-1 term
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1
        
        # Inner M2: (X10+X11)Y00
        X_in = add_combos(get_local_part(X_outer, 1, 0), get_local_part(X_outer, 1, 1))
        Y_in = get_local_part(Y_outer, 0, 0)
        # Z10 += 1, Z11 -= 1
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (1, 0):
                C_in.append((br, bc, lr, lc, c))
            elif (lr, lc) == (1, 1):
                C_in.append((br, bc, lr, lc, -c))
                
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1
        
        # Inner M3: X00(Y01-Y11)
        X_in = get_local_part(X_outer, 0, 0)
        Y_in = sub_combos(get_local_part(Y_outer, 0, 1), get_local_part(Y_outer, 1, 1))
        # Z01 += 1, Z11 += 1
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (0, 1):
                C_in.append((br, bc, lr, lc, c))
            elif (lr, lc) == (1, 1):
                C_in.append((br, bc, lr, lc, c))
                
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1
        
        # Inner M4: X11(Y10-Y00)
        X_in = get_local_part(X_outer, 1, 1)
        Y_in = sub_combos(get_local_part(Y_outer, 1, 0), get_local_part(Y_outer, 0, 0))
        # Z00 += 1, Z10 += 1
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (0, 0):
                C_in.append((br, bc, lr, lc, c))
            elif (lr, lc) == (1, 0):
                C_in.append((br, bc, lr, lc, c))
                
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1
        
        # Inner M5: (X00+X01)Y11
        X_in = add_combos(get_local_part(X_outer, 0, 0), get_local_part(X_outer, 0, 1))
        Y_in = get_local_part(Y_outer, 1, 1)
        # Z00 -= 1, Z01 += 1
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (0, 0):
                C_in.append((br, bc, lr, lc, -c))
            elif (lr, lc) == (0, 1):
                C_in.append((br, bc, lr, lc, c))
                
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1
        
        # Inner M6: (X10-X00)(Y00+Y01)
        X_in = sub_combos(get_local_part(X_outer, 1, 0), get_local_part(X_outer, 0, 0))
        Y_in = add_combos(get_local_part(Y_outer, 0, 0), get_local_part(Y_outer, 0, 1))
        # Z11 += 1
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (1, 1):
                C_in.append((br, bc, lr, lc, c))
                
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1
        
        # Inner M7: (X01-X11)(Y10+Y11)
        X_in = sub_combos(get_local_part(X_outer, 0, 1), get_local_part(X_outer, 1, 1))
        Y_in = add_combos(get_local_part(Y_outer, 1, 0), get_local_part(Y_outer, 1, 1))
        # Z00 += 1
        C_in = []
        for br, bc, lr, lc, c in C_outer:
            if (lr, lc) == (0, 0):
                C_in.append((br, bc, lr, lc, c))
                
        for br, bc, lr, lc, c in X_in:
            U[a_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in Y_in:
            V[b_idx(br, bc, lr, lc), col] += c
        for br, bc, lr, lc, c in C_in:
            W[w_idx(br, bc, lr, lc), col] += c
        col += 1

    assert col == rank
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
