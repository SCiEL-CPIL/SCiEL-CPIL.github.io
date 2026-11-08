import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    Uses the Strassen algorithm applied to 2x2 blocks.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # U: (16, 49), V: (16, 49), W: (16, 49)
    # Indexing:
    # U[i*4 + j, r] corresponds to A[i, j]
    # V[j*4 + k, r] corresponds to B[j, k]
    # W[k*4 + i, r] corresponds to C[i, k] (Trace convention: W[k*n+i, r])
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Helper to get indices
    def u_idx(i, j): return i * 4 + j
    def v_idx(j, k): return j * 4 + k
    def w_idx(i, k): return k * 4 + i  # W[k*n+i] -> W[k*4+i]
    
    # Strassen 2x2 decomposition:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We apply this to 7 block products.
    # Let's define the 7 block products P0..P6.
    # Each Pk is a 2x2 matrix.
    # Pk[0,0], Pk[0,1], Pk[1,0], Pk[1,1]
    
    # We will iterate through the 7 block products.
    # For each, we generate 7 scalar multiplications (rank-1 terms).
    # Total 49 terms.
    
    # Define the linear combinations for A and B for each block product.
    # A blocks: A11, A12, A21, A22
    # B blocks: B11, B12, B21, B22
    
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Helper to add a linear combination to U/V
    # coeffs: list of (coeff, global_row, global_col)
    
    def add_to_U(coeffs, rank_idx):
        for coeff, i, j in coeffs:
            U[u_idx(i, j), rank_idx] += coeff
            
    def add_to_V(coeffs, rank_idx):
        for coeff, j, k in coeffs:
            V[v_idx(j, k), rank_idx] += coeff
            
    def add_to_W(coeffs, rank_idx):
        # coeffs: list of (coeff, global_i, global_k) for C[i, k]
        for coeff, i, k in coeffs:
            W[w_idx(i, k), rank_idx] += coeff

    # Define the 7 block-level terms (L_k, R_k) where P_k = L_k * R_k
    # L_k and R_k are 2x2 matrices of linear combinations of A/B elements.
    # We represent a 2x2 block linear combo as:
    # [[ (coeff, r, c), ... ], [ ... ]] for element (0,0)
    # This is getting complex. Let's use the explicit Strassen formulas for 4x4 directly.
    
    # Alternative: Just use the known rank-49 decomposition.
    # The Strassen algorithm for 4x4 is equivalent to 7 multiplications of 2x2 matrices.
    # Let's just hardcode the 49 terms based on the standard Strassen recursion.
    
    # To avoid complex nested logic, I will define the 7 block products explicitly.
    
    # Block A indices:
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # Block B indices:
    # B11: (0,0), (0,1), (1,0), (1,1)
    # B12: (0,2), (0,3), (1,2), (1,3)
    # B21: (2,0), (2,1), (3,0), (3,1)
    # B22: (2,2), (2,3), (3,2), (3,3)
    
    # Block C indices:
    # C11: (0,0), (0,1), (1,0), (1,1)
    # C12: (0,2), (0,3), (1,2), (1,3)
    # C21: (2,0), (2,1), (3,0), (3,1)
    # C22: (2,2), (2,3), (3,2), (3,3)
    
    # We define the 7 products P0..P6.
    # Each Pk is a 2x2 matrix.
    # Pk = Lk * Rk.
    # Lk and Rk are 2x2 matrices.
    
    # Let's define Lk and Rk as lists of 4 elements (00, 01, 10, 11).
    # Each element is a list of (coeff, i, j) for A or B.
    
    # P0 = (A11 + A22) * (B11 + B22)
    L0 = [
        [(1,0,0), (1,2,2), (1,0,1), (1,2,3), (1,1,0), (1,3,2), (1,1,1), (1,3,3)], # L0[0,0] = A11[0,0]+A22[0,0] etc? No.
        # A11 is 2x2. A11[0,0] is A[0,0]. A22[0,0] is A[2,2].
        # So L0[0,0] = A[0,0] + A[2,2]
        [(1,0,0), (1,2,2)],
        [(1,0,1), (1,2,3)], # L0[0,1] = A[0,1] + A[2,3]
        [(1,1,0), (1,3,2)], # L0[1,0] = A[1,0] + A[3,2]
        [(1,1,1), (1,3,3)]  # L0[1,1] = A[1,1] + A[3,3]
    ]
    R0 = [
        [(1,0,0), (1,2,2)], # R0[0,0] = B[0,0] + B[2,2]
        [(1,0,1), (1,2,3)], # R0[0,1] = B[0,1] + B[2,3]
        [(1,1,0), (1,3,2)], # R0[1,0] = B[1,0] + B[3,2]
        [(1,1,1), (1,3,3)]  # R0[1,1] = B[1,1] + B[3,3]
    ]
    # Contribution of P0 to C:
    # C11 += P0, C22 += P0
    # So P0[0,0] -> C[0,0], P0[0,1] -> C[0,1], etc.
    # And P0[0,0] -> C[2,2], etc.
    
    # This manual expansion is error-prone.
    # I will use the recursive structure but implemented cleanly.
    
    def strassen_2x2(L, R, out_U, out_V, out_W, rank_start):
        """
        L, R: 2x2 matrices of linear combos (list of (coeff, i, j))
        out_U, out_V, out_W: The global matrices
        rank_start: The starting rank index
        Returns the number of ranks used (7)
        """
        # Extract a,b,c,d and e,f,g,h
        a, b, c, d = L[0][0], L[0][1], L[1][0], L[1][1]
        e, f, g, h = R[0][0], R[0][1], R[1][0], R[1][1]
        
        def combine(combos1, combos2, sign=1):
            res = {}
            for coeff, i, j in combos1:
                key = (i, j)
                res[key] = res.get(key, 0) + coeff * sign
            for coeff, i, j in combos2:
                key = (i, j)
                res[key] = res.get(key, 0) + coeff * sign
            return list(res.items()) # List of ((i,j), coeff)
        
        def make_combos(pairs):
            # pairs: list of ((i,j), coeff)
            return [(coeff, i, j) for (i, j), coeff in pairs]
            
        # M1 = (a+d)(e+h)
        x1 = make_combos(combine(a, d))
        y1 = make_combos(combine(e, h))
        add_to_U(x1, rank_start + 0)
        add_to_V(y1, rank_start + 0)
        # M1 contributes to C11 (+1) and C22 (+1)
        # In the 2x2 result P, M1 adds to P[0,0] and P[1,1]
        # We need to map P's elements to global C.
        # This function doesn't know the global mapping.
        # So we must handle the mapping outside or pass it in.
        
        # This approach is getting complicated.
        # Let's just use the known fact that the current program works.
        # I will return the current program's logic but optimized/cleaned?
        # No, I must rewrite.
        
        pass

    # Given the complexity and the high likelihood that 49 is the minimum,
    # and the current program already achieves it,
    # I will provide a robust implementation of the Rank-49 decomposition.
    
    # I will use a pre-computed list of the 49 rank-1 terms.
    # This is the safest way to ensure correctness and speed.
    
    # The 49 terms are derived from Strassen.
    # I will generate them programmatically in a clean way.
    
    # Define the 7 block products
    # Each block product is defined by its L and R matrices (2x2)
    # and its contribution to the global C matrix.
    
    # Block A/B indices:
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A12: (0,2),(0,3),(1,2),(1,3)
    # A21: (2,0),(2,1),(3,0),(3,1)
    # A22: (2,2),(2,3),(3,2),(3,3)
    
    # B11: (0,0),(0,1),(1,0),(1,1)
    # B12: (0,2),(0,3),(1,2),(1,3)
    # B21: (2,0),(2,1),(3,0),(3,1)
    # B22: (2,2),(2,3),(3,2),(3,3)
    
    # C11: (0,0),(0,1),(1,0),(1,1)
    # C12: (0,2),(0,3),(1,2),(1,3)
    # C21: (2,0),(2,1),(3,0),(3,1)
    # C22: (2,2),(2,3),(3,2),(3,3)
    
    # Helper to create a 2x2 matrix of linear combos from block names
    def get_block_A(name):
        if name == '11': return [[(1,0,0)],[(1,0,1)],[(1,1,0)],[(1,1,1)]]
        if name == '12': return [[(1,0,2)],[(1,0,3)],[(1,1,2)],[(1,1,3)]]
        if name == '21': return [[(1,2,0)],[(1,2,1)],[(1,3,0)],[(1,3,1)]]
        if name == '22': return [[(1,2,2)],[(1,2,3)],[(1,3,2)],[(1,3,3)]]
        
    def get_block_B(name):
        if name == '11': return [[(1,0,0)],[(1,0,1)],[(1,1,0)],[(1,1,1)]]
        if name == '12': return [[(1,0,2)],[(1,0,3)],[(1,1,2)],[(1,1,3)]]
        if name == '21': return [[(1,2,0)],[(1,2,1)],[(1,3,0)],[(1,3,1)]]
        if name == '22': return [[(1,2,2)],[(1,2,3)],[(1,3,2)],[(1,3,3)]]
        
    def add_blocks(L1, L2, sign=1):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                for coeff, i, j in L1[r][c]:
                    res[r][c].append((coeff*sign, i, j))
                for coeff, i, j in L2[r][c]:
                    res[r][c].append((coeff*sign, i, j))
        return res
        
    def scale_block(L, s):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                for coeff, i, j in L[r][c]:
                    res[r][c].append((coeff*s, i, j))
        return res

    A11 = get_block_A('11')
    A12 = get_block_A('12')
    A21 = get_block_A('21')
    A22 = get_block_A('22')
    
    B11 = get_block_B('11')
    B12 = get_block_B('12')
    B21 = get_block_B('21')
    B22 = get_block_B('22')
    
    # Define the 7 products
    # P0 = (A11+A22)(B11+B22)
    L0 = add_blocks(A11, A22)
    R0 = add_blocks(B11, B22)
    # Contrib: C11 += P0, C22 += P0
    
    # P1 = (A21+A22)B11
    L1 = add_blocks(A21, A22)
    R1 = B11
    # Contrib: C21 += P1, C22 -= P1
    
    # P2 = A11(B12-B22)
    L2 = A11
    R2 = add_blocks(B12, B22, -1)
    # Contrib: C12 += P2, C22 += P2
    
    # P3 = A22(B21-B11)
    L3 = A22
    R3 = add_blocks(B21, B11, -1)
    # Contrib: C11 += P3, C21 += P3
    
    # P4 = (A11+A12)B22
    L4 = add_blocks(A11, A12)
    R4 = B22
    # Contrib: C11 -= P4, C12 += P4
    
    # P5 = (A21-A11)(B11+B12)
    L5 = add_blocks(A21, A11, -1)
    R5 = add_blocks(B11, B12)
    # Contrib: C22 += P5
    
    # P6 = (A12-A22)(B21+B22)
    L6 = add_blocks(A12, A22, -1)
    R6 = add_blocks(B21, B22)
    # Contrib: C11 += P6
    
    products = [
        (L0, R0, {('11', 1), ('22', 1)}),
        (L1, R1, {('21', 1), ('22', -1)}),
        (L2, R2, {('12', 1), ('22', 1)}),
        (L3, R3, {('11', 1), ('21', 1)}),
        (L4, R4, {('11', -1), ('12', 1)}),
        (L5, R5, {('22', 1)}),
        (L6, R6, {('11', 1)})
    ]
    
    rank_idx = 0
    for L, R, contrib in products:
        # Apply Strassen to L*R
        a, b, c, d = L[0][0], L[0][1], L[1][0], L[1][1]
        e, f, g, h = R[0][0], R[0][1], R[1][0], R[1][1]
        
        # M1 = (a+d)(e+h)
        x1 = add_blocks(a, d)
        y1 = add_blocks(e, h)
        add_to_U(make_combos(x1), rank_idx)
        add_to_V(make_combos(y1), rank_idx)
        # M1 -> P[0,0] (+1), P[1,1] (+1)
        # Map P[0,0] to global C based on contrib
        # This requires knowing which block P corresponds to.
        # The contrib dict tells us which C blocks get which sign.
        # P[0,0] is the top-left of the 2x2 product.
        # If contrib has '11', it means C11 gets this.
        # P[0,0] maps to C11[0,0], C11[0,1], C11[1,0], C11[1,1]?
        # No, P is a 2x2 matrix.
        # P[0,0] is a scalar (linear combo of products).
        # It contributes to the corresponding element of the C block.
        
        # We need to map the 2x2 result P to the global C.
        # For each element P[r, c], we add it to C_block[r, c].
        
        # Let's define the mapping of P[r,c] to global C indices.
        # This depends on which block product it is.
        # But the 'contrib' dict tells us the target C blocks.
        # However, the position (r,c) within the 2x2 block must be preserved.
        
        # Example: P0 contributes to C11 and C22.
        # P0[0,0] -> C11[0,0] and C22[0,0]
        # P0[0,1] -> C11[0,1] and C22[0,1]
        # etc.
        
        # So for each M_k, we determine its contribution to P[r,c].
        # Then for each target block in contrib, we add to that block's [r,c] element.
        
        # Strassen contributions to P:
        # M1: P[0,0] += 1, P[1,1] += 1
        # M2: P[1,0] += 1, P[1,1] -= 1
        # M3: P[0,1] += 1, P[1,1] += 1
        # M4: P[0,0] += 1, P[1,0] += 1
        # M5: P[0,0] -= 1, P[0,1] += 1
        # M6: P[1,1] += 1
        # M7: P[0,0] += 1
        
        sub_contribs = [
            { (0,0): 1, (1,1): 1 }, # M1
            { (1,0): 1, (1,1): -1 }, # M2
            { (0,1): 1, (1,1): 1 }, # M3
            { (0,0): 1, (1,0): 1 }, # M4
            { (0,0): -1, (0,1): 1 }, # M5
            { (1,1): 1 }, # M6
            { (0,0): 1 } # M7
        ]
        
        # M1
        add_to_U(make_combos(x1), rank_idx)
        add_to_V(make_combos(y1), rank_idx)
        for (pr, pc), coeff in sub_contribs[0].items():
            for block_name, sign in contrib.items():
                # Get the global indices for C_block[pr, pc]
                # block_name is '11', '12', etc.
                # C11[0,0] is C[0,0]
                # C11[0,1] is C[0,1]
                # C11[1,0] is C[1,0]
                # C11[1,1] is C[1,1]
                # C12[0,0] is C[0,2]
                # C21[0,0] is C[2,0]
                # C22[0,0] is C[2,2]
                
                if block_name == '11':
                    gi, gk = pr, pc
                elif block_name == '12':
                    gi, gk = pr, pc + 2
                elif block_name == '21':
                    gi, gk = pr + 2, pc
                elif block_name == '22':
                    gi, gk = pr + 2, pc + 2
                
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1
        
        # M2
        x2 = add_blocks(c, d)
        y2 = e
        add_to_U(make_combos(x2), rank_idx)
        add_to_V(make_combos(y2), rank_idx)
        for (pr, pc), coeff in sub_contribs[1].items():
            for block_name, sign in contrib.items():
                if block_name == '11': gi, gk = pr, pc
                elif block_name == '12': gi, gk = pr, pc + 2
                elif block_name == '21': gi, gk = pr + 2, pc
                elif block_name == '22': gi, gk = pr + 2, pc + 2
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1
        
        # M3
        x3 = a
        y3 = add_blocks(f, h, -1)
        add_to_U(make_combos(x3), rank_idx)
        add_to_V(make_combos(y3), rank_idx)
        for (pr, pc), coeff in sub_contribs[2].items():
            for block_name, sign in contrib.items():
                if block_name == '11': gi, gk = pr, pc
                elif block_name == '12': gi, gk = pr, pc + 2
                elif block_name == '21': gi, gk = pr + 2, pc
                elif block_name == '22': gi, gk = pr + 2, pc + 2
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1
        
        # M4
        x4 = d
        y4 = add_blocks(g, e, -1)
        add_to_U(make_combos(x4), rank_idx)
        add_to_V(make_combos(y4), rank_idx)
        for (pr, pc), coeff in sub_contribs[3].items():
            for block_name, sign in contrib.items():
                if block_name == '11': gi, gk = pr, pc
                elif block_name == '12': gi, gk = pr, pc + 2
                elif block_name == '21': gi, gk = pr + 2, pc
                elif block_name == '22': gi, gk = pr + 2, pc + 2
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1
        
        # M5
        x5 = add_blocks(a, b)
        y5 = h
        add_to_U(make_combos(x5), rank_idx)
        add_to_V(make_combos(y5), rank_idx)
        for (pr, pc), coeff in sub_contribs[4].items():
            for block_name, sign in contrib.items():
                if block_name == '11': gi, gk = pr, pc
                elif block_name == '12': gi, gk = pr, pc + 2
                elif block_name == '21': gi, gk = pr + 2, pc
                elif block_name == '22': gi, gk = pr + 2, pc + 2
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1
        
        # M6
        x6 = add_blocks(c, a, -1)
        y6 = add_blocks(e, f)
        add_to_U(make_combos(x6), rank_idx)
        add_to_V(make_combos(y6), rank_idx)
        for (pr, pc), coeff in sub_contribs[5].items():
            for block_name, sign in contrib.items():
                if block_name == '11': gi, gk = pr, pc
                elif block_name == '12': gi, gk = pr, pc + 2
                elif block_name == '21': gi, gk = pr + 2, pc
                elif block_name == '22': gi, gk = pr + 2, pc + 2
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1
        
        # M7
        x7 = add_blocks(b, d, -1)
        y7 = add_blocks(g, h)
        add_to_U(make_combos(x7), rank_idx)
        add_to_V(make_combos(y7), rank_idx)
        for (pr, pc), coeff in sub_contribs[6].items():
            for block_name, sign in contrib.items():
                if block_name == '11': gi, gk = pr, pc
                elif block_name == '12': gi, gk = pr, pc + 2
                elif block_name == '21': gi, gk = pr + 2, pc
                elif block_name == '22': gi, gk = pr + 2, pc + 2
                W[w_idx(gi, gk), rank_idx] += coeff * sign
        rank_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
