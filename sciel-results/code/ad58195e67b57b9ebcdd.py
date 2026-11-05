import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 120  # seconds
    
    # Rank 65 construction: 49 (Strassen 4x4) + 16 (Direct 4x1)
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, j, c in u_coeffs:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_coeffs:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_coeffs:
            if c != 0:
                W[k * n + i][col] += c
        col += 1

    # Strassen 2x2 terms: (A_coeffs, B_coeffs, C_coeffs)
    # Indices are (row, col)
    s2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(1,1,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,-1),(1,0,1)]),
        ([(1,1,1)], [(0,0,1),(0,1,-1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,0,-1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,-1),(1,0,1)])
    ]

    # Strassen 4x4 terms:
    # We apply the 7 super-terms. Each super-term involves products of 2x2 blocks.
    # Each 2x2 block product is expanded using the 7 inner terms.
    
    # Super-term definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B11
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Block mapping:
    # A00: rows 0,1 cols 0,1 -> global (0,0),(0,1),(1,0),(1,1)
    # A01: rows 0,1 cols 2,3 -> global (0,2),(0,3),(1,2),(1,3)
    # A10: rows 2,3 cols 0,1 -> global (2,0),(2,1),(3,0),(3,1)
    # A11: rows 2,3 cols 2,3 -> global (2,2),(2,3),(3,2),(3,3)
    
    def get_block_coeffs(block_id, local_coeffs):
        # block_id: 0=A00, 1=A01, 2=A10, 3=A11
        # local_coeffs: list of (r, c, coeff) for 2x2 block
        global_coeffs = []
        for r, c, coeff in local_coeffs:
            if coeff == 0: continue
            if block_id == 0:
                gr, gc = r, c
            elif block_id == 1:
                gr, gc = r, c + 2
            elif block_id == 2:
                gr, gc = r + 2, c
            elif block_id == 3:
                gr, gc = r + 2, c + 2
            global_coeffs.append((gr, gc, coeff))
        return global_coeffs

    def combine_blocks(block_list, local_coeffs_list):
        # Combines multiple blocks with their respective local coefficients
        # block_list: e.g., [0, 3] for A00 + A11
        # local_coeffs_list: e.g., [s2[0][0], s2[0][0]] (same local coeffs for both if it's a sum)
        # Actually, it's easier to just sum the global coeffs directly.
        # Let's just sum the lists.
        res = []
        for i, block_id in enumerate(block_list):
            res.extend(get_block_coeffs(block_id, local_coeffs_list[i]))
        # Merge and simplify? Not strictly necessary for correctness, just adds zeros or duplicates.
        # Duplicates are fine, they just add up in the matrix.
        return res

    # Define the 7 super-terms' A and B block structures
    # Each entry is (A_blocks, A_local_coeffs, B_blocks, B_local_coeffs, C_blocks, C_local_coeffs)
    # C_blocks: which 2x2 blocks of C get the result
    # Note: C is 4x4, so C00, C01, C10, C11 are 2x2 blocks.
    
    super_terms = [
        # M1
        {
            'A_blocks': [0, 3], 'A_locals': [s2[0][0], s2[0][0]], # A00 + A11 (using identity-like coeffs? No, we need the specific Strassen A coeffs)
            # Wait, the Strassen formula M1 = (A00+A11)(B00+B11) applies to the *blocks*.
            # So the "local" coefficients for the A part of M1 are just the identity for the 2x2 block?
            # No, the Strassen algorithm *is* the decomposition.
            # M1 is one of the 7 scalar multiplications in the 2x2 Strassen.
            # The inputs to M1 are (A00+A11) and (B00+B11).
            # These are 2x2 matrices.
            # To compute the product of these two 2x2 matrices, we use the 7-term Strassen again.
            # So for each of the 7 inner terms (s2[0] to s2[6]), we need to map the "A" part and "B" part to the global 4x4 indices.
            
            # Inner term t has A_coeffs A_t and B_coeffs B_t.
            # The global A matrix for this inner term is: sum_{b in A_blocks} (A_b * A_t)
            # The global B matrix for this inner term is: sum_{b in B_blocks} (B_b * B_t)
            # The global C matrix for this inner term is: sum_{c in C_blocks} (C_c * C_t)
        }
    ]
    
    # Let's restructure.
    # We have 7 outer terms.
    # For each outer term, we have specific A-blocks and B-blocks involved.
    # For each of those, we iterate through the 7 inner Strassen terms.
    
    outer_defs = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_id, sign)
        # B_blocks: list of (block_id, sign)
        # C_blocks: list of (block_id, sign)
        # The product is sum(A_block * sign) * sum(B_block * sign) -> sum(C_block * sign)
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        # M2 = (A10+A11)B11
        # C10 += M2, C11 -= M2
        {
            'A': [(2, 1), (3, 1)],
            'B': [(3, 1)],
            'C': [(2, 1), (3, -1)]
        },
        # M3 = A00(B10-B11)
        # C00 -= M3, C10 += M3
        {
            'A': [(0, 1)],
            'B': [(2, 1), (3, -1)],
            'C': [(0, -1), (2, 1)]
        },
        # M4 = A11(B00-B01)
        # C01 += M4, C11 -= M4
        {
            'A': [(3, 1)],
            'B': [(0, 1), (1, -1)],
            'C': [(1, 1), (3, -1)]
        },
        # M5 = (A00+A01)B11
        # C01 += M5, C10 -= M5
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(1, 1), (2, -1)]
        },
        # M6 = (A10-A00)(B00+B01)
        # C00 += M6, C11 += M6
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(0, 1), (3, 1)]
        },
        # M7 = (A01-A11)(B10+B11)
        # C01 -= M7, C10 += M7
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(1, -1), (2, 1)]
        }
    ]
    
    for outer in outer_defs:
        for inner_idx in range(7):
            a_local = s2[inner_idx][0]
            b_local = s2[inner_idx][1]
            c_local = s2[inner_idx][2]
            
            u_coeffs = []
            v_coeffs = []
            w_coeffs = []
            
            # Expand A
            for block_id, sign in outer['A']:
                for r, c, coeff in a_local:
                    if coeff == 0: continue
                    gr, gc = r, c
                    if block_id == 0: pass
                    elif block_id == 1: gc += 2
                    elif block_id == 2: gr += 2
                    elif block_id == 3: gr += 2; gc += 2
                    u_coeffs.append((gr, gc, sign * coeff))
            
            # Expand B
            for block_id, sign in outer['B']:
                for r, c, coeff in b_local:
                    if coeff == 0: continue
                    gr, gc = r, c
                    if block_id == 0: pass
                    elif block_id == 1: gc += 2
                    elif block_id == 2: gr += 2
                    elif block_id == 3: gr += 2; gc += 2
                    v_coeffs.append((gr, gc, sign * coeff))
            
            # Expand C
            for block_id, sign in outer['C']:
                for r, c, coeff in c_local:
                    if coeff == 0: continue
                    gr, gc = r, c
                    if block_id == 0: pass
                    elif block_id == 1: gc += 2
                    elif block_id == 2: gr += 2
                    elif block_id == 3: gr += 2; gc += 2
                    # W index is k*n + i. Here r is row i, c is col k.
                    w_coeffs.append((gc, gr, sign * coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # 16 terms for the 5th column (index 4 in B, index 4 in C)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            u_coeffs = [(i, j, 1)]
            v_coeffs = [(j, 4, 1)]
            w_coeffs = [(4, i, 1)]
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    return {"U": U, "V": V, "W": W}
