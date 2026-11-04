import time
import sys

DIMENSIONS = [4, 4, 5]

def _add_term(U, V, W, r, u_idx, v_idx, w_idx, val=1):
    U[u_idx][r] += val
    V[v_idx][r] += val
    W[w_idx][r] += val

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    total_rank = 65
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Helper to get global index for A block (br, bc) and entry (r, c)
    def a_idx(br, bc, r, c):
        return (br * 2 + r) * m + (bc * 2 + c)

    # Helper to get global index for B block (br, bc) and entry (r, c)
    def b_idx(br, bc, r, c):
        return (br * 2 + r) * p + (bc * 2 + c)

    # Helper to get global index for C block (cr, cc) and entry (r, c)
    def c_idx(cr, cc, r, c):
        # W index: k * n + i
        # C is p x n
        # Block (cr, cc) is 2x2
        # Entry (r, c) in block -> row cr*2+r, col cc*2+c
        # W index: col * n + row
        return (cc * 2 + c) * n + (cr * 2 + r)

    # Define the 7 outer Strassen steps
    # Each step is a tuple (A_comb, B_comb) where A_comb is a 2x2 list of (block_row, block_col, coeff)
    # Actually, simpler: A_comb is a 2x2 matrix of coefficients for the 4 blocks.
    # A_comb[r][c] is a list of (block_idx, coeff)
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # A_comb: [[(0,1), (1,1)], [(2,1), (3,1)]]? No.
    # A_comb is 2x2.
    # A_comb[0][0] = A00. A_comb[1][1] = A11.
    # So A_comb[0][0] has block 0 coeff 1. A_comb[1][1] has block 3 coeff 1.
    
    outer_steps = [
        # 1
        ([[[(0, 1)], [(1, 1)]], [[(2, 1)], [(3, 1)]]], [[[(0, 1)], [(1, 1)]], [[(2, 1)], [(3, 1)]]]),
        # 2: (A10+A11)B00
        ([[[(0, 0)], [(1, 0)]], [[(2, 1), (3, 1)], [(4, 0), (5, 0)]]], [[[(0, 1)], [(1, 0)]], [[(2, 0)], [(3, 0)]]]),
        # 3: A00(B01-B11)
        ([[[(0, 1)], [(1, 0)]], [[(2, 0)], [(3, 0)]]], [[[(0, 0)], [(1, 1)]], [[(2, 0)], [(3, -1)]]]),
        # 4: A11(B10-B00)
        ([[[(0, 0)], [(1, 0)]], [[(2, 0)], [(3, 1)]]], [[[(0, -1)], [(1, 0)]], [[(2, 1)], [(3, 0)]]]),
        # 5: (A00+A01)B11
        ([[[(0, 1), (1, 1)], [(2, 0)]], [[(3, 0)], [(4, 0)]]], [[[(0, 0)], [(1, 0)]], [[(2, 0)], [(3, 1)]]]),
        # 6: (A10-A00)(B00+B01)
        ([[[(0, -1)], [(1, 0)]], [[(2, 1)], [(3, 0)]]], [[[(0, 1)], [(1, 1)]], [[(2, 0)], [(3, 0)]]]),
        # 7: (A01-A11)(B10+B11)
        ([[[(0, 0)], [(1, 1)]], [[(2, 0)], [(3, -1)]]], [[[(0, 0)], [(1, 0)]], [[(2, 1)], [(3, 1)]]]),
    ]
    
    # Correcting the A_comb/B_comb structure for clarity
    # A_comb[r][c] is a list of (block_idx, coeff)
    # block_idx 0: A00, 1: A01, 2: A10, 3: A11
    
    outer_steps = [
        # 1
        ([[[(0, 1)], []], [[], [(3, 1)]]], [[[(0, 1)], []], [[], [(3, 1)]]]),
        # 2
        ([[[], []], [[(2, 1), (3, 1)], []]], [[[(0, 1)], []], [[], []]]),
        # 3
        ([[[(0, 1)], []], [[], []]], [[[], [(1, 1)]], [[], [(3, -1)]]]),
        # 4
        ([[[], []], [[], [(3, 1)]]], [[[(0, -1)], []], [[[(2, 1)], []]]]),
        # 5
        ([[[(0, 1), (1, 1)], []], [[], []]], [[[], []], [[], [(3, 1)]]]),
        # 6
        ([[[(0, -1)], []], [[(2, 1)], []]], [[[(0, 1)], [(1, 1)]], [[], []]]),
        # 7
        ([[[], [(1, 1)]], [[], [(3, -1)]]], [[[], []], [[(2, 1)], [(3, 1)]]]),
    ]

    col = 0
    for k in range(7):
        A_comb, B_comb = outer_steps[k]
        
        # Inner Strassen: 7 terms
        # For each inner term, we need to determine the linear combination of A_comb entries and B_comb entries.
        # Then map to global A, B indices.
        
        # Inner Strassen terms (u_i, v_i) for 2x2 multiplication
        # u_i is a 2x2 matrix of coeffs for X (A_comb)
        # v_i is a 2x2 matrix of coeffs for Y (B_comb)
        
        inner_terms = [
            # 1: (X00+X11)(Y00+Y11)
            ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
            # 2: (X10+X11)Y00
            ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),
            # 3: X00(Y01-Y11)
            ([[1, 0], [0, 0]], [[0, 1], [0, -1]]),
            # 4: X11(Y10-Y00)
            ([[0, 0], [0, 1]], [[-1, 0], [1, 0]]),
            # 5: (X00+X01)Y11
            ([[1, 1], [0, 0]], [[0, 0], [0, 1]]),
            # 6: (X10-X00)(Y00+Y01)
            ([[ -1, 0], [1, 0]], [[1, 1], [0, 0]]),
            # 7: (X01-X11)(Y10+Y11)
            ([[0, 1], [0, -1]], [[0, 0], [1, 1]]),
        ]
        
        for i in range(7):
            u_coeff, v_coeff = inner_terms[i]
            
            # Build U and V for this term
            # U has indices for A entries
            # V has indices for B entries
            
            # For each entry (r, c) of the 2x2 result M_k
            # M_k[rc] = sum_{w} A_comb[r][w] B_comb[w][c]
            # This is handled by the inner Strassen decomposition of the 2x2 product.
            # The inner Strassen produces 7 terms that sum to the 4 entries of M_k.
            
            # We need to add 1 global term for this inner term i.
            # The U part is the linear combination of A entries defined by u_coeff and A_comb.
            # The V part is the linear combination of B entries defined by v_coeff and B_comb.
            # The W part is the contribution to C entries.
            
            # Calculate U indices and coeffs
            u_entries = {}
            for r in range(2):
                for c in range(2):
                    coeff = u_coeff[r][c]
                    if coeff != 0:
                        # A_comb[r][c] is a list of (block_idx, block_coeff)
                        for block_idx, block_coeff in A_comb[r][c]:
                            total_coeff = coeff * block_coeff
                            if total_coeff != 0:
                                # Map block_idx to 4 A entries
                                br = block_idx // 2
                                bc = block_idx % 2
                                for ar in range(2):
                                    for ac in range(2):
                                        idx = a_idx(br, bc, ar, ac)
                                        u_entries[idx] = u_entries.get(idx, 0) + total_coeff

            v_entries = {}
            for r in range(2):
                for c in range(2):
                    coeff = v_coeff[r][c]
                    if coeff != 0:
                        for block_idx, block_coeff in B_comb[r][c]:
                            total_coeff = coeff * block_coeff
                            if total_coeff != 0:
                                br = block_idx // 2
                                bc = block_idx % 2
                                for br2 in range(2):
                                    for bc2 in range(2):
                                        idx = b_idx(br, bc, br2, bc2)
                                        v_entries[idx] = v_entries.get(idx, 0) + total_coeff

            # Calculate W entries
            # The result of the inner Strassen term i is a 2x2 matrix contribution to M_k.
            # Let's call it M_k_contrib.
            # M_k_contrib[rc] = (sum u_coeff[r][w] A_comb[r][w]) * (sum v_coeff[w][c] B_comb[w][c])?
            # No, the inner Strassen term is a rank-1 term in the 2x2x2 tensor.
            # It contributes to all 4 entries of M_k.
            # The coefficient for M_k[rc] is determined by the inner Strassen W matrix.
            
            # Inner Strassen W matrices for the 7 terms:
            # Term 1: [[1, 0], [0, 1]]? No.
            # M1 contributes to C00, C11.
            # In the 2x2 product, M1 = (X00+X11)(Y00+Y11).
            # M1[00] = X00Y00 + X00Y11 + X11Y00 + X11Y11? No.
            # The Strassen terms are sums.
            
            # Let's just use the standard W coefficients for the 7 Strassen terms.
            # W1: [[1, 0], [0, 1]]? No.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for term i, the W matrix is:
            # i=1: [[1, 0], [0, 1]]
            # i=2: [[0, 0], [1, -1]]
            # i=3: [[0, 1], [0, 1]]
            # i=4: [[1, 0], [1, 0]]
            # i=5: [[-1, 1], [0, 0]]
            # i=6: [[0, 0], [0, 1]]
            # i=7: [[1, 0], [0, 0]]
            
            w_coeff = [
                [[1, 0], [0, 1]],
                [[0, 0], [1, -1]],
                [[0, 1], [0, 1]],
                [[1, 0], [1, 0]],
                [[-1, 1], [0, 0]],
                [[0, 0], [0, 1]],
                [[1, 0], [0, 0]],
            ][i]
            
            # Now map w_coeff to global C indices, considering the outer combination.
            # M_k contributes to C blocks with specific signs.
            # Outer combination:
            # C00 += M1 + M4 - M5 + M7
            # C01 += M3 + M5
            # C10 += M2 + M4
            # C11 += M1 - M2 + M3 + M6
            
            outer_w = [
                [[1, 0], [0, 1]],   # M1
                [[0, 0], [1, -1]],  # M2
                [[0, 1], [0, 1]],   # M3
                [[1, 0], [1, 0]],   # M4
                [[-1, 1], [0, 0]],  # M5
                [[0, 0], [0, 1]],   # M6
                [[1, 0], [0, 0]],   # M7
            ][k]
            
            w_entries = {}
            for r in range(2):
                for c in range(2):
                    coeff = w_coeff[r][c]
                    if coeff != 0:
                        # M_k[rc] contributes to C block (r, c) with sign outer_w[r][c]
                        total_coeff = coeff * outer_w[r][c]
                        if total_coeff != 0:
                            cr = r
                            cc = c
                            for cr2 in range(2):
                                for cc2 in range(2):
                                    idx = c_idx(cr, cc, cr2, cc2)
                                    w_entries[idx] = w_entries.get(idx, 0) + total_coeff

            # Add the term
            for idx, val in u_entries.items():
                U[idx][col] += val
            for idx, val in v_entries.items():
                V[idx][col] += val
            for idx, val in w_entries.items():
                W[idx][col] += val
            
            col += 1

    # Add 16 terms for the 5th column
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
