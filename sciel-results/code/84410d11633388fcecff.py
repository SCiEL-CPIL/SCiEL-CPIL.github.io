import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    def get_2x2_strassen_scalar_terms():
        """Generate scalar-level terms for 2x2x2 Strassen.
        Returns list of (a_idx, b_idx, c_idx, coeff) where indices are 0-3 for 2x2 matrices.
        """
        # 2x2 Strassen:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Each Mi is a product of linear forms. To get scalar terms, expand each.
        terms = []
        
        # M1 = (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # C00 += M1, C11 += M1
        terms.append((0, 0, 0, 1))
        terms.append((0, 3, 0, 1))
        terms.append((3, 0, 0, 1))
        terms.append((3, 3, 0, 1))
        terms.append((0, 0, 3, 1))
        terms.append((0, 3, 3, 1))
        terms.append((3, 0, 3, 1))
        terms.append((3, 3, 3, 1))
        
        # M2 = (a10+a11)*b00 = a10*b00 + a11*b00
        # C10 += M2, C11 -= M2
        terms.append((2, 0, 2, 1))
        terms.append((3, 0, 2, 1))
        terms.append((2, 0, 3, -1))
        terms.append((3, 0, 3, -1))
        
        # M3 = a00*(b01-b11) = a00*b01 - a00*b11
        # C01 += M3, C11 += M3
        terms.append((0, 1, 1, 1))
        terms.append((0, 3, 1, -1))
        terms.append((0, 1, 3, 1))
        terms.append((0, 3, 3, -1))
        
        # M4 = a11*(b10-b00) = a11*b10 - a11*b00
        # C00 += M4, C10 += M4
        terms.append((3, 2, 0, 1))
        terms.append((3, 0, 0, -1))
        terms.append((3, 2, 2, 1))
        terms.append((3, 0, 2, -1))
        
        # M5 = (a00+a01)*b11 = a00*b11 + a01*b11
        # C00 -= M5, C01 += M5, C11 += M5
        terms.append((0, 3, 0, -1))
        terms.append((1, 3, 0, -1))
        terms.append((0, 3, 1, 1))
        terms.append((1, 3, 1, 1))
        terms.append((0, 3, 3, 1))
        terms.append((1, 3, 3, 1))
        
        # M6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
        # C11 += M6
        terms.append((2, 0, 3, 1))
        terms.append((2, 1, 3, 1))
        terms.append((0, 0, 3, -1))
        terms.append((0, 1, 3, -1))
        
        # M7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
        # C00 += M7
        terms.append((1, 2, 0, 1))
        terms.append((1, 3, 0, 1))
        terms.append((3, 2, 0, -1))
        terms.append((3, 3, 0, -1))
        
        return terms
    
    def get_4x4_strassen_scalar_terms():
        """Generate scalar-level terms for 4x4x4 Strassen.
        Returns list of (a_idx, b_idx, c_idx, coeff) where indices are 0-15 for 4x4 matrices.
        """
        terms_2x2 = get_2x2_strassen_scalar_terms()
        
        # Block offsets for 4x4 split into 2x2 blocks:
        # A00: rows 0-1, cols 0-1 -> base index 0
        # A01: rows 0-1, cols 2-3 -> base index 2
        # A10: rows 2-3, cols 0-1 -> base index 8
        # A11: rows 2-3, cols 2-3 -> base index 10
        
        # B00: rows 0-1, cols 0-1 -> base index 0
        # B01: rows 0-1, cols 2-3 -> base index 2
        # B10: rows 2-3, cols 0-1 -> base index 8
        # B11: rows 2-3, cols 2-3 -> base index 10
        
        # C00: rows 0-1, cols 0-1 -> base index 0
        # C01: rows 0-1, cols 2-3 -> base index 2
        # C10: rows 2-3, cols 0-1 -> base index 8
        # C11: rows 2-3, cols 2-3 -> base index 10
        
        all_terms = []
        
        def add_block_terms(a_base, b_base, c_base, sign=1):
            for (a2, b2, c2, coeff) in terms_2x2:
                a_idx = a_base + a2
                b_idx = b_base + b2
                c_idx = c_base + c2
                all_terms.append((a_idx, b_idx, c_idx, coeff * sign))
        
        # M1 = (A00+A11) @ (B00+B11)
        # = A00@B00 + A00@B11 + A11@B00 + A11@B11
        # C00 += M1, C11 += M1
        add_block_terms(0, 0, 0, 1)    # A00@B00 -> C00
        add_block_terms(0, 2, 0, 1)    # A00@B11 -> C00
        add_block_terms(8, 0, 0, 1)    # A11@B00 -> C00
        add_block_terms(8, 2, 0, 1)    # A11@B11 -> C00
        add_block_terms(0, 0, 10, 1)   # A00@B00 -> C11
        add_block_terms(0, 2, 10, 1)   # A00@B11 -> C11
        add_block_terms(8, 0, 10, 1)   # A11@B00 -> C11
        add_block_terms(8, 2, 10, 1)   # A11@B11 -> C11
        
        # M2 = (A10+A11) @ B00
        # = A10@B00 + A11@B00
        # C10 += M2, C11 -= M2
        add_block_terms(8, 0, 8, 1)    # A10@B00 -> C10
        add_block_terms(10, 0, 8, 1)   # A11@B00 -> C10
        add_block_terms(8, 0, 10, -1)  # A10@B00 -> C11
        add_block_terms(10, 0, 10, -1) # A11@B00 -> C11
        
        # M3 = A00 @ (B01-B11)
        # = A00@B01 - A00@B11
        # C01 += M3, C11 += M3
        add_block_terms(0, 2, 2, 1)    # A00@B01 -> C01
        add_block_terms(0, 10, 2, -1)  # A00@B11 -> C01
        add_block_terms(0, 2, 10, 1)   # A00@B01 -> C11
        add_block_terms(0, 10, 10, -1) # A00@B11 -> C11
        
        # M4 = A11 @ (B10-B00)
        # = A11@B10 - A11@B00
        # C00 += M4, C10 += M4
        add_block_terms(10, 8, 0, 1)   # A11@B10 -> C00
        add_block_terms(10, 0, 0, -1)  # A11@B00 -> C00
        add_block_terms(10, 8, 8, 1)   # A11@B10 -> C10
        add_block_terms(10, 0, 8, -1)  # A11@B00 -> C10
        
        # M5 = (A00+A01) @ B11
        # = A00@B11 + A01@B11
        # C00 -= M5, C01 += M5, C11 += M5
        add_block_terms(0, 10, 0, -1)  # A00@B11 -> C00
        add_block_terms(2, 10, 0, -1)  # A01@B11 -> C00
        add_block_terms(0, 10, 2, 1)   # A00@B11 -> C01
        add_block_terms(2, 10, 2, 1)   # A01@B11 -> C01
        add_block_terms(0, 10, 10, 1)  # A00@B11 -> C11
        add_block_terms(2, 10, 10, 1)  # A01@B11 -> C11
        
        # M6 = (A10-A00) @ (B00+B01)
        # = A10@B00 + A10@B01 - A00@B00 - A00@B01
        # C11 += M6
        add_block_terms(8, 0, 10, 1)   # A10@B00 -> C11
        add_block_terms(8, 2, 10, 1)   # A10@B01 -> C11
        add_block_terms(0, 0, 10, -1)  # A00@B00 -> C11
        add_block_terms(0, 2, 10, -1)  # A00@B01 -> C11
        
        # M7 = (A01-A11) @ (B10+B11)
        # = A01@B10 + A01@B11 - A11@B10 - A11@B11
        # C00 += M7
        add_block_terms(2, 8, 0, 1)    # A01@B10 -> C00
        add_block_terms(2, 10, 0, 1)   # A01@B11 -> C00
        add_block_terms(10, 8, 0, -1)  # A11@B10 -> C00
        add_block_terms(10, 10, 0, -1) # A11@B11 -> C00
        
        return all_terms
    
    # Combine terms
    all_terms = get_4x4_strassen_scalar_terms()
    
    # Deduplicate
    term_map = defaultdict(float)
    for (a_idx, b_idx, c_idx, coeff) in all_terms:
        key = (a_idx, b_idx, c_idx)
        term_map[key] += coeff
    
    # Build U, V, W
    rank = len(term_map) + 12  # 49 + 12 = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for (a_idx, b_idx, c_idx), coeff in term_map.items():
        if abs(coeff) > 1e-10:
            U[a_idx][column] = coeff
            V[b_idx][column] = coeff
            W[c_idx][column] = coeff
            column += 1
    
    # Add schoolbook for last 3 columns
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    assert column == rank, f"Column mismatch: {column} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
