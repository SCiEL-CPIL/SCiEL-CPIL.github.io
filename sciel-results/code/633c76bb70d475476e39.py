import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 terms
    # A, B, C are 2x2 matrices
    # Term k: sum_{i,j} A_k[i,j] a_i_j * sum_{i,j} B_k[i,j] b_i_j -> C_k
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        {'A': np.array([[1, 0], [0, 1]]), 'B': np.array([[1, 0], [0, 1]]), 'C': np.array([[1, 0], [0, 1]])},
        # M2 = (A10+A11)B00
        {'A': np.array([[0, 0], [1, 1]]), 'B': np.array([[1, 0], [0, 0]]), 'C': np.array([[0, 0], [1, -1]])},
        # M3 = A00(B01-B11)
        {'A': np.array([[1, 0], [0, 0]]), 'B': np.array([[0, 1], [0, -1]]), 'C': np.array([[0, 1], [0, 1]])},
        # M4 = A11(B10-B00)
        {'A': np.array([[0, 0], [0, 1]]), 'B': np.array([[-1, 0], [1, 0]]), 'C': np.array([[1, 0], [1, 0]])},
        # M5 = (A00+A01)B11
        {'A': np.array([[1, 1], [0, 0]]), 'B': np.array([[0, 0], [0, 1]]), 'C': np.array([[-1, 1], [0, 0]])},
        # M6 = (A10-A00)(B00+B01)
        {'A': np.array([[-1, 0], [1, 0]]), 'B': np.array([[1, 1], [0, 0]]), 'C': np.array([[0, 0], [0, 1]])},
        # M7 = (A01-A11)(B10+B11)
        {'A': np.array([[0, 1], [0, -1]]), 'B': np.array([[0, 0], [1, 1]]), 'C': np.array([[1, 0], [0, 0]])},
    ]
    
    terms = []
    
    # Helper to add a term defined by block specs
    def add_term(a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (r, c, coeff)
        # b_blocks: list of (r, c, coeff)
        # c_blocks: list of (r, c, coeff)
        for t in strassen_2x2:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            for i in range(2):
                for j in range(2):
                    # U
                    for (r, c, coeff) in a_blocks:
                        u[(r + i) * m + (c + j)] += coeff * t['A'][i, j]
                    # V
                    for (r, c, coeff) in b_blocks:
                        v[(r + i) * p + (c + j)] += coeff * t['B'][i, j]
                    # W
                    for (r, c, coeff) in c_blocks:
                        w[(c + j) * n + (r + i)] += coeff * t['C'][i, j]
            
            terms.append((u, v, w))
            
    # 4x4x4 Part (49 terms)
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    add_term(
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)]
    )
    # M2 = (A10+A11)B00 -> C10 - C11
    add_term(
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)],
        [(2, 0, 1), (2, 2, -1)]
    )
    # M3 = A00(B01-B11) -> C01 + C11
    add_term(
        [(0, 0, 1)],
        [(0, 2, 1), (2, 2, -1)],
        [(0, 2, 1), (2, 2, 1)]
    )
    # M4 = A11(B10-B00) -> C00 + C10
    add_term(
        [(2, 2, 1)],
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (2, 0, 1)]
    )
    # M5 = (A00+A01)B11 -> -C00 + C01
    add_term(
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)],
        [(0, 0, -1), (0, 2, 1)]
    )
    # M6 = (A10-A00)(B00+B01) -> C11
    add_term(
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)]
    )
    # M7 = (A01-A11)(B10+B11) -> C00
    add_term(
        [(0, 2, 1), (2, 2, -1)],
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)]
    )
    
    # 4x4x2 Part (28 terms)
    # C_top = A00*B_top + A01*B_bottom
    # C_bottom = A10*B_top + A11*B_bottom
    # B_top is cols 4-5, rows 0-1
    # B_bottom is cols 4-5, rows 2-3
    
    # A00 * B_top
    add_term(
        [(0, 0, 1)],
        [(0, 4, 1)],
        [(0, 4, 1)]
    )
    # A01 * B_bottom
    add_term(
        [(0, 2, 1)],
        [(2, 4, 1)],
        [(0, 4, 1)]
    )
    # A10 * B_top
    add_term(
        [(2, 0, 1)],
        [(0, 4, 1)],
        [(2, 4, 1)]
    )
    # A11 * B_bottom
    add_term(
        [(2, 2, 1)],
        [(2, 4, 1)],
        [(2, 4, 1)]
    )
    
    rank = len(terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k, (u, v, w) in enumerate(terms):
        U[:, k] = u
        V[:, k] = v
        W[:, k] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
