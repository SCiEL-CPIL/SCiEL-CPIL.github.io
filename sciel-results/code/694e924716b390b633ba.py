import time
import numpy as np

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4
    rank = 49

    # Initialize matrices with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Helper to get indices
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen 2x2 terms
    # Each term is a tuple of (A_mask, B_mask, C_mask)
    # Masks are 2x2 arrays
    def get_strassen_2x2_terms():
        terms = []
        # t0: (A00+A11) * (B00+B11)
        A0 = np.array([[1, 0], [0, 1]])
        B0 = np.array([[1, 0], [0, 1]])
        C0 = np.array([[1, 0], [0, 1]])
        terms.append((A0, B0, C0))
        
        # t1: (A10+A11) * B00
        A1 = np.array([[0, 0], [1, 1]])
        B1 = np.array([[1, 0], [0, 0]])
        C1 = np.array([[0, 0], [1, -1]])
        terms.append((A1, B1, C1))
        
        # t2: A00 * (B01-B11)
        A2 = np.array([[1, 0], [0, 0]])
        B2 = np.array([[0, 1], [0, -1]])
        C2 = np.array([[0, 1], [0, 1]])
        terms.append((A2, B2, C2))
        
        # t3: A11 * (B10-B00)
        A3 = np.array([[0, 0], [0, 1]])
        B3 = np.array([[0, 0], [1, 0]])
        C3 = np.array([[1, 0], [1, 0]])
        terms.append((A3, B3, C3))
        
        # t4: (A00+A01) * B11
        A4 = np.array([[1, 1], [0, 0]])
        B4 = np.array([[0, 0], [0, 1]])
        C4 = np.array([[-1, 1], [0, 0]])
        terms.append((A4, B4, C4))
        
        # t5: (A10-A00) * (B00+B01)
        A5 = np.array([[-1, 0], [1, 0]])
        B5 = np.array([[1, 1], [0, 0]])
        C5 = np.array([[0, 0], [0, 1]])
        terms.append((A5, B5, C5))
        
        # t6: (A01-A11) * (B10+B11)
        A6 = np.array([[0, 1], [0, -1]])
        B6 = np.array([[0, 0], [1, 1]])
        C6 = np.array([[1, 0], [0, 0]])
        terms.append((A6, B6, C6))
        
        return terms

    strassen_terms = get_strassen_2x2_terms()

    # Block-level Strassen for 4x4
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    # C blocks: C00, C01, C10, C11 (2x2 each)
    
    block_mults = [
        # P0 = (A00 + A11) * (B00 + B11)
        {
            'A_blocks': { (0,0): 1, (1,1): 1 },
            'B_blocks': { (0,0): 1, (1,1): 1 },
            'C_contrib': { (0,0): 1, (1,1): 1 }
        },
        # P1 = (A10 + A11) * B00
        {
            'A_blocks': { (1,0): 1, (1,1): 1 },
            'B_blocks': { (0,0): 1 },
            'C_contrib': { (1,0): 1, (1,1): -1 }
        },
        # P2 = A00 * (B01 - B11)
        {
            'A_blocks': { (0,0): 1 },
            'B_blocks': { (0,1): 1, (1,1): -1 },
            'C_contrib': { (0,1): 1, (1,1): 1 }
        },
        # P3 = A11 * (B10 - B00)
        {
            'A_blocks': { (1,1): 1 },
            'B_blocks': { (1,0): 1, (0,0): -1 },
            'C_contrib': { (0,0): 1, (1,0): 1 }
        },
        # P4 = (A00 + A01) * B11
        {
            'A_blocks': { (0,0): 1, (0,1): 1 },
            'B_blocks': { (1,1): 1 },
            'C_contrib': { (0,0): -1, (0,1): 1 }
        },
        # P5 = (A10 - A00) * (B00 + B01)
        {
            'A_blocks': { (1,0): 1, (0,0): -1 },
            'B_blocks': { (0,0): 1, (0,1): 1 },
            'C_contrib': { (1,1): 1 }
        },
        # P6 = (A01 - A11) * (B10 + B11)
        {
            'A_blocks': { (0,1): 1, (1,1): -1 },
            'B_blocks': { (1,0): 1, (1,1): 1 },
            'C_contrib': { (0,0): 1 }
        }
    ]

    rank_idx = 0
    for b_mult in block_mults:
        for s_term in strassen_terms:
            if time.time() - start_time > DEADLINE:
                return {"U": U, "V": V, "W": W}
            
            A_mask, B_mask, C_mask = s_term
            
            # Combine block and local terms
            # U elements
            U_elems = {}
            for (bi, bj), c_b in b_mult['A_blocks'].items():
                for li in range(2):
                    for lj in range(2):
                        c_l = A_mask[li, lj]
                        if c_l != 0:
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            idx = u_idx(gi, gj)
                            U_elems[idx] = U_elems.get(idx, 0) + c_b * c_l
            
            # V elements
            V_elems = {}
            for (bi, bj), c_b in b_mult['B_blocks'].items():
                for li in range(2):
                    for lj in range(2):
                        c_l = B_mask[li, lj]
                        if c_l != 0:
                            gj = 2 * bi + li
                            gk = 2 * bj + lj
                            idx = v_idx(gj, gk)
                            V_elems[idx] = V_elems.get(idx, 0) + c_b * c_l
            
            # W elements
            W_elems = {}
            for (ci, cj), c_b in b_mult['C_contrib'].items():
                for li in range(2):
                    for lj in range(2):
                        c_l = C_mask[li, lj]
                        if c_l != 0:
                            gi = 2 * ci + li
                            gj = 2 * cj + lj
                            # W index is k*n + i. Here gj is col (k), gi is row (i).
                            idx = w_idx(gj, gi)
                            W_elems[idx] = W_elems.get(idx, 0) + c_b * c_l
            
            for idx, val in U_elems.items():
                if val != 0:
                    U[idx, rank_idx] = val
            for idx, val in V_elems.items():
                if val != 0:
                    V[idx, rank_idx] = val
            for idx, val in W_elems.items():
                if val != 0:
                    W[idx, rank_idx] = val
            
            rank_idx += 1

    return {"U": U, "V": V, "W": W}
