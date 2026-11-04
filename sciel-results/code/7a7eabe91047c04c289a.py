import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # We'll build the decomposition term by term
    terms = []  # each term is (u, v, w) where u is length n*m=16, v is length m*p=28, w is length p*n=28
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a term where u_dict maps (i,j) -> coeff, v_dict maps (j,k) -> coeff, w_dict maps (k,i) -> coeff"""
        u = [0] * (n * m)
        for (i, j), val in u_dict.items():
            u[i * m + j] = val
        v = [0] * (m * p)
        for (j, k), val in v_dict.items():
            v[j * p + k] = val
        w = [0] * (p * n)
        for (k, i), val in w_dict.items():
            w[k * n + i] = val
        terms.append((u, v, w))
    
    # Strassen for 4x4x4 part (k=0,1,2,3)
    # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # Split B into 2x2 blocks: B00, B01, B10, B11 (each 2x2)
    # For k in 0..3:
    #   A00: i in {0,1}, j in {0,1}
    #   A01: i in {0,1}, j in {2,3}
    #   A10: i in {2,3}, j in {0,1}
    #   A11: i in {2,3}, j in {2,3}
    #   B00: j in {0,1}, k in {0,1}
    #   B01: j in {0,1}, k in {2,3}
    #   B10: j in {2,3}, k in {0,1}
    #   B11: j in {2,3}, k in {2,3}
    
    # Strassen products for 4x4x4:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10+A11)(B10-B01)
    # m7 = (A00-A01)(B00+B10)
    
    # Each mi is a 2x2x2 multiplication, which we do with schoolbook (4 terms)
    
    # Helper to add a 2x2x2 schoolbook multiplication with linear combinations
    def add_2x2x2(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict mapping (local_i, local_j) -> coeff for A block
        v_coeffs: dict mapping (local_j, local_k) -> coeff for B block  
        w_coeffs: dict mapping (local_k, local_i) -> coeff for C block
        local indices are 0,1 for each dimension
        """
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    # u part: sum over all (i,j) in u_coeffs
                    u = {}
                    for (li2, lj2), coeff in u_coeffs.items():
                        i = li2  # local i maps to global i offset
                        j = lj2  # local j maps to global j offset
                        # But we need to map to actual global indices
                        # This is getting complex. Let me use a different approach.
                        pass
        # Let me just do it more directly
        pass
    
    # Let me take a simpler approach: just use schoolbook for everything but with
    # some Strassen-like optimizations for the 4x4x4 part.
    
    # Actually, let me just use schoolbook for simplicity and correctness.
    # The current score is 1.0 with rank 112. Let me try to get a better decomposition.
    
    # For 4x4x7, the known lower bound is around 60-70.
    # Let me try a mixed approach: Strassen for the 4x4x4 part, schoolbook for 4x4x3.
    
    # Clear terms
    terms = []
    
    # 4x4x4 part: use Strassen (7 products, each 2x2x2 schoolbook = 4 terms, total 28)
    # 4x4x3 part: schoolbook = 4*4*3 = 48 terms
    # Total: 76 terms
    
    # Strassen for 4x4x4:
    # A blocks: A00 (i:0-1, j:0-1), A01 (i:0-1, j:2-3), A10 (i:2-3, j:0-1), A11 (i:2-3, j:2-3)
    # B blocks: B00 (j:0-1, k:0-1), B01 (j:0-1, k:2-3), B10 (j:2-3, k:0-1), B11 (j:2-3, k:2-3)
    # C blocks: C00 (k:0-1, i:0-1), C01 (k:0-1, i:2-3), C10 (k:2-3, i:0-1), C11 (k:2-3, i:2-3)
    
    # For each Strassen product, we need to express it as sum of rank-1 terms.
    # A 2x2x2 schoolbook has 4 terms.
    
    # Let me define helper functions to create the coefficient vectors
    
    def strassen_term(u_block_coeffs, v_block_coeffs, w_block_coeffs, 
                      u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        """
        Add terms for a 2x2x2 multiplication where:
        - u_block_coeffs: dict (li, lj) -> coeff, where li,lj are local indices 0,1
        - v_block_coeffs: dict (lj, lk) -> coeff
        - w_block_coeffs: dict (lk, li) -> coeff
        - offsets map local indices to global
        """
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    # u part
                    for (li2, lj2), coeff in u_block_coeffs.items():
                        gi = u_offset_i + li2
                        gj = u_offset_j + lj2
                        u[gi * m + gj] = coeff
                    
                    # v part
                    for (lj2, lk2), coeff in v_block_coeffs.items():
                        gj = v_offset_j + lj2
                        gk = v_offset_k + lk2
                        v[gj * p + gk] = coeff
                    
                    # w part
                    for (lk2, li2), coeff in w_block_coeffs.items():
                        gk = w_offset_k + lk2
                        gi = w_offset_i + li2
                        w[gk * n + gi] = coeff
                    
                    terms.append((u, v, w))
    
    # Strassen products:
    # m1 = (A00+A11)(B00+B11)
    # u: A00 + A11, v: B00 + B11, w: C00 + C11 (but actually C = m1 - m4 + m6 - m2 for C00, etc.)
    # Wait, the Strassen formula gives C blocks as combinations of mi.
    # So each mi contributes to multiple C blocks.
    
    # Let me just do schoolbook for everything to ensure correctness.
    # Rank 112.
    
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
