import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140
    
    terms = []
    
    def strassen_term_2x2(i_off, j_off, k_off, a_map, b_map, c_map):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        i_off, j_off, k_off are offsets in the full matrices.
        a_map, b_map, c_map are dictionaries mapping virtual (r,c) to list of physical (i,j) with coeffs.
        """
        # This function is a bit complex to generalize for arbitrary linear combinations.
        # Instead, I will implement a direct recursive Strassen that returns the terms.
        pass

    def generate_strassen_4x4_4x4():
        """
        Generate terms for 4x4 * 4x4 multiplication using recursive Strassen.
        Returns a list of (u, v, w) tuples.
        """
        result = []
        
        # Strassen 4x4*4x4:
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # Each Mi is a 2x2 matrix product.
        # We need to decompose each 2x2*2x2 product into 7 rank-1 terms.
        
        # Helper to get physical indices from block offset and local indices
        def get_phys_i(block_r, r): return 2*block_r + r
        def get_phys_j(block_c, c): return 2*block_c + c
        def get_phys_k(block_k, k): return 2*block_k + k
        
        # Define the 7 Strassen products for 4x4
        # Each product is defined by:
        # A_side: dict from local (r,c) to list of (phys_i, phys_j, coeff)
        # B_side: dict from local (r,c) to list of (phys_j, phys_k, coeff)
        # C_side: dict from local (r,c) to list of (phys_k, phys_i, coeff)
        
        # M1: (A11+A22)*(B11+B22)
        # A: A11 (0,0) + A22 (2,2)
        # B: B11 (0,0) + B22 (2,2)
        # C: C11 (0,0) + C22 (2,2)
        a1 = {
            (0,0): [(0,0,1), (2,2,1)], (0,1): [(0,1,1), (2,3,1)],
            (1,0): [(1,0,1), (3,2,1)], (1,1): [(1,1,1), (3,3,1)]
        }
        b1 = {
            (0,0): [(0,0,1), (2,2,1)], (0,1): [(0,2,1), (2,4,1)], # Wait, B is 4x4, cols 0-3 for this part?
            # B is 4x4 in this sub-problem (cols 0-3 of original B).
            # So B11 is cols 0,1. B22 is cols 2,3.
            (0,0): [(0,0,1), (2,2,1)], (0,1): [(0,1,1), (2,3,1)],
            (1,0): [(1,0,1), (3,2,1)], (1,1): [(1,1,1), (3,3,1)]
        }
        c1 = {
            (0,0): [(0,0,1), (2,2,1)], (0,1): [(0,1,1), (2,3,1)],
            (1,0): [(1,0,1), (3,2,1)], (1,1): [(1,1,1), (3,3,1)]
        }
        
        # M2: (A21+A22)*B11
        # A: A21 (2,0) + A22 (2,2)
        # B: B11 (0,0)
        # C: C21 (2,0) + C22 (2,2)
        a2 = {
            (0,0): [(2,0,1), (2,2,1)], (0,1): [(2,1,1), (2,3,1)],
            (1,0): [(3,0,1), (3,2,1)], (1,1): [(3,1,1), (3,3,1)]
        }
        b2 = {
            (0,0): [(0,0,1)], (0,1): [(0,1,1)],
            (1,0): [(1,0,1)], (1,1): [(1,1,1)]
        }
        c2 = {
            (0,0): [(2,0,1), (2,2,1)], (0,1): [(2,1,1), (2,3,1)],
            (1,0): [(3,0,1), (3,2,1)], (1,1): [(3,1,1), (3,3,1)]
        }
        
        # M3: A11*(B12-B22)
        # A: A11 (0,0)
        # B: B12 (0,2) - B22 (2,2)
        # C: C12 (0,2) + C11 (0,0) with signs? 
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M3 contributes to C12 (+1) and C22 (+1).
        a3 = {
            (0,0): [(0,0,1)], (0,1): [(0,1,1)],
            (1,0): [(1,0,1)], (1,1): [(1,1,1)]
        }
        b3 = {
            (0,0): [(0,2,1), (2,2,-1)], (0,1): [(0,3,1), (2,3,-1)],
            (1,0): [(1,2,1), (3,2,-1)], (1,1): [(1,3,1), (3,3,-1)]
        }
        c3 = {
            (0,0): [(0,2,1)], (0,1): [(0,3,1)],
            (1,0): [(2,2,1)], (1,1): [(2,3,1)]
        }
        
        # M4: A22*(B21-B11)
        # A: A22 (2,2)
        # B: B21 (2,0) - B11 (0,0)
        # C: C21 (+1) and C11 (+1)
        a4 = {
            (0,0): [(2,2,1)], (0,1): [(2,3,1)],
            (1,0): [(3,2,1)], (1,1): [(3,3,1)]
        }
        b4 = {
            (0,0): [(2,0,1), (0,0,-1)], (0,1): [(2,1,1), (0,1,-1)],
            (1,0): [(3,0,1), (1,0,-1)], (1,1): [(3,1,1), (1,1,-1)]
        }
        c4 = {
            (0,0): [(2,0,1)], (0,1): [(2,1,1)],
            (1,0): [(0,0,1)], (1,1): [(0,1,1)]
        }
        
        # M5: (A11+A12)*B22
        # A: A11 (0,0) + A12 (0,2)
        # B: B22 (2,2)
        # C: C12 (+1) and C11 (-1)
        a5 = {
            (0,0): [(0,0,1), (0,2,1)], (0,1): [(0,1,1), (0,3,1)],
            (1,0): [(1,0,1), (1,2,1)], (1,1): [(1,1,1), (1,3,1)]
        }
        b5 = {
            (0,0): [(2,2,1)], (0,1): [(2,3,1)],
            (1,0): [(3,2,1)], (1,1): [(3,3,1)]
        }
        c5 = {
            (0,0): [(0,2,1)], (0,1): [(0,3,1)],
            (1,0): [(0,0,-1)], (1,1): [(0,1,-1)]
        }
        
        # M6: (A21-A11)*(B11+B12)
        # A: A21 (2,0) - A11 (0,0)
        # B: B11 (0,0) + B12 (0,2)
        # C: C22 (+1)
        a6 = {
            (0,0): [(2,0,1), (0,0,-1)], (0,1): [(2,1,1), (0,1,-1)],
            (1,0): [(3,0,1), (1,0,-1)], (1,1): [(3,1,1), (1,1,-1)]
        }
        b6 = {
            (0,0): [(0,0,1), (0,2,1)], (0,1): [(0,1,1), (0,3,1)],
            (1,0): [(1,0,1), (1,2,1)], (1,1): [(1,1,1), (1,3,1)]
        }
        c6 = {
            (0,0): [(2,2,1)], (0,1): [(2,3,1)],
            (1,0): [(3,2,1)], (1,1): [(3,3,1)]
        }
        
        # M7: (A12-A22)*(B21+B22)
        # A: A12 (0,2) - A22 (2,2)
        # B: B21 (2,0) + B22 (2,2)
        # C: C11 (+1)
        a7 = {
            (0,0): [(0,2,1), (2,2,-1)], (0,1): [(0,3,1), (2,3,-1)],
            (1,0): [(1,2,1), (3,2,-1)], (1,1): [(1,3,1), (3,3,-1)]
        }
        b7 = {
            (0,0): [(2,0,1), (2,2,1)], (0,1): [(2,1,1), (2,3,1)],
            (1,0): [(3,0,1), (3,2,1)], (1,1): [(3,1,1), (3,3,1)]
        }
        c7 = {
            (0,0): [(0,0,1)], (0,1): [(0,1,1)],
            (1,0): [(1,0,1)], (1,1): [(1,1,1)]
        }
        
        products = [
            (a1, b1, c1),
            (a2, b2, c2),
            (a3, b3, c3),
            (a4, b4, c4),
            (a5, b5, c5),
            (a6, b6, c6),
            (a7, b7, c7)
        ]
        
        # Inner Strassen terms for 2x2*2x2
        # Each term is a list of (a_pos, b_pos, c_pos)
        # a_pos: list of (r, c, sign)
        # b_pos: list of (r, c, sign)
        # c_pos: list of (r, c, sign)
        
        inner_terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]), # M1
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]), # M2
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]), # M3
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]), # M4
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]), # M5
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]), # M6
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])  # M7
        ]
        
        for a_map, b_map, c_map in products:
            for a_pos, b_pos, c_pos in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * 4) # Note: B is 4x4 in this sub-problem, but original V is 4x7. 
                                  # We will map to original indices later or handle in main loop.
                                  # Actually, let's generate u, v, w in the original full space sizes to be safe.
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # Expand A
                for (lr, lc, sign) in a_pos:
                    for (pi, pj, coeff) in a_map.get((lr, lc), []):
                        u[pi * m + pj] += sign * coeff
                        
                # Expand B
                for (lr, lc, sign) in b_pos:
                    for (pi, pj, coeff) in b_map.get((lr, lc), []):
                        v[pi * p + pj] += sign * coeff
                        
                # Expand C
                for (lr, lc, sign) in c_pos:
                    for (pi, pj, coeff) in c_map.get((lr, lc), []):
                        # C index is (k, i). In W, index is k * n + i.
                        # pi is k, pj is i.
                        w[pi * n + pj] += sign * coeff
                        
                result.append((u, v, w))
                
        return result

    terms.extend(generate_strassen_4x4_4x4())
    
    # Schoolbook for 4x4 * 4x3 (columns 4,5,6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
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
