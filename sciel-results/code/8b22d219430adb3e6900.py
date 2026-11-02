import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    # U: (n*m) x rank = 10 x rank
    # V: (m*p) x rank = 25 x rank
    # W: (p*n) x rank = 10 x rank
    
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        """
        Add 7 Strassen terms for a 2x2x2 block.
        i indices: {i_start, i_start+1}
        j indices: {j_start, j_start+1}
        k indices: {k_start, k_start+1}
        """
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        u = {(i0, j0): 1, (i1, j1): 1}
        v = {(j0, k0): 1, (j1, k1): 1}
        w = {(k0, i0): 1, (k1, i1): 1}
        terms.append((u, v, w))
        
        # M2 = (a10 + a11)b00
        u = {(i1, j0): 1, (i1, j1): 1}
        v = {(j0, k0): 1}
        w = {(k0, i1): 1, (k1, i1): -1}
        terms.append((u, v, w))
        
        # M3 = a00(b01 - b11)
        u = {(i0, j0): 1}
        v = {(j0, k1): 1, (j1, k1): -1}
        w = {(k1, i0): 1, (k1, i1): 1}
        terms.append((u, v, w))
        
        # M4 = a11(b10 - b00)
        u = {(i1, j1): 1}
        v = {(j1, k0): 1, (j0, k0): -1}
        w = {(k0, i0): 1, (k0, i1): 1}
        terms.append((u, v, w))
        
        # M5 = (a00 + a01)b11
        u = {(i0, j0): 1, (i0, j1): 1}
        v = {(j1, k1): 1}
        w = {(k0, i0): -1, (k1, i0): 1}
        terms.append((u, v, w))
        
        # M6 = (a10 - a00)(b00 + b01)
        u = {(i1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k1): 1}
        w = {(k1, i1): 1}
        terms.append((u, v, w))
        
        # M7 = (a01 - a11)(b10 + b11)
        u = {(i0, j1): 1, (i1, j1): -1}
        v = {(j1, k0): 1, (j1, k1): 1}
        w = {(k0, i0): 1}
        terms.append((u, v, w))

    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """
        Add schoolbook terms for a rectangular block.
        """
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Decomposition Plan for 2x5x5
    # We partition j and k into 2x2 blocks and edges.
    # j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # Block 1: j in {0,1}, k in {0,1} -> Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: j in {0,1}, k in {2,3} -> Strassen
    add_strassen(0, 0, 2)
    
    # Block 3: j in {0,1}, k in {4} -> Schoolbook (2x1x2 block effectively, but i is 2)
    # i in {0,1}, j in {0,1}, k in {4}
    add_schoolbook_block(0, 2, 0, 2, 4, 5)
    
    # Block 4: j in {2,3}, k in {0,1} -> Strassen
    add_strassen(0, 2, 0)
    
    # Block 5: j in {2,3}, k in {2,3} -> Strassen
    add_strassen(0, 2, 2)
    
    # Block 6: j in {2,3}, k in {4} -> Schoolbook
    add_schoolbook_block(0, 2, 2, 4, 4, 5)
    
    # Block 7: j in {4}, k in {0,1} -> Schoolbook
    add_schoolbook_block(0, 2, 4, 5, 0, 2)
    
    # Block 8: j in {4}, k in {2,3} -> Schoolbook
    add_schoolbook_block(0, 2, 4, 5, 2, 4)
    
    # Block 9: j in {4}, k in {4} -> Schoolbook
    add_schoolbook_block(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}
