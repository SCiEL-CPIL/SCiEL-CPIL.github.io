import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    
    This implements the tensor decomposition:
    T[i, j, k] = sum_{r=0}^{48} U[i, k, r] * V[k, j, r] * W[j, i, r]
    
    Note on Indexing:
    U is flattened as (n*m, rank) -> U[i*n+k, r]
    V is flattened as (m*p, rank) -> V[k*n+j, r]
    W is flattened as (p*n, rank) -> W[k*n+i, r] where k corresponds to j (col of C) 
    and i corresponds to i (row of C) based on the "trace convention" hint and 
    standard column-major interpretation often found in such problems.
    Specifically, the code uses W[j*4+i, r] for C[i,j].
    
    Rank 49 is the best known upper bound for 4x4 matrix multiplication.
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin
    
    n = 4
    rank = 49
    
    # Initialize matrices
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    def u_idx(i, k):
        return i * n + k
    
    def v_idx(k, j):
        return k * n + j
    
    def w_idx(i, j):
        # Convention: W[k*n+i, r] maps to C[i,j]
        # We interpret the 'k' in the formula as the column index 'j' of C
        # and 'i' in the formula as the row index 'i' of C.
        # So index = j * 4 + i
        return j * n + i

    # Recursive Strassen construction
    # Outer level: 4x4 matrices split into 2x2 blocks
    # Inner level: 2x2 matrices split into 1x1 scalars (using Strassen's 7 terms)
    
    # Strassen terms for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Each term is defined by coefficients for A elements, B elements, and C elements
    # A elements: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B elements: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C elements: (0,0), (0,1), (1,0), (1,1)
    
    strassen_2x2 = [
        # P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # P4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Outer Strassen terms for 4x4 (block level)
    # Blocks are indexed by (br, bc) where br, bc in {0, 1}
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # The 7 Strassen products at the block level:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    outer_terms = [
        # P1: A00+A11, B00+B11 -> C00, C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2: A10+A11, B00 -> C10, -C11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P3: A00, B01-B11 -> C01, C11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # P4: A11, B10-B00 -> C00, C10
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P5: A00+A01, B11 -> -C00, C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6: A10-A00, B00+B01 -> C11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7: A01-A11, B10+B11 -> C00
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Construct the rank-49 decomposition
    # Each rank-1 term corresponds to a pair (outer_term, inner_term)
    
    for t_outer in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[t_outer]
        
        for t_inner in range(7):
            a_elems, b_elems, c_elems = strassen_2x2[t_inner]
            r = t_outer * 7 + t_inner
            
            # Fill U
            for (br, bc), coeff_a_block in a_blocks.items():
                for (er, ec), coeff_a_elem in a_elems.items():
                    # Global row = 2*br + er, Global col = 2*bc + ec
                    row = 2 * br + er
                    col = 2 * bc + ec
                    U[u_idx(row, col)][r] += coeff_a_block * coeff_a_elem
            
            # Fill V
            for (br, bc), coeff_b_block in b_blocks.items():
                for (er, ec), coeff_b_elem in b_elems.items():
                    # Global row = 2*br + er, Global col = 2*bc + ec
                    row = 2 * br + er
                    col = 2 * bc + ec
                    V[v_idx(row, col)][r] += coeff_b_block * coeff_b_elem
            
            # Fill W
            for (br, bc), coeff_c_block in c_blocks.items():
                for (er, ec), coeff_c_elem in c_elems.items():
                    # Global row = 2*br + er, Global col = 2*bc + ec
                    row = 2 * br + er
                    col = 2 * bc + ec
                    W[w_idx(row, col)][r] += coeff_c_block * coeff_c_elem

    # Convert to NumPy arrays for potential further processing or return
    # The problem asks for a dict with 'U', 'V', 'W' which can be lists or arrays.
    # Lists of lists are standard and safe.
    
    return {"U": U, "V": V, "W": W}
