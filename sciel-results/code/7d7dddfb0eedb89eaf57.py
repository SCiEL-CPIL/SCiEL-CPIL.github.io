import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0
    
    # We will build the decomposition manually based on the Strassen partitioning.
    # We start with a list of terms. Each term is a tuple (u_indices, v_indices, w_indices, coeffs).
    # Actually, it's easier to build U, V, W column by column.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will store the terms as a list of dictionaries or tuples.
    # Term structure: 
    # For a rank-1 term, we have vectors u, v, w.
    # u is a vector of length n*m with non-zero entries at positions i*m+j.
    # v is a vector of length m*p with non-zero entries at positions j*p+k.
    # w is a vector of length p*n with non-zero entries at positions k*n+i.
    # The tensor is sum over r of u_r \otimes v_r \otimes w_r.
    # Specifically, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    # We will accumulate the coefficients for each (i,j,k) in a dictionary for each column?
    # No, we are constructing the decomposition. We need to output U, V, W such that the sum matches.
    # So we define a set of "basis" terms.
    
    # Strategy:
    # 1. Define the 8 Strassen blocks.
    # 2. For each block, add 7 terms.
    # 3. For all other (i,j,k) triples, add 1 schoolbook term.
    
    terms = [] # List of (u_dict, v_dict, w_dict)
    
    def add_term(u_entries, v_entries, w_entries):
        # u_entries: dict {index: coeff}
        # v_entries: dict {index: coeff}
        # w_entries: dict {index: coeff}
        terms.append((u_entries, v_entries, w_entries))

    # Define Strassen block generator
    def strassen_block(i0, j0, k0):
        # i0, j0, k0 are the starting indices for the 2x2x2 block
        # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        
        # Indices for A (U)
        # A11: (i0, j0) -> idx u11
        # A12: (i0, j0+1) -> idx u12
        # A21: (i0+1, j0) -> idx u21
        # A22: (i0+1, j0+1) -> idx u22
        
        u11 = (i0 * m + j0)
        u12 = (i0 * m + j0 + 1)
        u21 = ((i0 + 1) * m + j0)
        u22 = ((i0 + 1) * m + j0 + 1)
        
        # Indices for B (V)
        # B11: (j0, k0) -> idx v11
        # B12: (j0, k0+1) -> idx v12
        # B21: (j0+1, k0) -> idx v21
        # B22: (j0+1, k0+1) -> idx v22
        
        v11 = (j0 * p + k0)
        v12 = (j0 * p + k0 + 1)
        v21 = ((j0 + 1) * p + k0)
        v22 = ((j0 + 1) * p + k0 + 1)
        
        # Indices for C (W)
        # C11: (k0, i0) -> idx w11
        # C12: (k0, i0+1) -> idx w12
        # C21: (k0+1, i0) -> idx w21
        # C22: (k0+1, i0+1) -> idx w22
        
        w11 = (k0 * n + i0)
        w12 = (k0 * n + i0 + 1)
        w21 = ((k0 + 1) * n + i0)
        w22 = ((k0 + 1) * n + i0 + 1)
        
        # Strassen M1..M7
        # M1 = (A11+A22)(B11+B22)
        add_term({u11: 1, u22: 1}, {v11: 1, v22: 1}, {w11: 1, w22: 1, w12: -1, w21: 1}) 
        # Wait, W coefficients must match the linear combination in C.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 terms explicitly based on Strassen formulas.
        # Each term r has U_r, V_r, W_r.
        # T = sum U_r[i,j] V_r[j,k] W_r[k,i]
        
        # M1: U={u11:1, u22:1}, V={v11:1, v22:1}, W={w11:1, w22:1}
        # This contributes to C11 and C22.
        # But we need to distribute the M's to the C's correctly.
        # It is better to define the 7 M terms as intermediate tensors, then combine them into C.
        # However, the decomposition is a sum of rank-1 tensors.
        # The standard Strassen decomposition of the 2x2x2 tensor consists of 7 rank-1 tensors.
        
        # Term 1: M1
        # U: u11:1, u22:1
        # V: v11:1, v22:1
        # W: w11:1, w22:1
        add_term({u11: 1, u22: 1}, {v11: 1, v22: 1}, {w11: 1, w22: 1})
        
        # Term 2: M2
        # U: u21:1, u22:1
        # V: v11:1
        # W: w21:1
        add_term({u21: 1, u22: 1}, {v11: 1}, {w21: 1})
        
        # Term 3: M3
        # U: u11:1
        # V: v12:1, v22:-1
        # W: w12:1
        add_term({u11: 1}, {v12: 1, v22: -1}, {w12: 1})
        
        # Term 4: M4
        # U: u22:1
        # V: v21:1, v11:-1
        # W: w11:1, w21:1
        # Wait, M4 = A22(B21-B11).
        # Contribution to C11 and C21.
        # C11 gets +M4. C21 gets +M4.
        # So W should have w11:1, w21:1.
        add_term({u22: 1}, {v21: 1, v11: -1}, {w11: 1, w21: 1})
        
        # Term 5: M5
        # U: u11:1, u12:1
        # V: v22:1
        # W: w11:-1, w12:1
        # M5 = (A11+A12)B22.
        # Contribution to C11 (-M5) and C12 (+M5).
        # So W should have w11:-1, w12:1.
        add_term({u11: 1, u12: 1}, {v22: 1}, {w11: -1, w12: 1})
        
        # Term 6: M6
        # U: u21:1, u11:-1
        # V: v11:1, v12:1
        # W: w22:1
        # M6 = (A21-A11)(B11+B12).
        # Contribution to C22 (+M6).
        # So W should have w22:1.
        add_term({u21: 1, u11: -1}, {v11: 1, v12: 1}, {w22: 1})
        
        # Term 7: M7
        # U: u12:1, u22:-1
        # V: v21:1, v22:1
        # W: w11:1, w22:1
        # M7 = (A12-A22)(B21+B22).
        # Contribution to C11 (+M7) and C22 (+M7).
        # So W should have w11:1, w22:1.
        add_term({u12: 1, u22: -1}, {v21: 1, v22: 1}, {w11: 1, w22: 1})

    # Define the 8 blocks
    blocks = [
        (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
        (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2)
    ]
    
    covered = set()
    for i0, j0, k0 in blocks:
        strassen_block(i0, j0, k0)
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    covered.add((i, j, k))
                    
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})
                    
    # Now build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
