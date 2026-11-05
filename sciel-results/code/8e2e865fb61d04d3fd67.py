"""Rewritten program: deterministic Strassen-based rank-83 decomposition for 3x4*4x7."""
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3 rows of A, m=4 cols of A / rows of B, p=7 cols of B
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    # Strategy: Use Strassen on a 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # This saves 1 rank term: 8 schoolbook -> 7 Strassen
    # Remaining terms: 84 - 8 = 76 schoolbook terms
    # Total rank: 7 + 76 = 83
    
    rank = 83
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen terms for block i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen's 7 products for 2x2 * 2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor decomposition terms:
    # For each Strassen term t, we need:
    # U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] summed over t gives 1 for each (i,j,k) in the block
    # and 0 for (i,j,k) outside the block (since U,V,W entries are 0 outside the block indices)
    
    # The 7 Strassen terms:
    strassen_terms = [
        # (name, U_entries, V_entries, W_entries)
        # U_entries: dict mapping (i,j) -> coefficient
        # V_entries: dict mapping (j,k) -> coefficient  
        # W_entries: dict mapping (k,i) -> coefficient
        ("M1", 
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): -1, (0,1): 1, (1,0): 1}),  # C00: M1, C01: no, C10: no, C11: M1
        # Wait, I need to be more careful.
        # Each term contributes to multiple C entries.
        # Let me redo this properly.
    ]
    
    # Let me think about this differently.
    # In the tensor decomposition, for each rank-1 term c:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # 
    # For the Strassen block, we want:
    # For (i,j,k) in {0,1}x{0,1}x{0,1}: sum = 1
    # For (i,j,k) not in the block: sum = 0 (handled by other terms)
    #
    # The Strassen algorithm computes:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # In tensor form, each M_k is a product of a linear form in A and a linear form in B.
    # The contribution to C_ik is a linear combination of the M_k's.
    #
    # So for each M_k, we have:
    # U entries: coefficients of the linear form in A (indexed by (i,j))
    # V entries: coefficients of the linear form in B (indexed by (j,k))
    # W entries: coefficients in the linear combination for C (indexed by (k,i))
    
    # Let me define the 7 terms properly:
    
    def add_term(U, V, W, col, u_entries, v_entries, w_entries):
        """Add a rank-1 term to U, V, W at column col."""
        for (i, j), coeff in u_entries.items():
            U[i*m+j][col] = coeff
        for (j, k), coeff in v_entries.items():
            V[j*p+k][col] = coeff
        for (k, i), coeff in w_entries.items():
            W[k*n+i][col] = coeff
    
    # M1 = (A00+A11)*(B00+B11)
    # A linear form: A00 + A11 -> {(0,0):1, (1,1):1}
    # B linear form: B00 + B11 -> {(0,0):1, (1,1):1}
    # C contributions: C00 += 1, C11 += 1
    # W entries: {(0,0):1, (1,1):1}
    add_term(U, V, W, 0, 
             {(0,0):1, (1,1):1},
             {(0,0):1, (1,1):1},
             {(0,0):1, (1,1):1})
    
    # M2 = (A10+A11)*B00
    # A: {(1,0):1, (1,1):1}
    # B: {(0,0):1}
    # C: C10 += 1, C11 -= 1
    add_term(U, V, W, 1,
             {(1,0):1, (1,1):1},
             {(0,0):1},
             {(0,1):1, (1,1):-1})
    
    # M3 = A00*(B01-B11)
    # A: {(0,0):1}
    # B: {(0,1):1, (1,1):-1}
    # C: C01 += 1, C11 += 1
    add_term(U, V, W, 2,
             {(0,0):1},
             {(0,1):1, (1,1):-1},
             {(1,0):1, (1,1):1})
    
    # M4 = A11*(B10-B00)
    # A: {(1,1):1}
    # B: {(1,0):1, (0,0):-1}
    # C: C00 += 1, C10 += 1
    add_term(U, V, W, 3,
             {(1,1):1},
             {(1,0):1, (0,0):-1},
             {(0,0):1, (0,1):1})
    
    # M5 = (A00+A01)*B11
    # A: {(0,0):1, (0,1):1}
    # B: {(1,1):1}
    # C: C00 -= 1, C01 += 1
    add_term(U, V, W, 4,
             {(0,0):1, (0,1):1},
             {(1,1):1},
             {(0,0):-1, (1,0):1})
    
    # M6 = (A10-A00)*(B00+B01)
    # A: {(1,0):1, (0,0):-1}
    # B: {(0,0):1, (0,1):1}
    # C: C11 += 1
    add_term(U, V, W, 5,
             {(1,0):1, (0,0):-1},
             {(0,0):1, (0,1):1},
             {(1,1):1})
    
    # M7 = (A01-A11)*(B10+B11)
    # A: {(0,1):1, (1,1):-1}
    # B: {(1,0):1, (1,1):1}
    # C: C00 += 1
    add_term(U, V, W, 6,
             {(0,1):1, (1,1):-1},
             {(1,0):1, (1,1):1},
             {(0,0):1})
    
    # Now add schoolbook terms for all (i,j,k) NOT in the Strassen block
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in the Strassen block
                if i in (0,1) and j in (0,1) and k in (0,1):
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # col should be 83
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")
    
    return {"U": U, "V": V, "W": W}
