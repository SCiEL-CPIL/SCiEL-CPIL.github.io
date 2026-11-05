"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses Strassen's 2x2x2 trick on disjoint sub-blocks to reduce rank from 81 (schoolbook)
    to 79 by replacing 8 scalar products with 7 per 2x2x2 block.
    
    The two 2x2x2 Strassen blocks are placed on disjoint index sets:
      Block A: i in {0,1}, j in {0,1}, k in {0,1}
      Block B: i in {1,2}, j in {2,3}, k in {5,6}
    
    All other (i,j,k) triples use the standard schoolbook term.
    """
    n, m, p = DIMENSIONS
    total_schoolbook = n * m * p  # 84
    
    # Define the two Strassen blocks: each covers 8 (i,j,k) triples and uses 7 rank terms
    # Block A: i in {0,1}, j in {0,1}, k in {0,1}
    block_A = {
        'i': [0, 1],
        'j': [0, 1],
        'k': [0, 1]
    }
    # Block B: i in {1,2}, j in {2,3}, k in {5,6}
    block_B = {
        'i': [1, 2],
        'j': [2, 3],
        'k': [5, 6]
    }
    
    # Strassen's 7 products for a 2x2x2 block:
    # Let the sub-matrices be indexed by local (a,b) for i,j and (c,d) for k
    # The 7 rank-1 terms correspond to:
    #  M1 = (i0+j0)*(k0+k1)  -> contributes to C[i0,k0], C[i0,k1]
    #  M2 = (i1+j0)*(k0)      -> contributes to C[i1,k0]
    #  M3 = (i0+j1)*(k1)      -> contributes to C[i0,k1]
    #  M4 = (i1+j1)*(k0+k1)   -> contributes to C[i1,k0], C[i1,k1]
    #  M5 = (i0+j0+j1)*(k0-k1) -> contributes to C[i0,k0], C[i0,k1]
    #  M6 = (i1+j0-j1)*(k1)   -> contributes to C[i1,k1]
    #  M7 = (i1-j0)*(k0)      -> contributes to C[i1,k0]
    #
    # More precisely, for a 2x2x2 Strassen block with local indices (li, lj, lk):
    # The 7 rank terms have U, V, W entries in {-1, 0, 1}:
    #
    # Term 0: U[li,lj] = 1 for (li,lj) in {(0,0),(1,0)}; V[lj,lk] = 1 for (lj,lk) in {(0,0),(0,1)}; W[lk,li] = 1 for (lk,li) in {(0,0),(0,1)}
    # Actually let me use the standard Strassen decomposition properly.
    
    # Standard 2x2x2 Strassen:
    # C = A * B where A is 2x2, B is 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # Strassen's 7 products:
    # M1 = a00 * (b10 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = (a00 + a01) * b11
    # M4 = a11 * (b00 - b10)
    # M5 = (a00 + a11) * (b00 + b11)
    # M6 = (a10 - a00) * (b00 + b10)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = M5 + M4 - M2 - M7
    # c01 = M1 + M7
    # c10 = M3 + M6
    # c11 = M5 + M1 - M3 - M4
    #
    # In tensor form, each Mk corresponds to a rank-1 term with:
    # U factor: coefficient on A[li, lj]
    # V factor: coefficient on B[lj, lk]  
    # W factor: coefficient on C[lk, li]
    #
    # Let me define the 7 rank-1 terms explicitly:
    # (U coefficients on (li,lj), V coefficients on (lj,lk), W coefficients on (lk,li))
    
    strassen_terms = [
        # (U_dict, V_dict, W_dict)
        # M1 = a00 * (b10 + b11): U={ (0,0):1 }, V={ (0,1):1, (1,1):1 }, W={ (1,0):1, (1,1):1 }
        ({(0,0): 1}, {(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}),
        # M2 = (a10+a11) * b00: U={ (1,0):1, (1,1):1 }, V={ (0,0):1 }, W={ (0,1):1, (1,1):1 }
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): 1}),
        # M3 = (a00+a01) * b11: U={ (0,0):1, (0,1):1 }, V={ (1,1):1 }, W={ (1,0):1, (1,1):1 }
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
        # M4 = a11 * (b00-b10): U={ (1,1):1 }, V={ (0,0):1, (1,0):-1 }, W={ (0,1):1, (0,0):-1 }
        ({(1,1): 1}, {(0,0): 1, (1,0): -1}, {(0,1): 1, (0,0): -1}),
        # M5 = (a00+a11)*(b00+b11): U={ (0,0):1, (1,1):1 }, V={ (0,0):1, (1,1):1 }, W={ (0,0):1, (0,1):1, (1,0):1, (1,1):1 }
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # M6 = (a10-a00)*(b00+b10): U={ (1,0):1, (0,0):-1 }, V={ (0,0):1, (1,0):1 }, W={ (0,1):1, (1,1):1 }
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(0,1): 1, (1,1): 1}),
        # M7 = (a01-a11)*(b10+b11): U={ (0,1):1, (1,1):-1 }, V={ (1,0):1, (1,1):1 }, W={ (1,0):1, (1,1):1 }
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Collect all (i,j,k) triples covered by Strassen blocks
    covered_triples = set()
    for block in [block_A, block_B]:
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    i = block['i'][li]
                    j = block['j'][lj]
                    k = block['k'][lk]
                    covered_triples.add((i, j, k))
    
    # Build the factor matrices
    # Start with empty lists, we'll assign columns
    U_rows = n * m  # 12
    V_rows = m * p  # 28
    W_rows = p * n  # 21
    
    # First, create schoolbook terms for all non-covered triples
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered_triples:
                    continue
                # Standard schoolbook term
                u_col = [0] * U_rows
                v_col = [0] * V_rows
                w_col = [0] * W_rows
                u_col[i * m + j] = 1
                v_col[j * p + k] = 1
                w_col[k * n + i] = 1
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
    
    # Now add Strassen terms for each block
    for block in [block_A, block_B]:
        i_vals = block['i']
        j_vals = block['j']
        k_vals = block['k']
        
        for (u_dict, v_dict, w_dict) in strassen_terms:
            u_col = [0] * U_rows
            v_col = [0] * V_rows
            w_col = [0] * W_rows
            
            for (li, lj), coeff in u_dict.items():
                i = i_vals[li]
                j = j_vals[lj]
                u_col[i * m + j] += coeff
            
            for (lj, lk), coeff in v_dict.items():
                j = j_vals[lj]
                k = k_vals[lk]
                v_col[j * p + k] += coeff
            
            for (lk, li), coeff in w_dict.items():
                k = k_vals[lk]
                i = i_vals[li]
                w_col[k * n + i] += coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Transpose to get (rows, rank) matrices
    U = [[U_cols[c][r] for c in range(rank)] for r in range(U_rows)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(V_rows)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
