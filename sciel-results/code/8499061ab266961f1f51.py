"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 17
    V = [[0] * rank for _ in range(m * p)] # 9 x 17
    W = [[0] * rank for _ in range(p * n)] # 6 x 17
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
        
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i in 0..1, j in 0..1 -> u_idx = i*3 + j
    # B indices: j in 0..1, k in 0..1 -> v_idx = j*3 + k
    # C indices: k in 0..1, i in 0..1 -> w_idx = k*2 + i
    
    # P1 = A00 * (B00 + B11)
    add_term(0*3+0, 0*3+0, 0*2+0) # A00, B00, C00
    add_term(0*3+0, 1*3+1, 0*2+0) # A00, B11, C00
    
    # P2 = (A10 + A11) * B00
    add_term(1*3+0, 0*3+0, 1*2+0) # A10, B00, C10
    add_term(1*3+1, 0*3+0, 1*2+0) # A11, B00, C10
    
    # P3 = (A00 + A01) * B11
    add_term(0*3+0, 1*3+1, 0*2+1) # A00, B11, C01
    add_term(0*3+1, 1*3+1, 0*2+1) # A01, B11, C01
    
    # P4 = A11 * (B01 - B10)
    add_term(1*3+1, 0*3+1, 1*2+1) # A11, B01, C11
    add_term(1*3+1, 1*3+0, 1*2+1, -1) # A11, B10, C11 (-1)
    
    # P5 = (A01 + A11) * (B00 + B10)
    add_term(0*3+1, 0*3+0, 0*2+0) # A01, B00, C00
    add_term(0*3+1, 1*3+0, 0*2+0) # A01, B10, C00
    add_term(1*3+1, 0*3+0, 1*2+0) # A11, B00, C10
    add_term(1*3+1, 1*3+0, 1*2+0) # A11, B10, C10
    
    # P6 = (A00 - A10) * (B01 + B11)
    add_term(0*3+0, 0*3+1, 0*2+1) # A00, B01, C01
    add_term(0*3+0, 1*3+1, 0*2+1) # A00, B11, C01
    add_term(1*3+0, 0*3+1, 1*2+1, -1) # A10, B01, C11 (-1)
    add_term(1*3+0, 1*3+1, 1*2+1, -1) # A10, B11, C11 (-1)
    
    # P7 = (A01 - A10) * (B00 + B10)
    add_term(0*3+1, 0*3+0, 0*2+0) # A01, B00, C00
    add_term(0*3+1, 1*3+0, 0*2+0) # A01, B10, C00
    add_term(1*3+0, 0*3+0, 1*2+0, -1) # A10, B00, C10 (-1)
    add_term(1*3+0, 1*3+0, 1*2+0, -1) # A10, B10, C10 (-1)
    
    # Wait, the above add_term calls are adding individual monomials to the columns.
    # This is NOT how Strassen works in this tensor decomposition format.
    # In the tensor decomposition, each column corresponds to ONE product of a linear form in U, a linear form in V, and a linear form in W.
    # My add_term above adds 1 to specific entries. This effectively creates rank-1 tensors for each monomial.
    # If I do this for all monomials, I get the schoolbook rank (18).
    # To get Strassen rank 7, I need to combine the linear forms.
    # e.g., P1 = (A00) * (B00 + B11).
    # This corresponds to:
    # U vector has 1 at A00.
    # V vector has 1 at B00 and 1 at B11.
    # W vector has 1 at C00 and 1 at C11? No.
    # P1 contributes to C00 and C11?
    # C00 = P1 + ...
    # C11 = P1 + ...
    # So W vector should have 1 at C00 and 1 at C11.
    
    # Let's restart the construction logic.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    def add_strassen_term(u_vec, v_vec, w_vec):
        nonlocal col
        for idx, val in u_vec.items():
            U[idx][col] += val
        for idx, val in v_vec.items():
            V[idx][col] += val
        for idx, val in w_vec.items():
            W[idx][col] += val
        col += 1
        
    # Indices:
    # A: i*3+j
    # B: j*3+k
    # C: k*2+i
    
    # P1 = A00 * (B00 + B11)
    # Contributes to C00 and C11?
    # C00 = P1 + P4 - P5 + P7
    # C11 = P1 - P2 + P3 + P6
    # So P1 adds to C00 and C11.
    add_strassen_term(
        {0*3+0: 1}, # A00
        {0*3+0: 1, 1*3+1: 1}, # B00, B11
        {0*2+0: 1, 1*2+1: 1} # C00, C11
    )
    
    # P2 = (A10 + A11) * B00
    # Contributes to C10 and C11?
    # C10 = P2 + P4
    # C11 = P1 - P2 + ...
    # So P2 adds to C10 and -C11.
    add_strassen_term(
        {1*3+0: 1, 1*3+1: 1}, # A10, A11
        {0*3+0: 1}, # B00
        {1*2+0: 1, 1*2+1: -1} # C10, C11
    )
    
    # P3 = (A00 + A01) * B11
    # Contributes to C01 and C11?
    # C01 = P3 + P5
    # C11 = ... + P3
    add_strassen_term(
        {0*3+0: 1, 0*3+1: 1}, # A00, A01
        {1*3+1: 1}, # B11
        {0*2+1: 1, 1*2+1: 1} # C01, C11
    )
    
    # P4 = A11 * (B01 - B10)
    # Contributes to C10 and C11?
    # C10 = P2 + P4
    # C11 = P4 ...
    add_strassen_term(
        {1*3+1: 1}, # A11
        {0*3+1: 1, 1*3+0: -1}, # B01, B10
        {1*2+0: 1, 1*2+1: 1} # C10, C11
    )
    
    # P5 = (A01 + A11) * (B00 + B10)
    # Contributes to C00 and C01?
    # C00 = ... - P5 ...
    # C01 = P3 + P5
    add_strassen_term(
        {0*3+1: 1, 1*3+1: 1}, # A01, A11
        {0*3+0: 1, 1*3+0: 1}, # B00, B10
        {0*2+0: -1, 0*2+1: 1} # C00, C01
    )
    
    # P6 = (A00 - A10) * (B01 + B11)
    # Contributes to C01 and C11?
    # C01 = ...
    # C11 = ... + P6
    # Wait, C01 does not have P6 in standard Strassen?
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # So P6 only contributes to C11.
    add_strassen_term(
        {0*3+0: 1, 1*3+0: -1}, # A00, A10
        {0*3+1: 1, 1*3+1: 1}, # B01, B11
        {1*2+1: 1} # C11
    )
    
    # P7 = (A01 - A10) * (B00 + B10)
    # Contributes to C00?
    # C00 = ... + P7
    add_strassen_term(
        {0*3+1: 1, 1*3+0: -1}, # A01, A10
        {0*3+0: 1, 1*3+0: 1}, # B00, B10
        {0*2+0: 1} # C00
    )
    
    # Part 2: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    
    # Term 8: A00 * B02 -> C02
    add_strassen_term(
        {0*3+0: 1},
        {0*3+2: 1},
        {2*2+0: 1}
    )
    
    # Term 9: A01 * B12 -> C02
    add_strassen_term(
        {0*3+1: 1},
        {1*3+2: 1},
        {2*2+0: 1}
    )
    
    # Term 10: A10 * B02 -> C12
    add_strassen_term(
        {1*3+0: 1},
        {0*3+2: 1},
        {2*2+1: 1}
    )
    
    # Term 11: A11 * B12 -> C12
    add_strassen_term(
        {1*3+1: 1},
        {1*3+2: 1},
        {2*2+1: 1}
    )
    
    # Part 3: A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # C00 += A02 B20
    # C01 += A02 B21
    # C02 += A02 B22
    # C10 += A12 B20
    # C11 += A12 B21
    # C12 += A12 B22
    
    # Term 12: A02 * B20 -> C00
    add_strassen_term(
        {0*3+2: 1},
        {2*3+0: 1},
        {0*2+0: 1}
    )
    
    # Term 13: A02 * B21 -> C01
    add_strassen_term(
        {0*3+2: 1},
        {2*3+1: 1},
        {0*2+1: 1}
    )
    
    # Term 14: A02 * B22 -> C02
    add_strassen_term(
        {0*3+2: 1},
        {2*3+2: 1},
        {2*2+0: 1}
    )
    
    # Term 15: A12 * B20 -> C10
    add_strassen_term(
        {1*3+2: 1},
        {2*3+0: 1},
        {1*2+0: 1}
    )
    
    # Term 16: A12 * B21 -> C11
    add_strassen_term(
        {1*3+2: 1},
        {2*3+1: 1},
        {1*2+1: 1}
    )
    
    # Term 17: A12 * B22 -> C12
    add_strassen_term(
        {1*3+2: 1},
        {2*3+2: 1},
        {2*2+1: 1}
    )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
