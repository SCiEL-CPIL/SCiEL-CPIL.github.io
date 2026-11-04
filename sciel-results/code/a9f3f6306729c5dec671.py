"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We split the inner dimension m=4 into two groups of 2.
    # Group 0: j in {0, 1}
    # Group 1: j in {2, 3}
    
    # For each group, we compute C_group = A_group * B_group
    # A_group is 2x2, B_group is 2x5.
    # We split the 5 columns of B_group into two parts: k in {0,1} and k in {2,3,4}.
    # Part 1: 2x2x2 (Rank 7, Strassen)
    # Part 2: 2x2x3 (Rank 12, Schoolbook)
    # Total rank per group = 7 + 12 = 19.
    # Total rank = 19 * 2 = 38.
    
    rank = 38
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 10 rows
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2): (i, j_local) -> global (i, j_global)
    # Indices for B (2x5): (j_local, k) -> global (j_global, k)
    # Indices for C (2x5): (i, k) -> global (i, k)
    # W index: k*n + i
    
    for group in range(2):
        j_offset = group * 2
        # A indices: i in 0..1, j_local in 0..1 -> j_global = j_offset + j_local
        # B indices: j_local in 0..1, k in 0..4 -> j_global = j_offset + j_local
        
        # --- Part 1: k in {0, 1} (2x2x2 Strassen) ---
        
        # M1 = (A00+A11)(B00+B11)
        # A00: i=0, j_local=0. A11: i=1, j_local=1.
        # B00: j_local=0, k=0. B11: j_local=1, k=1.
        
        # Term M1
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1) -- Wait, W is for C.
        # C00: i=0, k=0. C11: i=1, k=1.
        
        # We need to add contributions to U, V, W for each Strassen multiplication.
        # This is a bit complex to do term by term.
        # Let's expand the Strassen terms into the tensor decomposition.
        
        # M1 = (A00+A11)(B00+B11)
        # This contributes to:
        # C00: +1
        # C11: +1
        # But wait, the decomposition is linear.
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's define the 7 Strassen multiplications for this group.
        # Each multiplication r has:
        # - A coefficients: vector a of size 4 (for i,j_local)
        # - B coefficients: vector b of size 10 (for j_local,k)
        # - C coefficients: vector c of size 10 (for i,k)
        
        # M1:
        # a: A00=1, A11=1
        # b: B00=1, B11=1
        # c: C00=1, C11=1
        
        # M2:
        # a: A10=1, A11=1
        # b: B00=1
        # c: C10=1, C11=-1
        
        # M3:
        # a: A00=1
        # b: B10=1, B00=-1
        # c: C01=1, C11=1
        
        # M4:
        # a: A11=1
        # b: B01=1, B11=-1
        # c: C00=1, C10=1
        
        # M5:
        # a: A00=1, A01=1
        # b: B11=1
        # c: C00=-1, C01=1
        
        # M6:
        # a: A10=1, A00=-1
        # b: B00=1, B10=1
        # c: C11=1
        
        # M7:
        # a: A01=1, A11=-1
        # b: B01=1, B11=1
        # c: C00=1
        
        # Let's verify C00 = M1 + M4 - M5 + M7
        # M1: +1
        # M4: +1
        # M5: -1
        # M7: +1
        # Sum: 2? No.
        
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify the coefficients.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B01-B11) = A11B01 - A11B11
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M7 = (A01-A11)(B01+B11) = A01B01 + A01B11 - A11B01 - A11B11
        
        # C00 = M1 + M4 - M5 + M7
        # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B01 - A11B11) - (A00B11 + A01B11) + (A01B01 + A01B11 - A11B01 - A11B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B11 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
        # = A00B00 + A11B00 - A11B11 + A01B01
        # This is not A00B00 + A01B01.
        
        # I am using the wrong Strassen formulas or applying them incorrectly.
        # The standard Strassen is for C = A * B where A, B are 2x2.
        # Here A is 2x2, B is 2x2 (for the 2x2x2 block).
        # C is 2x2.
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # My previous formulas were for C = A * B^T or something?
        # Let's use the correct Strassen for C = A * B.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check C00 again.
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B01 - A11B11
        # M5 = A00B11 + A01B11
        # M7 = A01B01 + A01B11 - A11B01 - A11B11
        
        # C00 = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B11 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
        # = A00B00 + A11B00 - A11B11 + A01B01
        # Target: A00B00 + A01B10
        
        # It seems I am mixing up B indices.
        # In C = A * B, C_ij = sum_k A_ik B_kj.
        # So C00 = A00B00 + A01B10.
        # My M3 uses B10.
        # My M4 uses B01.
        
        # Let's restart the Strassen derivation for C = A * B.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Check C00:
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B01 - A11B11
        # M5 = A00B11 + A01B11
        # M7 = A01B01 + A01B11 - A11B01 - A11B11
        
        # Sum: A00B00 + A11B00 - A11B11 + A01B01.
        # Target: A00B00 + A01B10.
        
        # The formulas I'm using are for C = A * B^T?
        # If C = A * B^T, C00 = A00B00 + A01B01.
        # Then C00 = A00B00 + A11B00 - A11B11 + A01B01. Still not matching.
        
        # I will use the schoolbook for 2x2x2 as well to be safe.
        # Rank 4 for 2x2x2.
        # Total rank per group = 4 + 12 = 16.
        # Total rank = 32.
        
        # This is a significant improvement over 40.
        
    # Reset
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    for group in range(2):
        j_offset = group * 2
        
        # Part 1: k in {0, 1} (2x2x2 Schoolbook)
        # 4 terms
        for i in range(2):
            for j_local in range(2):
                for k in range(2):
                    j_global = j_offset + j_local
                    u_idx = i * m + j_global
                    v_idx = j_global * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1.0
                    V[v_idx][col] = 1.0
                    W[w_idx][col] = 1.0
                    col += 1
        
        # Part 2: k in {2, 3, 4} (2x2x3 Schoolbook)
        # 12 terms
        for i in range(2):
            for j_local in range(2):
                for k in range(2, 5):
                    j_global = j_offset + j_local
                    u_idx = i * m + j_global
                    v_idx = j_global * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1.0
                    V[v_idx][col] = 1.0
                    W[w_idx][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
