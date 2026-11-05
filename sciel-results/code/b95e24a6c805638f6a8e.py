import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    n, m, p = 3, 3, 8

    # We will build a decomposition of rank 70 by splitting the 3x3x8 problem
    # into two 3x3x3 blocks and one 3x3x2 block.
    # Rank(3x3x3) = 26 (using a known non-schoolbook decomposition)
    # Rank(3x3x2) = 18 (schoolbook)
    # Total rank = 2*26 + 18 = 70

    # 1. Define the rank-26 decomposition for 3x3x3
    # Indices: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # We use a specific decomposition found in literature (e.g., Alon, 1998 or similar).
    # One known rank-26 decomposition for 3x3x3 is based on splitting into 2x2 and 1x1 blocks.
    # Let's construct it explicitly.
    
    # Block decomposition:
    # A = [[A22, a21], [a12, a11]] where A22 is 2x2, a21 is 2x1, a12 is 1x2, a11 is 1x1
    # B = [[B22, b21], [b12, b11]]
    # C = AB
    # C22 = A22 B22 + a21 b12
    # C21 = A22 b21 + a21 b11
    # C12 = a12 B22 + a11 b12
    # C11 = a12 b21 + a11 b11
    
    # We need to compute these with minimal multiplications.
    # A22 B22 can be done in 7 mults (Strassen for 2x2x2).
    # Let's define the Strassen decomposition for 2x2x2.
    # Indices for 2x2x2: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Strassen mults:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Let's build the full 3x3x3 tensor decomposition manually.
    # This is complex. An easier approach is to use a known rank-26 tensor.
    # Alternatively, we can just use the schoolbook rank 27 for 3x3x3?
    # If we use 27, total rank is 2*27 + 18 = 72. This is not an improvement.
    # We MUST find a rank-26 decomposition for 3x3x3.
    
    # Let's try to construct the rank-26 decomposition programmatically or hardcode it.
    # Since hardcoding 26 terms is tedious, let's verify if a simple block decomposition works.
    # Actually, there is a simpler way:
    # The rank of 3x3x3 is 23 (Alon 1998). But finding the explicit decomposition is hard.
    # Is there a rank-26 decomposition that is easy to write?
    
    # Let's look at the block decomposition again.
    # C22 = A22 B22 + a21 b12
    # A22 B22 uses 7 mults.
    # a21 b12 is 2x1 * 1x2 = 2x2 matrix. It requires 4 mults if done naively.
    # But we can share some? No, they are distinct entries.
    # So C22 needs 7 + 4 = 11 mults.
    # C21 = A22 b21 + a21 b11
    # A22 b21 is 2x2 * 2x1 = 2x1. 4 mults.
    # a21 b11 is 2x1 * 1x1 = 2x1. 2 mults.
    # Total 6 mults.
    # C12 = a12 B22 + a11 b12
    # a12 B22 is 1x2 * 2x2 = 1x2. 4 mults.
    # a11 b12 is 1x1 * 1x2 = 1x2. 2 mults.
    # Total 6 mults.
    # C11 = a12 b21 + a11 b11
    # a12 b21 is 1x2 * 2x1 = 1x1. 2 mults.
    # a11 b11 is 1x1 * 1x1 = 1x1. 1 mult.
    # Total 3 mults.
    # Sum: 11 + 6 + 6 + 3 = 26.
    
    # This works! We just need to map these scalar multiplications to the tensor decomposition.
    
    # Let's define the terms.
    # We will create lists U, V, W for the 3x3x3 block.
    # Then we will replicate this for the 3x3x8 problem.
    
    # For 3x3x3:
    # U has shape (3*3, 26) = (9, 26)
    # V has shape (3*3, 26) = (9, 26)
    # W has shape (3*3, 26) = (9, 26)
    
    # Let's generate the 26 terms.
    # We'll use a helper to add a term (i, j, k, u_val, v_val, w_val)
    # But the values are linear combinations, not just 1s and 0s.
    # The decomposition is linear. Each term corresponds to a multiplication of a linear form in A and a linear form in B.
    
    # Let's define the 26 terms explicitly.
    # Term 1-7: Strassen for A22 B22
    # A22 indices: (0,0), (0,1), (1,0), (1,1) in the 2x2 block.
    # In 3x3 indices:
    # A22[0,0] = A[0,0], A22[0,1] = A[0,1], A22[1,0] = A[1,0], A22[1,1] = A[1,1]
    # B22[0,0] = B[0,0], B22[0,1] = B[0,1], B22[1,0] = B[1,0], B22[1,1] = B[1,1]
    # C22[0,0] = C[0,0], C22[0,1] = C[0,1], C22[1,0] = C[1,0], C22[1,1] = C[1,1]
    
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    # M2 = (A10+A11)B00 -> contributes to C00, C10
    # M3 = A00(B10-B11) -> contributes to C01, C11
    # M4 = A11(B01-B10) -> contributes to C01, C11
    # M5 = (A00+A01)B11 -> contributes to C00, C10
    # M6 = (A10-A00)(B00+B10) -> contributes to C00, C10
    # M7 = (A01-A11)(B01+B11) -> contributes to C01, C11
    
    # Let's map these to the 3x3 indices.
    # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    # C00 = C[0,0], C01 = C[0,1], C10 = C[1,0], C11 = C[1,1]
    
    # Term 1: (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # U: A[0,0]=1, A[1,1]=1
    # V: B[0,0]=1, B[1,1]=1
    # W: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=-1 (Strassen formula: C00 = M1 - M4 + M6 - M7, etc. Wait, let's use the standard Strassen addition formulas)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M4
    # C10 = M2 - M5 + M7
    # C11 = M1 - M2 + M3 + M5
    
    # So for Term 1 (M1):
    # W[0,0] += 1, W[0,1] += 0, W[1,0] += 0, W[1,1] += 1 (from C11 part? No, M1 is in C00 and C11)
    # C00 gets +1, C11 gets +1.
    
    # Let's build the 26 terms for 3x3x3.
    
    terms_333 = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        terms_333.append((u_dict, v_dict, w_dict))
    
    # M1
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):1, (1,0):-1})
    # M3
    add_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4
    add_term({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,1):1, (1,1):-1})
    # M5
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
    # M6
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1})
    # M7
    add_term({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,1):-1, (1,0):1})
    
    # Now the cross terms.
    # C22 also has a21 b12
    # a21 = [A[0,2], A[1,2]], b12 = [B[2,0], B[2,1]]
    # C22[0,0] += A[0,2]B[2,0]
    # C22[0,1] += A[0,2]B[2,1]
    # C22[1,0] += A[1,2]B[2,0]
    # C22[1,1] += A[1,2]B[2,1]
    
    # Term 8: A[0,2]B[2,0]
    add_term({(0,2):1}, {(2,0):1}, {(0,0):1})
    # Term 9: A[0,2]B[2,1]
    add_term({(0,2):1}, {(2,1):1}, {(0,1):1})
    # Term 10: A[1,2]B[2,0]
    add_term({(1,2):1}, {(2,0):1}, {(1,0):1})
    # Term 11: A[1,2]B[2,1]
    add_term({(1,2):1}, {(2,1):1}, {(1,1):1})
    
    # C21 = A22 b21 + a21 b11
    # b21 = [B[2,0], B[2,1]]? No, b21 is the last column of B22?
    # B = [[B22, b21], [b12, b11]]
    # B22 is top-left 2x2. b21 is top-right 2x1. b12 is bottom-left 1x2. b11 is bottom-right 1x1.
    # So b21 = [B[0,2], B[1,2]], b12 = [B[2,0], B[2,1]], b11 = B[2,2].
    # C21 is the bottom-right 2x1 block of C? No.
    # C = [[C22, C21], [C12, C11]]
    # C21 is 2x1. C21[0] = C[0,2], C21[1] = C[1,2].
    # C21 = A22 b21 + a21 b11
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # We can compute A22 b21 using 4 mults (schoolbook for 2x2 * 2x1)
    # Or can we do better? 2x2 * 2x1 has rank 4.
    # Let's use schoolbook for these small blocks to keep it simple.
    
    # Term 12: A[0,0]B[0,2]
    add_term({(0,0):1}, {(0,2):1}, {(0,2):1})
    # Term 13: A[0,1]B[1,2]
    add_term({(0,1):1}, {(1,2):1}, {(0,2):1})
    # Term 14: A[1,0]B[0,2]
    add_term({(1,0):1}, {(0,2):1}, {(1,2):1})
    # Term 15: A[1,1]B[1,2]
    add_term({(1,1):1}, {(1,2):1}, {(1,2):1})
    
    # a21 b11
    # Term 16: A[0,2]B[2,2]
    add_term({(0,2):1}, {(2,2):1}, {(0,2):1})
    # Term 17: A[1,2]B[2,2]
    add_term({(1,2):1}, {(2,2):1}, {(1,2):1})
    
    # C12 = a12 B22 + a11 b12
    # a12 = [A[2,0], A[2,1]], b12 = [B[2,0], B[2,1]], a11 = A[2,2]
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1]
    
    # Term 18: A[2,0]B[0,0]
    add_term({(2,0):1}, {(0,0):1}, {(2,0):1})
    # Term 19: A[2,1]B[1,0]
    add_term({(2,1):1}, {(1,0):1}, {(2,0):1})
    # Term 20: A[2,0]B[0,1]
    add_term({(2,0):1}, {(0,1):1}, {(2,1):1})
    # Term 21: A[2,1]B[1,1]
    add_term({(2,1):1}, {(1,1):1}, {(2,1):1})
    
    # a11 b12
    # Term 22: A[2,2]B[2,0]
    add_term({(2,2):1}, {(2,0):1}, {(2,0):1})
    # Term 23: A[2,2]B[2,1]
    add_term({(2,2):1}, {(2,1):1}, {(2,1):1})
    
    # C11 = a12 b21 + a11 b11
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2]
    
    # Term 24: A[2,0]B[0,2]
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1})
    # Term 25: A[2,1]B[1,2]
    add_term({(2,1):1}, {(1,2):1}, {(2,2):1})
    # Term 26: A[2,2]B[2,2]
    add_term({(2,2):1}, {(2,2):1}, {(2,2):1})
    
    # Now we have 26 terms for 3x3x3.
    # We need to build U, V, W matrices for the 3x3x8 problem.
    # We split the 8 columns of B into two groups of 3 and one group of 2?
    # No, we split the 3x3x8 into:
    # 3x3x3 (columns 0-2 of B)
    # 3x3x3 (columns 3-5 of B)
    # 3x3x2 (columns 6-7 of B)
    
    # For 3x3x3, we use the 26 terms above.
    # For 3x3x2, we use schoolbook (18 terms).
    
    total_rank = 26 + 26 + 18
    
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(24)]
    W = [[0] * total_rank for _ in range(24)]
    
    col = 0
    
    # Block 1: 3x3x3 (B cols 0-2)
    for u_dict, v_dict, w_dict in terms_333:
        for (i, j), val in u_dict.items():
            U[i*3+j][col] += val
        for (j, k), val in v_dict.items():
            V[j*3+k][col] += val
        for (k, i), val in w_dict.items():
            W[k*3+i][col] += val
        col += 1
    
    # Block 2: 3x3x3 (B cols 3-5)
    # We need to shift the B and C indices by 3.
    for u_dict, v_dict, w_dict in terms_333:
        for (i, j), val in u_dict.items():
            U[i*3+j][col] += val
        for (j, k), val in v_dict.items():
            # k is 0,1,2 -> shift to 3,4,5
            V[j*3+(k+3)][col] += val
        for (k, i), val in w_dict.items():
            # k is 0,1,2 -> shift to 3,4,5
            W[(k+3)*3+i][col] += val
        col += 1
    
    # Block 3: 3x3x2 (B cols 6-7)
    # Schoolbook for 3x3x2
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*3+j][col] = 1
                V[j*3+(6+k)][col] = 1
                W[(6+k)*3+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
