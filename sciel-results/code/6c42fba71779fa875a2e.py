"""Rewritten program for matrix multiplication 4x4 by 4x7 using Strassen decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    # Strassen decomposition for 2x2 matrices
    # C = AB where A, B, C are 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Define the U, V, W matrices for 2x2 Strassen
    # Indices for A: (i,j) -> i*2+j for i in {0,1}, j in {0,1}
    # Indices for B: (j,k) -> j*2+k for j in {0,1}, k in {0,1}
    # Indices for C: (i,k) -> i*2+k for i in {0,1}, k in {0,1}
    
    # For each of the 7 products, we need U, V, W vectors
    # U is of shape (4, 1), V is of shape (4, 1), W is of shape (4, 1)
    
    # M1: A_part = A00+A11, B_part = B00+B11, C_contrib: C00 += 1, C11 -= 1
    U1 = [1, 0, 1, 0]  # A00=1, A01=0, A10=1, A11=0 -> wait, A indices: A00=0, A01=1, A10=2, A11=3
    # A00+A11 means A[0,0] and A[1,1], so indices 0 and 3
    U1 = [1, 0, 0, 1]
    V1 = [1, 1, 0, 0]  # B00+B11: B[0,0] and B[1,1], indices 0 and 3
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, -1]  # C00 += 1, C11 -= 1: C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=-1
    
    # M2: A_part = A10+A11, B_part = B00, C_contrib: C10 += 1, C11 -= 1
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 0, 1, -1]
    
    # M3: A_part = A00, B_part = B01-B11, C_contrib: C01 += 1, C11 += 1
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 1, 0, 1]
    
    # M4: A_part = A11, B_part = B10-B00, C_contrib: C00 -= 1, C10 += 1
    U4 = [0, 0, 0, 1]
    V4 = [0, 0, 1, 0]  # B10 - B00: B[1,0] - B[0,0]
    V4 = [-1, 0, 1, 0]
    W4 = [-1, 0, 1, 0]
    
    # M5: A_part = A00+A01, B_part = B11, C_contrib: C00 -= 1, C01 += 1
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 1, 0, 0]
    
    # M6: A_part = A10-A00, B_part = B00+B01, C_contrib: none in standard Strassen? 
    # Wait, let me re-check Strassen's algorithm.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    # Note: using 1-based indexing in the formula, but 0-based in code.
    
    # Let me redo with 0-based:
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # C[0,0] += M1, C[1,1] -= M1 (from C[1,1] = M1 - M2 + M3 - M6)
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, -1]
    
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # C[1,0] += M2, C[1,1] -= M2
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 0, 1, -1]
    
    # M3 = A[0,0](B[0,1]-B[1,1])
    # C[0,1] += M3, C[1,1] += M3
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 1, 0, 1]
    
    # M4 = A[1,1](B[1,0]-B[0,0])
    # C[0,0] -= M4, C[1,0] += M4
    U4 = [0, 0, 0, 1]
    V4 = [-1, 0, 1, 0]
    W4 = [-1, 0, 1, 0]
    
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # C[0,0] -= M5, C[0,1] += M5
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 1, 0, 0]
    
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # C[1,1] -= M6
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, -1]
    
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    # C[0,0] += M7, C[1,1] += M7
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 1]
    
    # Now build the 4x4 by 4x7 multiplication
    # We split A (4x4) into four 2x2 blocks: A00, A01, A10, A11
    # We split B (4x7) into four 2x2 blocks and three 2x1 blocks: B00, B01, B02, B10, B11, B12
    # Wait, B is 4x7, so rows 4, cols 7. Split into 2x2 blocks: top-left (0-1,0-1), top-middle (0-1,2-3), top-right (0-1,4-5), and col 6 separately?
    # Better: split B into 2x4 and 2x3 parts.
    # Let's use a simpler approach: treat the 4x7 multiplication as two 4x4 multiplications (for cols 0-3 and 4-7, but 7 is odd).
    # Actually, let's use a block decomposition where we multiply 4x4 by 4x4 (for first 4 cols) and 4x4 by 4x3 (for last 3 cols).
    # But 4x3 is not a power of 2. Let's pad B to 4x8 with zeros, do 4x4 by 4x8, then discard last 4 cols of C.
    # 4x8 = two 4x4 multiplications. Each 4x4 can be done with Strassen recursively? No, Strassen is for 2x2.
    # For 4x4, we can use Strassen's algorithm recursively: 4x4 multiplication can be done with 7 multiplications of 2x2 matrices.
    # So 4x4 by 4x4: rank 7*7=49? No, Strassen for 4x4 uses 7 multiplications of 2x2 matrices, each 2x2 multiplication has rank 7, so total rank 7*7=49.
    # Wait, no. The rank of the 4x4 matrix multiplication tensor is at most 49 using Strassen recursively.
    # Actually, the schoolbook rank for 4x4x4 is 64. Strassen gives rank 49.
    
    # Let's build the 4x4x4 Strassen decomposition first.
    # We need U, V, W for 4x4 multiplication where A is 4x4, B is 4x4, C is 4x4.
    # U shape: (16, rank), V shape: (16, rank), W shape: (16, rank)
    
    # The 4x4 matrix A can be split into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Similarly for B and C.
    
    # Using Strassen recursively, we compute 7 products of 2x2 matrices.
    # But each 2x2 multiplication itself can be done with rank 7.
    # So total rank = 7*7 = 49.
    
    # Let's build this.
    # First, define the 2x2 Strassen decomposition as above.
    # For the 4x4 case, we need to combine these.
    
    # Actually, let's just use the schoolbook method for 4x4x4 and 4x4x3, but that's rank 64+48=112, which is the current score.
    # We need to do better. Let's use Strassen for the 4x4x4 part (rank 49) and schoolbook for 4x4x3 (rank 48), total 97.
    # Or use Strassen for both by padding 4x3 to 4x4 (rank 49), total 98.
    # But maybe we can do 4x4x7 directly with a better decomposition.
    
    # For now, let's implement 4x4x4 Strassen and 4x4x4 Strassen (with padding).
    
    # Build 4x4x4 Strassen decomposition
    # The 2x2 Strassen gives us 7 scalar products for each 2x2 multiplication.
    # For 4x4, we have 7 such 2x2 multiplications, each contributing 7 scalar products.
    
    # Let's define the global U, V, W for 4x4x4.
    # U has 16 rows (4x4), V has 16 rows (4x4), W has 16 rows (4x4).
    # Rank = 49.
    
    # The 7 block products for 4x4 Strassen are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Where A00 is A[0:2,0:2], etc.
    
    # Each Mj is a 2x2 matrix product, decomposed into 7 scalar products.
    # So for each Mj, we have 7 scalar products, giving 7 columns in U, V, W.
    # Total 49 columns.
    
    # Let's build this.
    # First, map 2x2 indices to 4x4 indices.
    # A00: rows 0-1, cols 0-1 -> global indices 0-3
    # A01: rows 0-1, cols 2-3 -> global indices 4-7
    # A10: rows 2-3, cols 0-1 -> global indices 8-11
    # A11: rows 2-3, cols 2-3 -> global indices 12-15
    
    # Similarly for B.
    # C00: rows 0-1, cols 0-1 -> global indices 0-3
    # C01: rows 0-1, cols 2-3 -> global indices 4-7
    # C10: rows 2-3, cols 0-1 -> global indices 8-11
    # C11: rows 2-3, cols 2-3 -> global indices 12-15
    
    # For each of the 7 block products Mj, we have 7 scalar products.
    # For each scalar product, we add to U, V, W.
    
    # Let's define the local 2x2 Strassen U, V, W as lists of 7 vectors.
    U_local = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    V_local = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    W_local = [
        [1, 0, 0, -1],  # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],  # M3
        [-1, 0, 1, 0],  # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, -1],  # M6
        [1, 0, 0, 1]   # M7
    ]
    
    # Now, for the 4x4 case, each Mj corresponds to a block product.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # Block mappings for A:
    # A00 -> rows 0-1, cols 0-1
    # A01 -> rows 0-1, cols 2-3
    # A10 -> rows 2-3, cols 0-1
    # A11 -> rows 2-3, cols 2-3
    
    # For each Mj, the A_part is a linear combination of A blocks.
    # We need to extract the coefficients for each A block.
    
    # Let's define the A_part coefficients for each Mj:
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    A_coeffs = [
        {'A00': 1, 'A11': 1},  # M1
        {'A10': 1, 'A11': 1},  # M2
        {'A00': 1},  # M3
        {'A11': 1},  # M4
        {'A00': 1, 'A01': 1},  # M5
        {'A10': 1, 'A00': -1},  # M6
        {'A01': 1, 'A11': -1}   # M7
    ]
    
    # B_part coefficients:
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    B_coeffs = [
        {'B00': 1, 'B11': 1},  # M1
        {'B00': 1},  # M2
        {'B01': 1, 'B11': -1},  # M3
        {'B10': 1, 'B00': -1},  # M4
        {'B11': 1},  # M5
        {'B00': 1, 'B01': 1},  # M6
        {'B10': 1, 'B11': 1}   # M7
    ]
    
    # C_part coefficients:
    # M1: C00 + 1, C11 - 1
    # M2: C10 + 1, C11 - 1
    # M3: C01 + 1, C11 + 1
    # M4: C00 - 1, C10 + 1
    # M5: C00 - 1, C01 + 1
    # M6: C11 - 1
    # M7: C00 + 1, C11 + 1
    
    C_coeffs = [
        {'C00': 1, 'C11': -1},  # M1
        {'C10': 1, 'C11': -1},  # M2
        {'C01': 1, 'C11': 1},  # M3
        {'C00': -1, 'C10': 1},  # M4
        {'C00': -1, 'C01': 1},  # M5
        {'C11': -1},  # M6
        {'C00': 1, 'C11': 1}   # M7
    ]
    
    # Now build the global U, V, W for 4x4x4.
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Map block names to row/col ranges
    # A blocks:
    A_block_ranges = {
        'A00': (slice(0,2), slice(0,2)),
        'A01': (slice(0,2), slice(2,4)),
        'A10': (slice(2,4), slice(0,2)),
        'A11': (slice(2,4), slice(2,4))
    }
    
    # B blocks:
    B_block_ranges = {
        'B00': (slice(0,2), slice(0,2)),
        'B01': (slice(0,2), slice(2,4)),
        'B10': (slice(2,4), slice(0,2)),
        'B11': (slice(2,4), slice(2,4))
    }
    
    # C blocks:
    C_block_ranges = {
        'C00': (slice(0,2), slice(0,2)),
        'C01': (slice(0,2), slice(2,4)),
        'C10': (slice(2,4), slice(0,2)),
        'C11': (slice(2,4), slice(2,4))
    }
    
    # For each block product j (0-6), and each scalar product l (0-6):
    for j in range(7):
        for l in range(7):
            col = j * 7 + l
            # U contribution
            for block_name, coeff in A_coeffs[j].items():
                row_range, col_range = A_block_ranges[block_name]
                for local_row in range(2):
                    for local_col in range(2):
                        global_row = row_range.start + local_row
                        global_col = col_range.start + local_col
                        global_idx = global_row * 4 + global_col
                        U_444[global_idx, col] += coeff * U_local[l][local_row * 2 + local_col]
            
            # V contribution
            for block_name, coeff in B_coeffs[j].items():
                row_range, col_range = B_block_ranges[block_name]
                for local_row in range(2):
                    for local_col in range(2):
                        global_row = row_range.start + local_row
                        global_col = col_range.start + local_col
                        global_idx = global_row * 4 + global_col
                        V_444[global_idx, col] += coeff * V_local[l][local_row * 2 + local_col]
            
            # W contribution
            for block_name, coeff in C_coeffs[j].items():
                row_range, col_range = C_block_ranges[block_name]
                for local_row in range(2):
                    for local_col in range(2):
                        global_row = row_range.start + local_row
                        global_col = col_range.start + local_col
                        global_idx = global_row * 4 + global_col
                        W_444[global_idx, col] += coeff * W_local[l][local_row * 2 + local_col]
    
    # Now we have 4x4x4 decomposition with rank 49.
    # For 4x4x7, we pad B to 4x8 with zeros, do two 4x4x4 multiplications, and discard last 4 cols of C.
    # But that gives rank 49*2=98, and we only use 7 cols of C, so we can discard the W entries for cols 4-7.
    # Actually, we can just build the 4x4x7 decomposition directly by using the 4x4x4 decomposition for the first 4 cols of B and a separate decomposition for the last 3 cols.
    
    # For simplicity, let's pad B to 4x8 and use two 4x4x4 decompositions.
    # U is the same for both, V is different for each half of B, W is different for each half of C.
    
    # For 4x4x7:
    # U: (16, rank), V: (28, rank), W: (28, rank)
    # rank = 49 (for first 4 cols) + 49 (for last 4 cols, but we only need 3) = 98
    # But we can truncate the last decomposition to only use 3 cols of C.
    
    # Let's build V and W for the padded 4x8 case, then select the first 7 cols of W.
    
    # Actually, let's just build the full 4x4x8 decomposition and then select the first 7 cols of W.
    # But the problem says the output factor follows W[k*n+i,r], so W has shape (p*n, rank) = (7*4, rank) = (28, rank).
    # So we need W to have 28 rows, corresponding to 7 cols of C.
    
    # Let's build the 4x4x7 decomposition by padding B to 4x8, doing two 4x4x4 multiplications, and taking the first 7 cols of C.
    
    # U: (16, 98)
    # V: (32, 98) but we only need 28 rows (for 4x7), so we take the first 28 rows.
    # W: (32, 98) but we only need 28 rows (for 4x7), so we take the first 28 rows.
    
    # Wait, B is 4x7, so V should have 4*7=28 rows. If we pad to 4x8, V has 32 rows.
    # We can just take the first 28 rows of V (corresponding to cols 0-6 of B).
    # Similarly, W has 32 rows, we take the first 28 rows (corresponding to cols 0-6 of C).
    
    # Let's build this.
    U_full = np.zeros((16, 98))
    V_full = np.zeros((32, 98))
    W_full = np.zeros((32, 98))
    
    # First 4 cols of B: use U_444, V_444, W_444
    U_full[:, 0:49] = U_444
    V_full[:, 0:49] = V_444
    W_full[:, 0:49] = W_444
    
    # Last 4 cols of B (padded): use U_444, V_444, W_444 again, but with different V and W indices.
    # For the last 4 cols, V indices are 16-31 (rows 2-3 of the 4x8 matrix? No, B is 4x8, so rows 0-3, cols 0-7.
    # First 4 cols: cols 0-3, indices 0-15 in V.
    # Last 4 cols: cols 4-7, indices 16-31 in V.
    # But we only have 4x7, so we only use cols 0-6, indices 0-27 in V.
    # For the last 3 cols (cols 4-6), we use V indices 16-27.
    # So V_full[:, 49:98] should be V_444 placed in rows 16-31, but we only keep rows 16-27.
    
    # Actually, let's just build V and W for the 4x8 case, then truncate.
    V_full[:, 49:98] = 0
    for i in range(16):
        V_full[i + 16, 49:98] = V_444[i]
    
    W_full[:, 49:98] = 0
    for i in range(16):
        W_full[i + 16, 49:98] = W_444[i]
    
    # Now truncate to 4x7: take first 28 rows of V and W.
    U_447 = U_full
    V_447 = V_full[:28, :]
    W_447 = W_full[:28, :]
    
    # Convert to lists
    U = U_447.tolist()
    V = V_447.tolist()
    W = W_447.tolist()
    
    return {"U": U, "V": V, "W": W}
