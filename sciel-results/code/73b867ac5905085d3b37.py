import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = 3, 5, 7

    # Identify 2x2x2 blocks: pairs of rows (i), columns (j), output indices (k)
    # A block is defined by (i1,i2), (j1,j2), (k1,k2)
    # We need to partition {0,1,2} for i, {0,1,2,3,4} for j, {0,1,2,3,4,5,6} for k
    # into pairs and singletons such that we maximize the number of 2x2x2 blocks.
    
    # For n=3: pairs (0,1), singleton (2)
    # For m=5: pairs (0,1), (2,3), singleton (4)
    # For p=7: pairs (0,1), (2,3), (4,5), singleton (6)
    
    i_pairs = [(0, 1)]
    i_singles = [2]
    
    j_pairs = [(0, 1), (2, 3)]
    j_singles = [4]
    
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    k_singles = [6]
    
    # Build the decomposition
    # We will create U, V, W as lists of lists (or numpy arrays) with total rank
    # Rank = (#blocks * 7) + (#remaining_terms * 1)
    # #blocks = len(i_pairs) * len(j_pairs) * len(k_pairs) = 1 * 2 * 3 = 6
    # #remaining_terms = total_terms - (#blocks * 8) = 105 - 48 = 57
    # Total rank = 6*7 + 57 = 42 + 57 = 99
    
    total_terms = n * m * p
    num_blocks = len(i_pairs) * len(j_pairs) * len(k_pairs)
    num_remaining = total_terms - num_blocks * 8
    rank = num_blocks * 7 + num_remaining
    
    U = []
    V = []
    W = []
    
    col = 0
    
    def add_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # Strassen decomposition for 2x2x2
        # Indices in block:
        # U rows: i1*m + j1, i1*m + j2, i2*m + j1, i2*m + j2
        # V rows: j1*p + k1, j1*p + k2, j2*p + k1, j2*p + k2
        # W rows: k1*n + i1, k1*n + i2, k2*n + i1, k2*n + i2
        
        # Strassen factors (rank 7)
        # Let's use the standard Strassen 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B21 - B22)
        # M4 = A22 (B12 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # In tensor terms:
        # U factors correspond to A coefficients
        # V factors correspond to B coefficients
        # W factors correspond to C coefficients
        
        # We need to map these to the specific rows/cols of U, V, W.
        
        # U rows for this block:
        # u0 = i1*m + j1 (A11)
        # u1 = i1*m + j2 (A12)
        # u2 = i2*m + j1 (A21)
        # u3 = i2*m + j2 (A22)
        
        # V rows for this block:
        # v0 = j1*p + k1 (B11)
        # v1 = j1*p + k2 (B12)
        # v2 = j2*p + k1 (B21)
        # v3 = j2*p + k2 (B22)
        
        # W rows for this block:
        # w0 = k1*n + i1 (C11)
        # w1 = k1*n + i2 (C21)
        # w2 = k2*n + i1 (C12)
        # w3 = k2*n + i2 (C22)
        
        # Strassen products:
        # 1: (A11+A22)(B11+B22) -> M1
        #    U: u0=1, u3=1; V: v0=1, v3=1; W: w0=1, w1=1, w2=1, w3=1 (sum of all C)
        #    Wait, C = [[M1-M4-M5+M7, M5+M6], [M2-M7, M1+M3-M2-M6]]
        #    So M1 contributes to all C11, C12, C21, C22 with coefficient +1
        # 2: (A21+A22)B11 -> M2
        #    U: u2=1, u3=1; V: v0=1; W: C21 -= M2, C22 += M2 (coeff -1, +1)
        #    Actually C21 = M2 - M7, C22 = M1+M3-M2-M6.
        #    So M2 contributes -1 to C21, +1 to C22.
        # 3: A11(B21-B22) -> M3
        #    U: u0=1; V: v2=1, v3=-1; W: C22 += M3
        # 4: A22(B12-B11) -> M4
        #    U: u3=1; V: v1=1, v0=-1; W: C11 -= M4
        # 5: (A11+A12)B22 -> M5
        #    U: u0=1, u1=1; V: v3=1; W: C11 -= M5, C12 += M5
        # 6: (A21-A11)(B11+B12) -> M6
        #    U: u2=1, u0=-1; V: v0=1, v1=1; W: C12 += M6, C22 -= M6
        # 7: (A12-A22)(B21+B22) -> M7
        #    U: u1=1, u3=-1; V: v2=1, v3=1; W: C11 += M7, C21 -= M7
        
        # Let's construct the 7 rank-1 tensors for this block.
        # Each rank-1 tensor is (U_col, V_col, W_col)
        
        # We'll add 7 columns to U, V, W.
        
        # Block 1
        u = [0]*len(U) if U else []
        v = [0]*len(V) if V else []
        w = [0]*len(W) if W else []
        
        # It's easier to build the full matrices at once or append.
        # Let's initialize U, V, W as lists of lists of zeros with the final shape.
        # But we don't know the exact number of columns until we finish.
        # We know rank = 99.
        
        # So let's just return and build properly.
        
    # Re-initialize U, V, W with known rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal col
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val
    
    # Helper to add a Strassen block
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        
        u0 = i1 * m + j1
        u1 = i1 * m + j2
        u2 = i2 * m + j1
        u3 = i2 * m + j2
        
        v0 = j1 * p + k1
        v1 = j1 * p + k2
        v2 = j2 * p + k1
        v3 = j2 * p + k2
        
        w0 = k1 * n + i1
        w1 = k1 * n + i2
        w2 = k2 * n + i1
        w3 = k2 * n + i2
        
        # M1: (A11+A22)(B11+B22)
        # U: u0=1, u3=1
        # V: v0=1, v3=1
        # W: w0=1, w1=1, w2=1, w3=1
        add_term(u0, v0, w0, 1, 1, 1)
        # But add_term only adds to one row in each matrix.
        # We need to add to multiple rows in U for the same column.
        # My add_term signature is (u_row, v_row, w_row, u_val, v_val, w_val)
        # This implies a single (u_row, v_row, w_row) triplet gets the value.
        # But Strassen terms involve sums of rows in U/V/W.
        # E.g., M1 has U vector with 1s at u0 and u3.
        # So we need to modify add_term to allow setting multiple rows,
        # or just directly set U[u0][col] += 1, U[u3][col] += 1, etc.
        
        # Let's rewrite the helper to be more flexible or just inline the updates.
        
        # M1
        U[u0][col] += 1; U[u3][col] += 1
        V[v0][col] += 1; V[v3][col] += 1
        W[w0][col] += 1; W[w1][col] += 1; W[w2][col] += 1; W[w3][col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        # U: u2=1, u3=1
        # V: v0=1
        # W: w1=-1, w3=1  (C21 = M2-M7, C22 = ...-M2-... -> wait, C22 has -M2? No, C22 = M1+M3-M2-M6. So -M2.)
        # Let's re-verify Strassen formulas.
        # C11 = M1 - M4 - M5 + M7
        # C12 = M5 + M6
        # C21 = M2 - M7
        # C22 = M1 + M3 - M2 - M6
        
        # M2 contributes:
        # To C21: +1
        # To C22: -1
        U[u2][col] += 1; U[u3][col] += 1
        V[v0][col] += 1
        W[w1][col] += 1; W[w3][col] += -1
        col += 1
        
        # M3: A11(B21-B22)
        # U: u0=1
        # V: v2=1, v3=-1
        # W: C22: +1
        U[u0][col] += 1
        V[v2][col] += 1; V[v3][col] += -1
        W[w3][col] += 1
        col += 1
        
        # M4: A22(B12-B11)
        # U: u3=1
        # V: v1=1, v0=-1
        # W: C11: -1
        U[u3][col] += 1
        V[v1][col] += 1; V[v0][col] += -1
        W[w0][col] += -1
        col += 1
        
        # M5: (A11+A12)B22
        # U: u0=1, u1=1
        # V: v3=1
        # W: C11: -1, C12: +1
        U[u0][col] += 1; U[u1][col] += 1
        V[v3][col] += 1
        W[w0][col] += -1; W[w2][col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        # U: u2=1, u0=-1
        # V: v0=1, v1=1
        # W: C12: +1, C22: -1
        U[u2][col] += 1; U[u0][col] += -1
        V[v0][col] += 1; V[v1][col] += 1
        W[w2][col] += 1; W[w3][col] += -1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        # U: u1=1, u3=-1
        # V: v2=1, v3=1
        # W: C11: +1, C21: -1
        U[u1][col] += 1; U[u3][col] += -1
        V[v2][col] += 1; V[v3][col] += 1
        W[w0][col] += 1; W[w1][col] += -1
        col += 1
        
    # Add all Strassen blocks
    for i_pair in i_pairs:
        for j_pair in j_pairs:
            for k_pair in k_pairs:
                if time.time() > deadline:
                    break
                i1, i2 = i_pair
                j1, j2 = j_pair
                k1, k2 = k_pair
                add_strassen_block(i1, i2, j1, j2, k1, k2)
    
    # Now add the remaining terms (schoolbook)
    # These are terms (i, j, k) where NOT (i in i_pair, j in j_pair, k in k_pair)
    # Actually, the "remaining" terms are all (i,j,k) that are not covered by any block.
    # A term (i,j,k) is covered if i is in some i_pair, j is in some j_pair, k is in some k_pair
    # AND specifically i,j,k are in the *same* block? No.
    # The blocks partition the tensor space.
    # The term (i,j,k) is in a block if i in {i1,i2}, j in {j1,j2}, k in {k1,k2} for some block.
    # Since the pairs are disjoint, each (i,j,k) is in at most one block.
    # We need to identify which (i,j,k) are NOT in any block.
    
    # Let's mark covered indices
    covered = set()
    for i_pair in i_pairs:
        for j_pair in j_pairs:
            for k_pair in k_pairs:
                i1, i2 = i_pair
                j1, j2 = j_pair
                k1, k2 = k_pair
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Verify col matches rank
    assert col == rank, f"Rank mismatch: col={col}, rank={rank}"
    
    # Convert to lists of floats (or ints)
    # The problem says "integer/half-integer entries". Our entries are integers.
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
