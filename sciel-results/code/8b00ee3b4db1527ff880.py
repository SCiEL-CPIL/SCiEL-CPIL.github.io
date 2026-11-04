"""Matrix multiplication 4x5x5 - rank-minimal tensor decomposition."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2.
    Each term is (U_entries, V_entries, W_entries) where entries are dicts
    mapping (row, col) -> value in a 2x2 matrix.
    """
    terms = []
    # P1: (A11+A22)(B11+B22)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # P2: (A21+A22)B11
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1}
    ))
    # P3: A11(B21-B22)
    terms.append((
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1}
    ))
    # P4: (A11+A12)B22
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,1): 1}
    ))
    # P5: (A11-A21)(B11+B12)
    terms.append((
        {(0,0): 1, (1,0): -1},
        {(0,0): 1, (0,1): 1},
        {(0,0): 1, (1,0): 1}
    ))
    # P6: (A12-A22)(B21+B22)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(1,1): 1}
    ))
    # P7: (A11+A22)(B11-B22)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (1,1): -1}
    ))
    return terms

def _build_decomposition(n, m, p):
    """Build a decomposition for n x m x p matrix multiplication.
    
    Uses block Strassen where possible, falling back to schoolbook for
    non-power-of-2 or non-square blocks.
    
    Returns (U, V, W, rank) where U is (n*m, rank), V is (m*p, rank),
    W is (p*n, rank).
    """
    # For now, use schoolbook as baseline: rank = n*m*p
    # We can improve this with block methods.
    
    # Strategy: 
    # 1. If n, m, p are all powers of 2, use recursive Strassen.
    # 2. Otherwise, use a hybrid approach:
    #    - Pad to next power of 2
    #    - Apply Strassen
    #    - Extract the relevant sub-problem
    #    This gives rank = (next_pow2(n) * next_pow2(m) * next_pow2(p)) / (n*m*p) * strassen_rank
    #    But actually padding increases the problem size, so we need to be careful.
    
    # Alternative: Use the fact that we can decompose into blocks.
    # For 4x5x5: 
    #   View as 4x(4+1) x (4+1) = 4x4x4 + 4x4x1 + 4x1x4 + 4x1x1
    #   The 4x4x4 part can use Strassen (rank 49)
    #   The other parts use schoolbook
    #   Total rank = 49 + 16 + 4 + 4 = 73? No, that's not quite right.
    
    # Let me think more carefully.
    # C = A * B where A is 4x5, B is 5x5, C is 4x5.
    # Split A = [A1 | A2] where A1 is 4x4, A2 is 4x1.
    # Split B = [B1; B2] where B1 is 4x5, B2 is 1x5.
    # Then C = A1*B1 + A2*B2.
    # A1*B1 is 4x4 * 4x5 = 4x5. This can be done with rank 49 (Strassen for 4x4) + some extra for the 5th column.
    # Actually, 4x4x5: view as 4x4x4 + 4x4x1. Rank = 49 + 16 = 65.
    # A2*B2 is 4x1 * 1x5 = 4x5. Rank = 4*1*5 = 20 (schoolbook).
    # Total: 65 + 20 = 85.
    
    # Alternatively: Split B differently.
    # B = [B1 | B2] where B1 is 5x4, B2 is 5x1.
    # C = A*B1 + A*B2.
    # A*B1 is 4x5 * 5x4 = 4x4. Rank = 49 (Strassen).
    # A*B2 is 4x5 * 5x1 = 4x1. Rank = 4*5*1 = 20.
    # Total: 49 + 20 = 69.
    
    # Even better: Split both.
    # A = [A1 | A2] (4x4, 4x1)
    # B = [B1; B2] (4x5, 1x5)
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # A1*B1: 4x4 * 4x5 -> 4x5. Rank ~ 49 + 16 = 65? Or use 4x4x4 (rank 49) for the 4x4 part and 4x4x1 (rank 16) for the last column.
    # Actually, let's be more precise.
    
    # Let me just implement a general block decomposition.
    
    # For 4x5x5, the best simple decomposition I can think of:
    # Split B into B_top (4x5) and B_bottom (1x5).
    # C = A * B_top + A * B_bottom
    # A * B_top: 4x5 * 4x5. Hmm, this is still 4x5x5.
    
    # Let me try a different split.
    # Split A into A_top (3x5) and A_bottom (1x5).
    # C_top = A_top * B (3x5 * 5x5 = 3x5). Rank = 3*5*5 = 75 (schoolbook) or better.
    # C_bottom = A_bottom * B (1x5 * 5x5 = 1x5). Rank = 1*5*5 = 25.
    # Total: 75 + 25 = 100. No improvement.
    
    # The key insight is that Strassen helps when we have square blocks.
    # For 4x5x5, we can extract a 4x4x4 sub-problem.
    
    # Let me define the decomposition explicitly for 4x5x5.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, k in 0..4.
    # 
    # Split j into j in 0..3 (top) and j=4 (bottom).
    # C[i,k] = sum_{j=0}^3 A[i,j]*B[j,k] + A[i,4]*B[4,k]
    # 
    # The first term is a 4x4x5 multiplication: A1 (4x4) * B1 (4x5) = C1 (4x5).
    # The second term is a 4x1x5 multiplication: A2 (4x1) * B2 (1x5) = C2 (4x5).
    
    # For A1*B1 (4x4 * 4x5):
    # Split B1 into B1_left (4x4) and B1_right (4x1).
    # C1 = A1*B1_left + A1*B1_right
    # A1*B1_left: 4x4 * 4x4 = 4x4. Use Strassen, rank 49.
    # A1*B1_right: 4x4 * 4x1 = 4x1. Rank = 4*4*1 = 16 (schoolbook).
    # Total for C1: 49 + 16 = 65.
    
    # For A2*B2 (4x1 * 1x5):
    # Rank = 4*1*5 = 20 (schoolbook).
    
    # Total: 65 + 20 = 85.
    
    # Can we do better? 
    # Instead of splitting B1 into 4x4 + 4x1, what if we use a different decomposition?
    # 4x4x5: We can view this as 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65.
    # Or: 4x4x5 = 4x4x2 + 4x4x3? No, that doesn't help.
    
    # What about using the fact that 4x4x5 has rank at most 4*4*5 = 80 (schoolbook),
    # and Strassen gives 49 + 16 = 65?
    
    # Let me see if there's a better decomposition for 4x4x5.
    # 4x4x5: Split the 5 into 4+1.
    # 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65.
    # Alternatively: 4x4x5 = 4x4x5. Is there a Strassen-like decomposition?
    # Not directly, since 5 is not a power of 2.
    
    # So the total rank is 65 + 20 = 85.
    
    # Let me implement this decomposition.
    
    if n == 4 and m == 5 and p == 5:
        return _build_4x5x5()
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _build_4x5x5():
    """Build a rank-85 decomposition for 4x5x5 matrix multiplication.
    
    C = A * B where A is 4x5, B is 5x5, C is 4x5.
    Split: C = A1*B1 + A2*B2 where
      A1 = A[:, 0:4] (4x4), B1 = B[0:4, :] (4x5)
      A2 = A[:, 4:5] (4x1), B2 = B[4:5, :] (1x5)
    
    Further split: A1*B1 = A1*B1_left + A1*B1_right where
      B1_left = B1[:, 0:4] (4x4), B1_right = B1[:, 4:5] (4x1)
    
    So:
      Term 1: A1 * B1_left (4x4 * 4x4 = 4x4), rank 49 (Strassen)
      Term 2: A1 * B1_right (4x4 * 4x1 = 4x1), rank 16 (schoolbook)
      Term 3: A2 * B2 (4x1 * 1x5 = 4x5), rank 20 (schoolbook)
    
    Total rank: 49 + 16 + 20 = 85.
    """
    n, m, p = 4, 5, 5
    total_rank = 85
    
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 85
    
    col = 0
    
    # Term 1: A1 * B1_left (4x4 * 4x4 = 4x4), Strassen rank 49
    # A1 is A[0:4, 0:4], B1_left is B[0:4, 0:4], result goes to C[0:4, 0:4]
    strassen_terms = _strassen_2x2_terms()
    
    # Build 4x4x4 Strassen decomposition
    # For 4x4x4, use two levels of Strassen
    # Top level: 7 terms, each a 2x2 block multiplication
    # Inner level: 7 Strassen terms for each 2x2 multiplication
    # Total: 49 terms
    
    # I'll construct the 49 terms for 4x4x4
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    
    # For the 4x4x4 problem:
    # A4 is 4x4, B4 is 4x4, C4 is 4x4
    # Index: U4[i*4+j], V4[j*4+k], W4[k*4+i]
    
    # Two-level Strassen:
    # Level 1: View 4x4 as 2x2 blocks of 2x2
    # Level 2: Apply Strassen to each 2x2 block multiplication
    
    # Top-level Strassen terms for 4x4:
    # P1: (A00+A11)(B00+B11) -> C00 + C11
    # P2: (A10+A11)B00 -> C10
    # P3: A00(B10-B11) -> C01
    # P4: (A00+A01)B11 -> C01
    # P5: (A00-A10)(B00+B01) -> C00 + C10
    # P6: (A01-A11)(B10+B11) -> C11
    # P7: (A00+A11)(B00-B11) -> C00 + C11
    
    # Where A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # Similarly for B and C.
    
    # For each top-level term, the 2x2 block multiplication is decomposed using
    # the 7 inner Strassen terms.
    
    # I need to carefully construct the U, V, W factors.
    
    # Let me define the top-level terms with their block coefficients.
    # Each top-level term has:
    # - A_coeff: dict mapping (br, bc) -> coeff
    # - B_coeff: dict mapping (br, bc) -> coeff  
    # - C_coeff: dict mapping (br, bc) -> coeff
    
    top_terms = [
        # P1: (A00+A11)(B00+B11) -> C00 + C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2: (A10+A11)B00 -> C10
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # P3: A00(B10-B11) -> C01
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}),
        # P4: (A00+A01)B11 -> C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        # P5: (A00-A10)(B00+B01) -> C00 + C10
        ({(0,0): 1, (1,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (1,0): 1}),
        # P6: (A01-A11)(B10+B11) -> C11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
        # P7: (A00+A11)(B00-B11) -> C00 + C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}, {(0,0): 1, (1,1): -1}),
    ]
    
    # For each top-level term t and inner term s:
    # The U factor is a 16-vector (4x4 matrix flattened).
    # 
    # The key insight: In the two-level Strassen decomposition, the 4x4 matrix
    # is viewed as a 2x2 block matrix. The top-level term specifies how the
    # blocks are combined, and the inner term specifies the Strassen pattern
    # within each 2x2 block multiplication.
    #
    # For the U factor of term (t, s):
    # - The top-level term t has A_coeff mapping block (br, bc) to coefficient a_c.
    # - The inner Strassen term s has u_s mapping (r, c) to value u_val.
    # - The U factor has non-zero entries at positions corresponding to the
    #   A blocks involved in t.
    #
    # Specifically, for each A block (br, bc) with coefficient a_c:
    #   For each (r, c) in {0,1}^2:
    #     U[br*2+r, bc*2+c] = a_c * u_s[br, bc]
    #
    # Wait, I need to be more careful. The inner Strassen term s has u_s[r, c]
    # for r, c in {0,1}. This u_s[r, c] is the coefficient for the (r, c) entry
    # of the 2x2 matrix A_t. A_t[r, c] is the A block at position (r, c).
    #
    # So the effective coefficient for A block (r, c) is A_coeff[(r, c)] * u_s[(r, c)].
    # And this coefficient applies to all 4 entries of the 2x2 block.
    
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    term_idx = 0
    for t_idx, (A_coeff, B_coeff, C_coeff) in enumerate(top_terms):
        for s_idx, (u_s, v_s, w_s) in enumerate(strassen_2x2):
            # Compute U4 for this term
            for (ar, ac), a_c in A_coeff.items():
                u_val = u_s.get((ar, ac), 0)
                if u_val != 0:
                    for dr in range(2):
                        for dc in range(2):
                            U4[(ar * 2 + dr) * 4 + (ac * 2 + dc)][term_idx] = a_c * u_val
            
            # Compute V4 for this term
            for (br, bc), b_c in B_coeff.items():
                v_val = v_s.get((br, bc), 0)
                if v_val != 0:
                    for dr in range(2):
                        for dc in range(2):
                            V4[(br * 2 + dr) * 4 + (bc * 2 + dc)][term_idx] = b_c * v_val
            
            # Compute W4 for this term
            for (cr, cc), c_c in C_coeff.items():
                w_val = w_s.get((cr, cc), 0)
                if w_val != 0:
                    for dr in range(2):
                        for dc in range(2):
                            W4[(cr * 2 + dr) * 4 + (cc * 2 + dc)][term_idx] = c_c * w_val
            
            term_idx += 1
    
    # Now copy U4, V4, W4 into the main U, V, W matrices.
    # U4[i*4+j] corresponds to A[i, j] where i, j in 0..3.
    # In the 4x5x5 problem, A[i, j] for i in 0..3, j in 0..3 is at U[i*5+j].
    # V4[j*4+k] corresponds to B[j, k] where j, k in 0..3.
    # In the 4x5x5 problem, B[j, k] for j in 0..3, k in 0..3 is at V[j*5+k].
    # W4[k*4+i] corresponds to C[k, i] where k, i in 0..3.
    # In the 4x5x5 problem, C[k, i] for k in 0..3, i in 0..3 is at W[k*4+i].
    
    for row in range(16):
        i = row // 4
        j = row % 4
        for t in range(49):
            U[i * 5 + j][col + t] = U4[row][t]
    
    for row in range(16):
        j = row // 4
        k = row % 4
        for t in range(49):
            V[j * 5 + k][col + t] = V4[row][t]
    
    for row in range(16):
        k = row // 4
        i = row % 4
        for t in range(49):
            W[k * 4 + i][col + t] = W4[row][t]
    
    col += 49
    
    # Term 2: A1 * B1_right (4x4 * 4x1 = 4x1), schoolbook rank 16
    # A1 is A[0:4, 0:4], B1_right is B[0:4, 4:5], result goes to C[0:4, 4:5]
    # C[i, 4] = sum_{j=0}^3 A[i, j] * B[j, 4]
    # Rank = 4*4*1 = 16
    
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # C[4, i] in flat indexing? No, W[k*n+i] = W[4*4+i]
            col += 1
    
    # Term 3: A2 * B2 (4x1 * 1x5 = 4x5), schoolbook rank 20
    # A2 is A[0:4, 4:5], B2 is B[4:5, 0:5], result goes to C[0:4, 0:5]
    # C[i, k] += A[i, 4] * B[4, k]
    # Rank = 4*1*5 = 20
    
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return U, V, W, total_rank

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n, m, p = 4, 5, 5
    
    U, V, W, rank = _build_decomposition(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen, but just in case
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
