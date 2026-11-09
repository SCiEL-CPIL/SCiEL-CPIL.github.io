"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W (each 4x7) for 2x2 * 2x2 matrix multiplication using Strassen."""
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A layout: A11=idx0, A12=idx1, A21=idx2, A22=idx3
    # B layout: B11=idx0, B12=idx1, B21=idx2, B22=idx3
    # C layout: C11=idx0, C12=idx1, C21=idx2, C22=idx3

    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1: (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M1 contributes to: C11(+1), C22(+1)
    W[0, 0] = 1; W[3, 0] = 1

    # M2: (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M2 contributes to: C21(+1), C22(-1)
    W[2, 1] = 1; W[3, 1] = -1

    # M3: A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M3 contributes to: C12(+1), C22(+1)
    W[1, 2] = 1; W[3, 2] = 1

    # M4: A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M4 contributes to: C11(+1), C21(+1)
    W[0, 3] = 1; W[2, 3] = 1

    # M5: (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M5 contributes to: C12(+1), C11(-1)
    W[1, 4] = 1; W[0, 4] = -1

    # M6: (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M6 contributes to: C22(+1)
    W[3, 5] = 1

    # M7: (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # M7 contributes to: C11(+1)
    W[0, 6] = 1

    return U, V, W


def _lift_2x2_to_4x4(U2, V2, W2):
    """
    Lift a 2x2*2x2 decomposition (U2: 4x7, V2: 4x7, W2: 4x7) to 4x4*4x4.
    Uses Strassen recursion: 4x4 = 2x2 blocks of 2x2.
    Each 2x2 sub-problem uses the 2x2 decomposition, giving 7*7=49 terms.
    But we can do better: the 4x4 Strassen has rank 7^2 = 49? No.
    
    Actually, for 4x4*4x4, we apply Strassen to 2x2 blocks where each block
    multiplication is 2x2*2x2. Each of the 7 Strassen products involves
    multiplying 2x2 matrices, which needs rank 7 each. So total rank = 7*7 = 49.
    
    But wait - we can use the 2x2 decomposition directly for each sub-block.
    The 4x4 matrix A is split into 4 blocks of 2x2:
    A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3),
    A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    
    Similarly for B and C.
    
    The Strassen algorithm for 4x4 uses 7 products of 2x2 matrices.
    Each 2x2 product uses the rank-7 decomposition.
    Total rank = 7 * 7 = 49.
    """
    # U2 shape: (4, 7) - 4 entries for 2x2 A, 7 rank terms
    # We need U4 shape: (16, 49) - 16 entries for 4x4 A, 49 rank terms
    
    n2 = 2  # block size
    n4 = 4  # full size
    rank2 = U2.shape[1]  # 7
    rank4 = rank2 * rank2  # 49
    
    U4 = np.zeros((n4 * n4, rank4))
    V4 = np.zeros((n4 * n4, rank4))
    W4 = np.zeros((n4 * n4, rank4))
    
    # Block offsets in the 4x4 layout
    # A11: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    
    # For a 2x2 block at (br, bc) in the 4x4, the flat index is:
    # (br*2 + lr) * 4 + (bc*2 + lc) where lr,lc in {0,1}
    
    def block_offset(br, bc):
        """Return the 4 flat indices for the 2x2 block at (br, bc) in 4x4."""
        idxs = []
        for lr in range(2):
            for lc in range(2):
                idxs.append((br * 2 + lr) * 4 + (bc * 2 + lc))
        return idxs
    
    # Strassen products for 4x4 (same structure as 2x2 but with 2x2 blocks):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each product involves multiplying two 2x2 matrices, using the rank-7 decomposition.
    # For product s (s=0..6), the A-side is a linear combination of blocks,
    # and the B-side is a linear combination of blocks.
    
    # Define the 7 Strassen products with their block combinations
    # A-side coefficients: dict from block (br,bc) to coefficient
    # B-side coefficients: dict from block (br,bc) to coefficient
    
    strassen_products = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # C_coeffs: dict from C block to coefficient
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),  # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),  # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),  # M7
    ]
    
    for s, (A_coeffs, B_coeffs, C_coeffs) in enumerate(strassen_products):
        for r2 in range(rank2):
            col4 = s * rank2 + r2
            # A-side: for each block (br,bc) with coefficient c_a
            for (abr, abc), c_a in A_coeffs.items():
                a_idxs = block_offset(abr, abc)
                for li, a_idx in enumerate(a_idxs):
                    U4[a_idx, col4] += c_a * U2[li, r2]
            
            # B-side
            for (bbr, bbc), c_b in B_coeffs.items():
                b_idxs = block_offset(bbr, bbc)
                for li, b_idx in enumerate(b_idxs):
                    V4[b_idx, col4] += c_b * V2[li, r2]
            
            # C-side
            for (cbr, cbc), c_c in C_coeffs.items():
                c_idxs = block_offset(cbr, cbc)
                for li, c_idx in enumerate(c_idxs):
                    W4[c_idx, col4] += c_c * W2[li, r2]
    
    return U4, V4, W4


def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    T = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T - expected))
    return max_err < tol, max_err


def _build_4x5x5():
    """
    Build decomposition for 4x5 * 5x5.
    
    Strategy:
    - Split A (4x5) into A_left (4x4) and A_right (4x1)
    - Split B (5x5) into B_top (5x4) and B_bot (5x1)
    - C = A*B = A_left * B_top + A_right * B_bot (both 4x4 and 4x1, but actually:
      C = A_left * B[:,0:4] + A_right * B[:,4:5]
      where A_left is 4x4 (cols 0-3), A_right is 4x1 (col 4)
      B[:,0:4] is 5x4, B[:,4:5] is 5x1
    
    Wait, I need to be more careful. A is 4x5, B is 5x5.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split the sum over j into j=0..3 and j=4:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    Part 1: A1 (4x4) * B1 (4x5) -> C1 (4x5) [using j=0..3]
    Part 2: A2 (4x1) * B2 (1x5) -> C2 (4x5) [using j=4]
    
    For Part 1: A1 is 4x4, B1 is 4x5. Split B1 into B1_left (4x4) and B1_right (4x1).
    C1 = A1 * B1_left + A1 * B1_right
    - A1 * B1_left: 4x4 * 4x4 -> use Strassen (rank 49)
    - A1 * B1_right: 4x4 * 4x1 -> schoolbook (rank 16)
    
    For Part 2: A2 is 4x1, B2 is 1x5.
    C2[i,k] = A2[i] * B2[k] = A2[i] * B[4,k]
    This is an outer product, rank 1! But wait, in the tensor decomposition framework:
    U has shape (n*m, rank) = (4*5, rank), V has shape (m*p, rank) = (5*5, rank), W has shape (p*n, rank) = (5*4, rank)
    
    For the outer product part: C[i,k] = A[i,4] * B[4,k]
    U[i*5+4, r] = 1, V[4*5+k, r] = 1, W[k*4+i, r] = 1 for one column r.
    That's rank 1.
    
    So total rank = 49 + 16 + 1 = 66.
    
    Hmm, that's better than 100 but not great. Let me reconsider.
    
    Actually, for Part 1: A1 (4x4) * B1 (4x5)
    I can split B1 into 4x4 + 4x1.
    A1 * B1_left (4x4*4x4): Strassen rank 49
    A1 * B1_right (4x4*4x1): This is 4x4 times 4x1, which is like 4 independent (4x1) vectors.
    C[i,k] for k=4: sum_{j=0}^{3} A1[i,j] * B1[j,4]
    This is a 4x1 result. Schoolbook rank = 4*4*1 = 16.
    
    Can I do better for 4x4 * 4x1? It's equivalent to computing 4 dot products of length 4.
    Each dot product needs rank 4 (schoolbook), so 4*4=16 total. I don't think we can do better
    for the outer product structure unless we use the full 4x4 structure.
    
    Actually, 4x4 * 4x1 is the same as 4x4 * 4x1. The minimum rank for this is... 
    It's the rank of the tensor [4,4,1]. This is just 4 independent rank-4 tensors, so rank 16.
    
    Total: 49 + 16 + 1 = 66.
    
    Let me try a different split. What if I don't split the 5th column of A separately?
    
    Alternative: Use the full 4x5 * 5x5 directly with a block approach.
    Split A into 2x5 + 2x5 (top 2 rows, bottom 2 rows).
    Split B into 5x4 + 5x1 (first 4 cols, last col).
    
    C = [A_top * B; A_bot * B] = [A_top * B_left + A_top * B_right; A_bot * B_left + A_bot * B_right]
    
    A_top * B_left: 2x5 * 5x4 -> 2x4
    A_bot * B_left: 2x5 * 5x4 -> 2x4
    A_top * B_right: 2x5 * 5x1 -> 2x1
    A_bot * B_right: 2x5 * 5x1 -> 2x1
    
    For 2x5 * 5x4: Can I use Strassen? Not directly since it's not square.
    But I can split the 5 into 4+1:
    2x5 * 5x4 = 2x4 * 4x4 + 2x1 * 1x4
    2x4 * 4x4: schoolbook rank = 2*4*4 = 32, or use Strassen on the 4x4 part?
    Actually, 2x4 * 4x4: the 4x4 can use Strassen (rank 49 for 4x4*4x4), but we only need
    2x4 * 4x4 which is like 2 rows of the 4x4*4x4 result.
    
    This is getting complicated. Let me just go with the 66-rank solution and verify it.
    """
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Part 1: A1 (4x4, cols 0-3) * B1_left (4x4, rows 0-3, cols 0-3) -> C11 (4x4)
    # Use Strassen: rank 49
    U2, V2, W2 = _strassen_2x2()
    U4, V4, W4 = _lift_2x2_to_4x4(U2, V2, W2)
    
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # These are for 4x4 * 4x4 with standard row-major layout.
    
    # Map to our 4x5 * 5x5 problem:
    # A1[i,j] for i in 0..3, j in 0..3 -> flat index i*5+j in U
    # B1[j,k] for j in 0..3, k in 0..3 -> flat index j*5+k in V
    # C1[i,k] for i in 0..3, k in 0..3 -> flat index k*4+i in W
    
    # U4 layout: flat index i*4+j for 4x4 A
    # We need to map i*4+j -> i*5+j (since A is 4x5, first 4 cols)
    
    U_part1 = np.zeros((nm, 49))
    V_part1 = np.zeros((mp, 49))
    W_part1 = np.zeros((pn, 49))
    
    for i in range(4):
        for j in range(4):
            src_idx = i * 4 + j
            dst_idx = i * 5 + j
            U_part1[dst_idx, :] = U4[src_idx, :]
    
    for j in range(4):
        for k in range(4):
            src_idx = j * 4 + k
            dst_idx = j * 5 + k
            V_part1[dst_idx, :] = V4[src_idx, :]
    
    for k in range(4):
        for i in range(4):
            src_idx = k * 4 + i
            dst_idx = k * 4 + i  # W is 5x4, k*4+i is already correct
            W_part1[dst_idx, :] = W4[src_idx, :]
    
    # Part 2: A1 (4x4, cols 0-3) * B1_right (4x1, rows 0-3, col 4) -> C12 (4x1)
    # C12[i,4] = sum_{j=0}^{3} A1[i,j] * B1[j,4]
    # Schoolbook: rank 4*4*1 = 16
    U_part2 = np.zeros((nm, 16))
    V_part2 = np.zeros((mp, 16))
    W_part2 = np.zeros((pn, 16))
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            U_part2[i * 5 + j, col2] = 1
            V_part2[j * 5 + 4, col2] = 1
            W_part2[4 * 4 + i, col2] = 1  # k=4, so k*4+i = 16+i
            col2 += 1
    
    # Part 3: A2 (4x1, col 4) * B2 (1x5, row 4) -> C2 (4x5)
    # C2[i,k] = A[i,4] * B[4,k]
    # This is an outer product, rank 1.
    U_part3 = np.zeros((nm, 1))
    V_part3 = np.zeros((mp, 1))
    W_part3 = np.zeros((pn, 1))
    
    # For each (i,k), we need a separate term? No, it's a single rank-1 term:
    # C[i,k] = A[i,4] * B[4,k]
    # U[i*5+4, 0] = 1 for all i? No, that would make U have all 1s in col 4.
    # Actually, the outer product A[:,4] * B[4,:] is rank 1 in the matrix sense,
    # but in the tensor decomposition, we need:
    # sum_r U[i*5+4, r] * V[4*5+k, r] * W[k*4+i, r] = A[i,4] * B[4,k]
    
    # For a single rank-1 term: U[i*5+4, 0] * V[4*5+k, 0] * W[k*4+i, 0] = A[i,4] * B[4,k]
    # This requires U[i*5+4, 0] = A[i,4], V[4*5+k, 0] = B[4,k], but A and B are variables!
    # We can't have U depend on A.
    
    # The correct interpretation: the tensor decomposition is fixed (independent of A, B).
    # For the outer product C[i,k] = A[i,4] * B[4,k], we need:
    # T[i*5+4, 4*5+k, k*4+i] = 1 for all i,k
    # And T[other] = 0 for this part.
    
    # This is a rank-1 tensor: U[i*5+4, 0] = 1, V[4*5+k, 0] = 1, W[k*4+i, 0] = 1
    # But wait, this gives T[i*5+4, 4*5+k, k*4+i] = 1*1*1 = 1 for all i,k.
    # That's exactly what we want! One column suffices.
    
    for i in range(4):
        U_part3[i * 5 + 4, 0] = 1
    for k in range(5):
        V_part3[4 * 5 + k, 0] = 1
    for k in range(5):
        for i in range(4):
            W_part3[k * 4 + i, 0] = 1
    
    # Combine all parts
    total_rank = 49 + 16 + 1
    U = np.hstack([U_part1, U_part2, U_part3])
    V = np.hstack([V_part1, V_part2, V_part3])
    W = np.hstack([W_part1, W_part2, W_part3])
    
    return U, V, W


def _try_improve(U, V, W, n, m, p, max_time=60):
    """Try to reduce the rank by removing redundant columns."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Build expected tensor
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try removing columns one by one (greedy)
    active = list(range(rank))
    current_rank = rank
    
    while current_rank > 25 and (time.time() - start_time) < max_time:
        removed = False
        for idx in range(len(active)):
            if (time.time() - start_time) > max_time:
                break
            
            # Try removing active[idx]
            test_active = active[:idx] + active[idx+1:]
            if not test_active:
                break
            
            U_test = U[:, test_active]
            V_test = V[:, test_active]
            W_test = W[:, test_active]
            
            T = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            err = np.max(np.abs(T - expected))
            
            if err < 1e-9:
                active = test_active
                current_rank = len(active)
                removed = True
                break
        
        if not removed:
            break
    
    if current_rank < rank:
        return U[:, active], V[:, active], W[:, active]
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    
    # Build the 66-rank decomposition
    U, V, W = _build_4x5x5()
    
    # Verify
    valid, err = _verify(U, V, W, n, m, p)
    
    if not valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook_nm_p(n, m, p)
    
    # Try to improve by removing redundant columns
    if time.time() < deadline - 10:
        U, V, W = _try_improve(U, V, W, n, m, p, max_time=min(60, deadline - time.time() - 10))
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U.tolist()]
    V_list = [[float(x) for x in row] for row in V.tolist()]
    W_list = [[float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
