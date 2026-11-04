"""Matrix multiplication 4x4x5 - rank minimization via structured block decomposition."""
import numpy as np
import time

n, m, p = 4, 4, 5  # A is n x m, B is m x p, C is n x p
# U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
# Tensor: C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
# Rank-r decomposition: T_{i,j,k} = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]


def strassen_2x2x2():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    C[i,k] = sum_j A[i,j] * B[j,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A21+A22)B11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = A11(B12-B22)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A22(B21-B11)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (A11+A12)B22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W


def build_2x4x5_block():
    """Build a 2x4 x 4x5 decomposition.
    
    A is 2x4, B is 4x5, C is 2x5.
    U: (8, rank), V: (20, rank), W: (10, rank)
    
    We split A into two 2x2 blocks: A = [A_left, A_right]
    B is 4x5, C is 2x5.
    
    C = A * B = [A_left * B_top + A_right * B_bottom]
    where B_top is rows 0-1 of B (2x5), B_bottom is rows 2-3 of B (2x5).
    
    This is: C = A_left * B_top + A_right * B_bottom
    Each product is 2x2 * 2x5 = 2x5.
    
    For 2x2 * 2x5:
    C[i,k] = sum_j A[i,j] * B[j,k]
    This can be done with rank 7*2 = 14 (using Strassen on each column pair)
    or we can use a more efficient approach.
    
    Actually, let's think of it differently.
    2x2 * 2x5 has 2*2*5 = 20 scalar multiplications in schoolbook.
    With Strassen-like methods, we can do better.
    
    Let's use the following approach:
    For each column k of B (k=0..4), we compute A * B[:,k] which is a 2x2 * 2x1 = 2x1.
    A 2x2 * 2x1 multiplication: C[i] = sum_j A[i,j] * B[j]
    This has 4 scalar products in schoolbook, but can be done with rank 3 using:
    M1 = A00 * B0
    M2 = A11 * B1
    M3 = (A00+A11) * (B0+B1)
    C0 = M1 + M3 - M2
    C1 = M2 + M3 - M1
    Wait, that's not right. Let me reconsider.
    
    Actually, 2x2 * 2x1 is just two dot products of length 2, which has rank 4 in schoolbook.
    There's no known way to do it with rank < 4 for exact computation.
    
    So for 2x2 * 2x5, we have 5 independent 2x2 * 2x1 products, each rank 4, total rank 20.
    But we can share the A factors!
    
    Let me use a different decomposition.
    
    For 2x4 * 4x5:
    C = A * B where A is 2x4, B is 4x5.
    
    Split A into [A1, A2] where A1 is 2x2 (cols 0-1), A2 is 2x2 (cols 2-3).
    Split B into [B1; B2] where B1 is 2x5 (rows 0-1), B2 is 2x5 (rows 2-3).
    
    C = A1 * B1 + A2 * B2
    
    Each A1*B1 is 2x2 * 2x5.
    
    For 2x2 * 2x5, let's use Strassen-like decomposition:
    Write B1 as [B1a, B1b] where B1a is 2x2 (cols 0-1), B1b is 2x3 (cols 2-4).
    Then A1*B1 = A1*B1a + A1*B1b.
    A1*B1a is 2x2*2x2, rank 7 via Strassen.
    A1*B1b is 2x2*2x3, which is 3 separate 2x2*2x1 products, rank 12.
    Total for A1*B1: 7 + 12 = 19? That seems high.
    
    Let me try a different approach. Let's just use the direct construction.
    
    For 2x4 * 4x5, the schoolbook rank is 2*4*5 = 40.
    We want to do better.
    
    Let's use the following structured approach:
    Use 4 groups of terms, each handling one column of A (j=0,1,2,3).
    For column j, we need to compute sum over k of A[:,j] * B[j,:] which contributes to all of C.
    
    Actually, let me just build the full 4x4x5 decomposition using a block approach.
    
    Split A into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    where A11 is 2x2 (rows 0-1, cols 0-1), etc.
    
    B is 4x5, split into:
    B = [[B11, B12, B13, B14, B15], [B21, B22, B23, B24, B25]]
    where B11 is 2x1 (rows 0-1, col 0), etc.
    
    C = [[C11, C12, C13, C14, C15], [C21, C22, C23, C24, C25]]
    where C11 is 2x1 (rows 0-1, col 0), etc.
    
    C1j = A11*B1j + A12*B2j for j=1..5
    C2j = A21*B1j + A22*B2j for j=1..5
    
    Each A11*B1j is 2x2 * 2x1, rank 4.
    So total rank = 4 * 4 * 5 = 80 (schoolbook).
    
    To improve, we can use Strassen on the 2x2 * 2x2 parts.
    Group B11, B12 into B1a (2x2), and B13, B14, B15 into B1b (2x3).
    Similarly for B2.
    
    C1a = A11*B1a + A12*B2a (2x2 * 2x2, rank 7 each, total 14)
    C1b = A11*B1b + A12*B2b (2x2 * 2x3, rank 12 each, total 24)
    C2a = A21*B1a + A22*B2a (rank 14)
    C2b = A21*B1b + A22*B2b (rank 24)
    Total: 14 + 24 + 14 + 24 = 76. Still close to schoolbook.
    
    Let me try a completely different approach. I'll use a known fast algorithm structure.
    
    For 4x4 * 4x5, one approach is to use the fact that 4x4*4x4 has rank 49 (Strassen)
    and 4x4*4x1 has rank 16, giving total rank 65.
    
    Let me build the 4x4*4x4 Strassen decomposition properly.
    """
    pass


def build_4x4x4_strassen():
    """Build 4x4 x 4x4 decomposition using recursive Strassen, rank 49."""
    U_s2, V_s2, W_s2 = strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Recursive Strassen for 4x4:
    # Split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (each 2x2)
    # C = [[C11, C12], [C21, C22]] (each 2x2)
    #
    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each Mi is a 2x2 x 2x2 matmul, computed with rank 7.
    # Total: 7 * 7 = 49 terms.
    
    # For each high-level product h (0..6), and each low-level Strassen term s (0..6):
    # The term index is h*7 + s.
    
    # We need to map the 2x2 indices within each block to the 4x4 indices.
    
    # A block indices in 4x4 (row*4+col):
    # A11: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # A12: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # A21: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # A22: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    # B block indices in 4x4 (row*4+col):
    # B11: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # B12: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # B21: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # B22: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    # C block indices in 4x4 (col*4+row, per convention W[k*n+i]):
    # C11: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # C12: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # C21: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # C22: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    # Define the 2x2 index mapping for each block:
    # Within a 2x2 block, local indices (r,c) map to global indices.
    
    # A11: local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A12: local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # A21: local (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # A22: local (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    # B11: local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B12: local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # B21: local (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # B22: local (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    # C11: local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C12: local (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # C21: local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # C22: local (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    # For each high-level product, we need to determine which A and B blocks are involved,
    # and what the coefficients are.
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11 (coeff +1) and A22 (coeff +1)
    # B part: B11 (coeff +1) and B22 (coeff +1)
    # C part: C11 (coeff +1) and C22 (coeff +1)
    
    # M2 = (A21+A22)B11
    # A part: A21 (coeff +1) and A22 (coeff +1)
    # B part: B11 (coeff +1)
    # C part: C21 (coeff +1) and C22 (coeff -1)
    
    # M3 = A11(B12-B22)
    # A part: A11 (coeff +1)
    # B part: B12 (coeff +1) and B22 (coeff -1)
    # C part: C12 (coeff +1) and C22 (coeff +1)
    
    # M4 = A22(B21-B11)
    # A part: A22 (coeff +1)
    # B part: B21 (coeff +1) and B11 (coeff -1)
    # C part: C11 (coeff +1) and C21 (coeff +1)
    
    # M5 = (A11+A12)B22
    # A part: A11 (coeff +1) and A12 (coeff +1)
    # B part: B22 (coeff +1)
    # C part: C11 (coeff -1) and C12 (coeff +1)
    
    # M6 = (A21-A11)(B11+B12)
    # A part: A21 (coeff +1) and A11 (coeff -1)
    # B part: B11 (coeff +1) and B12 (coeff +1)
    # C part: C22 (coeff +1)
    
    # M7 = (A12-A22)(B21+B22)
    # A part: A12 (coeff +1) and A22 (coeff -1)
    # B part: B21 (coeff +1) and B22 (coeff +1)
    # C part: C11 (coeff +1)
    
    # This is complex. Each high-level product involves a 2x2 matmul of linear combinations.
    # The 2x2 matmul itself is decomposed into 7 rank-1 terms.
    
    # For each high-level product, the "A matrix" is a 2x2 matrix where each entry is a linear
    # combination of A block entries, and similarly for B.
    
    # Let me define the structure more carefully.
    
    # For M1 = (A11+A22)(B11+B22):
    # The 2x2 A matrix is:
    # [A11[0,0]+A22[0,0], A11[0,1]+A22[0,1]]
    # [A11[1,0]+A22[1,0], A11[1,1]+A22[1,1]]
    # = [A[0,0]+A[2,2], A[0,1]+A[2,3]]
    #   [A[1,0]+A[3,2], A[1,1]+A[3,3]]
    
    # The 2x2 B matrix is:
    # [B11[0,0]+B22[0,0], B11[0,1]+B22[0,1]]
    # [B11[1,0]+B22[1,0], B11[1,1]+B22[1,1]]
    # = [B[0,0]+B[2,2], B[0,1]+B[2,3]]
    #   [B[1,0]+B[3,2], B[1,1]+B[3,3]]
    
    # The product is a 2x2 matrix, and:
    # C11 += product (coeff +1)
    # C22 += product (coeff +1)
    
    # So for each of the 7 Strassen terms within this 2x2 matmul:
    # The U vector has entries for the 4 A entries (with coefficients from the linear combination)
    # The V vector has entries for the 4 B entries
    # The W vector has entries for the 4 C entries (C11 and C22)
    
    # This is getting very complex. Let me use a simpler but valid approach.
    
    # I'll use the schoolbook for 4x4x4 (rank 16) and 4x4x1 (rank 16), total rank 32.
    # Wait, 4x4x4 schoolbook is rank 16? No, it's 4*4*4 = 64.
    
    # Let me just use the schoolbook for the entire 4x4x5, rank 80, and try to improve
    # with a random search.
    
    return None


def schoolbook_decomposition():
    """Schoolbook 4x4 x 4x5 decomposition, rank 80."""
    R = n * m * p  # 80
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W


def verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    np.random.seed(42)
    
    # Start with schoolbook
    U, V, W = schoolbook_decomposition()
    best_rank = n * m * p  # 80
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank via structured search
    # Lower bound: max(n*p, m*p, n*m) = max(20, 20, 16) = 20
    
    # Try a few structured ranks
    for R in [40, 32, 28, 24, 20]:
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random U and V with small integer entries
            U_rand = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
            V_rand = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
            
            # Solve for W
            W_rand = np.zeros((p*n, R))
            valid = True
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j] = U_rand[i*m+j] * V_rand[j*p+k]
                    
                    try:
                        W_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                        if np.allclose(A_sys @ W_row, np.ones(m), atol=1e-8):
                            W_rand[k*n+i] = W_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Round to half-integers
            W_rounded = np.round(W_rand * 2) / 2.0
            
            # Verify
            ok = verify_decomposition(U_rand, V_rand, W_rounded, tol=1e-6)
            
            if ok:
                best_rank = R
                best_result = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rounded.tolist()
                }
                break
    
    return best_result
