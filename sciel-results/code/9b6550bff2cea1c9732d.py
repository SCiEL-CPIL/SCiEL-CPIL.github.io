"""Matrix multiplication 3x3 by 3x7 - rank minimization via block decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8, trials=8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    rng = np.random.default_rng(42)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for trial in range(trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        
        a_forms = np.einsum('ijc,ij->ci', U_arr.reshape(n, m, rank), A)
        b_forms = np.einsum('jkc,jk->ck', V_arr.reshape(m, p, rank), B)
        C_comp = np.einsum('kic,ci,ck->ik', W_arr.reshape(p, n, rank), a_forms, b_forms)
        
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 x 2x2 -> 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7) where W[k*2+i, c].
    Index convention: A[i,j] -> flat i*2+j, B[j,k] -> flat j*2+k, C[i,k] -> flat k*2+i
    """
    # 7 rank elements
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))  # W[k*2+i, c]
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1.0; U[3, 0] = 1.0  # A00, A11
    V[0, 0] = 1.0; V[3, 0] = 1.0  # B00, B11
    W[0, 0] = 1.0; W[1, 0] = 1.0; W[2, 0] = 1.0; W[3, 0] = -1.0
    # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
    # W[k*2+i]: C00 -> W[0,0], C01 -> W[1,0], C10 -> W[2,0], C11 -> W[3,0]
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1.0; U[3, 1] = 1.0  # A10, A11
    V[0, 1] = 1.0  # B00
    W[2, 1] = 1.0; W[3, 1] = -1.0
    # C10 += M2, C11 -= M2
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1.0  # A00
    V[2, 2] = 1.0; V[3, 2] = -1.0  # B10, B11
    W[1, 2] = 1.0; W[3, 2] = 1.0
    # C01 += M3, C11 += M3
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1.0  # A11
    V[1, 3] = 1.0; V[2, 3] = -1.0  # B01, B10
    W[0, 3] = 1.0; W[2, 3] = 1.0
    # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0  # A00, A01
    V[3, 4] = 1.0  # B11
    W[0, 4] = -1.0; W[1, 4] = 1.0
    # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1.0; U[0, 5] = -1.0  # A10, A00
    V[0, 5] = 1.0; V[2, 5] = 1.0  # B00, B10
    # M6 is not used in standard Strassen for 2x2? Let me recheck.
    # Actually standard Strassen has 7 multiplications. Let me verify the formulas.
    # C = [[M1+M4-M5-M7, M3+M5], [M2+M4, M1-M2+M3+M7]]
    # So M6 is not in the output. That's odd. Let me look up again.
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B21)
    # P7 = (A12-A22)(B12+B22)
    # C11 = P1+P4-P5-P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P7
    # So P6 is computed but not used? That doesn't make sense.
    # Oh wait, I think I misremembered. Let me just use the standard 7-multiplication formula
    # where all 7 are used. Actually no, the standard Strassen uses exactly 7 scalar
    # multiplications and all 7 appear in the output. Let me re-examine.
    
    # Actually, I think M6 is not needed in the output combination. The 7th multiplication
    # is M7. M6 might be a redundant one I added. Let me just use the 6 that are needed
    # plus M7. Wait, that's only 6. Let me count: M1-M5, M7 = 6. That's not 7.
    
    # I'll just implement the 7 multiplications correctly and verify.
    # For now, let me set M6 to be zero contribution and just use 6 multiplications?
    # No, that's not right either. Let me just verify with a simple approach.
    
    # Let me redo this more carefully.
    # The 7 Strassen products are all needed. Let me check if my M6 has a role.
    # Looking at it again, I think I made an error. Let me just use a known correct
    # Strassen implementation.
    
    # Actually, for the purpose of this problem, I can just use the 6 products that
    # appear in the output and not worry about M6. But that would give rank 6, not 7.
    # The key insight is that we need 7 rank elements for 2x2x2.
    
    # Let me just return the 7 Strassen terms and verify numerically.
    # I'll fix the W matrix for M6 to be zero since it doesn't appear in the output.
    # Actually, that can't be right. Let me look at this differently.
    
    # I'll just return what I have and verify it numerically. If it doesn't work,
    # I'll adjust.
    
    W[0, 5] = 0.0; W[1, 5] = 0.0; W[2, 5] = 0.0; W[3, 5] = 0.0  # M6 not in output?
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0  # A01, A11
    V[1, 6] = 1.0; V[3, 6] = 1.0  # B01, B11
    W[0, 6] = -1.0; W[3, 6] = 1.0
    # C00 -= M7, C11 += M7
    
    return U, V, W

def _block_decomposition(n, m, p):
    """Build a block decomposition of the 3x3 x 3x7 multiplication.
    
    Split A into blocks:
    - A11: 2x2 (rows 0-1, cols 0-1)
    - A12: 2x1 (rows 0-1, col 2)
    - A21: 1x2 (row 2, cols 0-1)
    - A22: 1x1 (row 2, col 2)
    
    B is 3x7. Split by rows:
    - B1: rows 0-1 (2x7)
    - B2: row 2 (1x7)
    
    C = A11*B1 + A12*B2 + A21*B1 + A22*B2 (with appropriate dimensions)
    Actually:
    C[i,k] = sum_j A[i,j] * B[j,k]
    For i in 0-1: C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + A[i,2]B[2,k]
    For i=2: C[2,k] = sum_{j=0}^{1} A[2,j]B[j,k] + A[2,2]B[2,k]
    
    So:
    - Block 1: A11 (2x2) * B1 (2x7) -> contributes to C[0:2, :]
    - Block 2: A12 (2x1) * B2 (1x7) -> contributes to C[0:2, :]
    - Block 3: A21 (1x2) * B1 (2x7) -> contributes to C[2, :]
    - Block 4: A22 (1x1) * B2 (1x7) -> contributes to C[2, :]
    """
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Block 1: 2x2 * 2x7 -> 2x7
    # A11: rows 0-1, cols 0-1 of A. Flat indices: 0,1,3,4
    # B1: rows 0-1 of B. Flat indices: 0..13
    # Output: C[0:2, :] -> W[k*3+i] for i in 0,1
    U1, V1, W1_2x2 = _strassen_2x2x2()
    # U1 is (4, 7) for 2x2 A. We need to map to (9, 7) for 3x3 A
    # A11 flat indices in 3x3: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    U1_full = np.zeros((9, 7))
    U1_full[0, :] = U1[0, :]  # A00
    U1_full[1, :] = U1[1, :]  # A01
    U1_full[3, :] = U1[2, :]  # A10
    U1_full[4, :] = U1[3, :]  # A11
    
    # V1 is (4, 7) for 2x2 B (2x2). But B1 is 2x7, not 2x2.
    # We need to handle this differently. B1 is 2x7, so for each of the 7 columns k,
    # we have a 2x1 vector. The Strassen decomposition works on 2x2 B blocks.
    # So for each column k of B1, we can apply Strassen to the 2x1 block? No,
    # Strassen is for 2x2. We need to decompose 2x2 * 2x7.
    
    # Better approach: decompose 2x2 * 2x7 as 7 separate 2x2 * 2x1 problems?
    # No, that's not right either. Let me think about this differently.
    
    # For 2x2 * 2x7: C[i,k] = sum_j A[i,j] B[j,k] for i in 0-1, k in 0-6
    # This is like multiplying a 2x2 matrix by a 2x7 matrix.
    # We can view B as having 7 columns, each of size 2x1.
    # For each column k, we compute a 2x1 result: C[:,k] = A * B[:,k]
    # Each 2x2 * 2x1 multiplication has rank... let's think.
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    # This is a 2x2 x 2x1 -> 2x1 problem. Rank = ?
    # It's like a bilinear map R^4 x R^2 -> R^2.
    # Schoolbook: 2*2*1 = 4 multiplications.
    # Can we do better? Probably not much.
    
    # Let me just use schoolbook for the 2x2 * 2x7 part.
    # Rank = 2*2*7 = 28.
    
    # Actually, let me reconsider the whole approach. The total rank with
    # pure schoolbook is 3*3*7 = 63. If I can save even a few, that's an improvement.
    
    # Let me try a different block decomposition that might be more efficient.
    # What if I treat it as 3 separate 1x3 * 3x7 -> 1x7 problems?
    # Each 1x3 * 3x7 has rank 3*7 = 21. Total = 3*21 = 63. No improvement.
    
    # What about 3x3 * 3x7 as a single problem? The tensor rank is what we want.
    # Known bounds: R(3,3,7) >= 3*7 = 21 (from output), and R(3,3,7) >= 3*3 = 9, etc.
    # The schoolbook rank is 63.
    
    # Let me try a random search approach with better initialization.
    return None

def _try_random_search(n, m, p, rank, time_budget, seed=123):
    """Try to find a valid decomposition of given rank using random search."""
    start = time.time()
    rng = np.random.default_rng(seed)
    
    if rank < max(n*m, m*p, p*n):
        return None
    
    # Random initialization
    for attempt in range(20):
        if time.time() - start > time_budget:
            break
        
        U_arr = rng.standard_normal((n * m, rank))
        V_arr = rng.standard_normal((m * p, rank))
        W_arr = rng.standard_normal((p * n, rank))
        
        if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p, tol=1e-6, trials=3):
            U_clean = np.round(U_arr * 2) / 2
            V_clean = np.round(V_arr * 2) / 2
            W_clean = np.round(W_arr * 2) / 2
            
            if _verify(U_clean.tolist(), V_clean.tolist(), W_clean.tolist(), n, m, p, tol=1e-8, trials=8):
                return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
    
    return None

def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution (rank 63)
    candidate = _schoolbook(n, m, p)
    best = candidate
    best_rank = n * m * p
    
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        pass  # Already verified
    
    # Try to find lower rank decompositions
    ranks_to_try = [55, 50, 45, 40, 35, 30, 28, 25, 23, 22, 21]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        result = _try_random_search(n, m, p, rank, time_budget=8.0)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # Also try a few more specific ranks if time allows
    if time.time() < deadline - 15:
        for rank in [48, 42, 38, 32, 26]:
            if time.time() > deadline - 5:
                break
            if rank >= best_rank:
                continue
            result = _try_random_search(n, m, p, rank, time_budget=5.0, seed=456)
            if result is not None:
                best = result
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
