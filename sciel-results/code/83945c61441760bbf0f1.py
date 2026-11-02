"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target = target * 8
    if np.max(np.abs(T_computed - T_target)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_3x3():
    """
    Return a valid decomposition for 3x3x3 matrix multiplication.
    Uses schoolbook rank 27 as safe baseline.
    """
    return _schoolbook(3, 3, 3)

def _strassen_block_5x5():
    """
    Construct a 5x5x5 decomposition using block structure.
    Split 5x5 into blocks: use 2x2 Strassen on overlapping sub-blocks.
    
    Strategy: Split each 5x5 matrix into:
    - A 2x2 block in top-left
    - Remaining parts handled with additional terms
    
    This gives a rank of at most 7*4 + additional terms.
    """
    # For a 5x5 matrix, we can think of it as:
    # [A11 A12 A13]
    # [A21 A22 A23]
    # [A31 A32 A33]
    # where A11, A22 are 2x2, others are 2x2, 2x1, 1x2, or 1x1
    
    # Use Strassen on the 2x2 core (indices 0-1)
    # Then handle the boundary rows/columns separately
    
    # This is complex; let's use a simpler approach:
    # Pad to 8x8 (next power of 2) and use recursive Strassen, then extract 5x5
    # But that gives rank 7^3 = 343 for 8x8, too high
    
    # Alternative: Use a direct construction
    # Split 5 = 2 + 2 + 1
    # Use Strassen for the 2x2 blocks and handle cross terms
    
    # For now, return schoolbook as safe fallback
    return None

def _random_search_rank(target_rank, n, m, p, T_flat, deadline, max_iters=100):
    """
    Try to find a valid decomposition of given rank using random search with quantization.
    """
    if time.time() > deadline - 5:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(nm, target_rank))
        V = np.random.randint(-2, 3, size=(mp, target_rank))
        
        # Build design matrix
        B_flat = np.zeros((target_rank, nm * mp), dtype=np.int64)
        for r in range(target_rank):
            B_flat[r] = np.outer(U[:, r], V[:, r]).flatten()
        
        # Solve for W using least squares
        try:
            W_sol, residuals, rank_sv, sv = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=1e-10)
        except:
            continue
        
        W_sol = W_sol.T  # shape (pn, target_rank)
        
        # Try different quantizations
        for scale in [1, 2, 4]:
            W_q = np.round(W_sol * scale) / float(scale)
            if _verify(U.tolist(), V.tolist(), W_q.tolist(), n, m, p, target_rank):
                return U.tolist(), V.tolist(), W_q.tolist()
        
        # Also try integer-only
        W_q_int = np.round(W_sol)
        if _verify(U.tolist(), V.tolist(), W_q_int.tolist(), n, m, p, target_rank):
            return U.tolist(), V.tolist(), W_q_int.tolist()
    
    return None

def _strassen_recursive(n, m, p, depth=0):
    """
    Recursive Strassen-style decomposition.
    For n=m=p, split into halves and use Strassen's 7 multiplications.
    """
    if n == 1:
        # 1x1x1: rank 1
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    if n != m or m != p:
        # Non-square: use schoolbook
        return _schoolbook(n, m, p)
    
    # Split n into two halves
    n1 = n // 2
    n2 = n - n1
    
    # For simplicity, only handle even n for now
    if n1 == 0 or n2 == 0 or n1 != n2:
        # Odd size: use schoolbook for this level
        return _schoolbook(n, m, p)
    
    # Recursively compute decompositions for n1 x n1 x n1
    U1, V1, W1 = _strassen_recursive(n1, n1, n1, depth+1)
    
    # Convert to numpy
    U1 = np.array(U1, dtype=np.int64)
    V1 = np.array(V1, dtype=np.int64)
    W1 = np.array(W1, dtype=np.int64)
    
    rank1 = U1.shape[1]
    
    # Now apply Strassen's 7 multiplications at this level
    # Each "multiplication" is a rank11 sub-problem
    # We need to combine the 7 Strassen terms with the recursive decomposition
    
    # Strassen terms for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Total rank = 7 * rank1
    total_rank = 7 * rank1
    
    U_full = np.zeros((n*n, total_rank), dtype=np.int64)
    V_full = np.zeros((n*n, total_rank), dtype=np.int64)
    W_full = np.zeros((n*n, total_rank), dtype=np.int64)
    
    # Helper to extract block from matrix
    def get_block(mat, row_start, row_end, col_start, col_end):
        return mat[row_start:row_end, col_start:col_end]
    
    # Strassen combination
    # For each of the 7 Strassen terms, we use the recursive decomposition
    # The U, V, W for each term are linear combinations of the block U, V, W
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # U rows: A11 and A22 both contribute with +1
    # V rows: B11 and B22 both contribute with +1
    # W rows: C11 and C22 both get +1
    
    for t in range(rank1):
        col = t
        # U: A11 block (rows 0:n1, cols 0:n1) and A22 block (rows n1:n, cols n1:n)
        U_full[0:n1, col] = U1[0:n1, t]  # A11 contribution
        U_full[n1:n, col] = U1[n1:n, t]  # A22 contribution (same pattern)
        # Wait, this is wrong. Let me think more carefully.
        
    # This approach is getting complex. Let me use a simpler method.
    # For odd n, just use schoolbook.
    return _schoolbook(n, m, p)

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try random search for lower ranks
    ranks_to_try = [100, 90, 80, 70, 65, 60, 55, 50, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = _random_search_rank(target_rank, n, m, p, T_flat, deadline, max_iters=30)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    return best_result
# EVOLVE-BLOCK-END
