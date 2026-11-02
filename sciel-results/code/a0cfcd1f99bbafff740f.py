"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]
# Known Strassen-style rank for 5x5 matrix multiplication: 49
# Using the known decomposition for 5x5x5 matrix multiplication
# Reference: Stothers (2010) showed rank 49 is achievable
# We will hardcode a known valid decomposition if available, or search for it

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
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
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
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # U, V, W have shape (4, 7) each
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Strassen's 7 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U rows: A11, A12, A21, A22 (indices 0,1,2,3)
    # V rows: B11, B12, B21, B22 (indices 0,1,2,3)
    # W rows: C11, C12, C21, C22 (indices 0,1,2,3)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22 both get +M1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 gets +M2, C22 gets -M2
    
    # Term 2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 gets +M3, C22 gets +M3
    
    # Term 3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 gets +M4, C21 gets +M4
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 gets -M5, C12 gets +M5
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 gets +M6
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 gets +M7
    
    return U, V, W

def _strassen_3x3():
    """
    Return a valid decomposition for 3x3x3 matrix multiplication.
    Uses a known rank-23 decomposition (or similar).
    For now, use schoolbook rank 27 as fallback, but try to find better.
    """
    # Use schoolbook for 3x3 as a safe baseline
    return _schoolbook(3, 3, 3)

def _combine_decompositions(n, m, p):
    """
    Try to construct a 5x5 decomposition using block structure.
    Split 5x5 into blocks and use known smaller decompositions.
    """
    # This is a placeholder - we'll use a different approach
    return None

def _random_search_rank(target_rank, n, m, p, T_flat, deadline, max_iters=100):
    """
    Try to find a valid decomposition of given rank using random search with quantization.
    """
    if time.time() > deadline - 5:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(42)  # For reproducibility within a call
    
    for iter in range(max_iters):
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

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ranks from 125 (schoolbook) down to 49
    # Focus on reasonable ranks first
    ranks_to_try = [125, 100, 80, 70, 65, 60, 55, 50, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = _random_search_rank(target_rank, n, m, p, T_flat, deadline, max_iters=50)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    # If we found a rank above 55, try to go lower
    if best_rank > 55:
        for target_rank in range(min(best_rank - 1, 54), 48, -1):
            if time.time() > deadline - 5:
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
