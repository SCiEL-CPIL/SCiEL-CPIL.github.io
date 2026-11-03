"""Matrix multiplication 2x3x8 tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float) if not isinstance(U, np.ndarray) else U.astype(float)
    V_arr = np.array(V, dtype=float) if not isinstance(V, np.ndarray) else V.astype(float)
    W_arr = np.array(W, dtype=float) if not isinstance(W, np.ndarray) else W.astype(float)
    
    C_computed = np.zeros((n, p))
    rank = U_arr.shape[1]
    for c in range(rank):
        X = np.zeros(n)
        for i in range(n):
            for j in range(m):
                X[i] += U_arr[i * m + j, c] * A[i, j]
        Y = np.zeros(p)
        for k in range(p):
            for j in range(m):
                Y[k] += V_arr[j * p + k, c] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += X[i] * Y[k] * W_arr[k * n + i, c]
    
    return np.allclose(C_computed, C_expected, atol=tol)


def _strassen_2x2():
    """Return Strassen's 7-rank decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_2x3x8_rank44():
    """Build structured 44-rank decomposition: 4 Strassen blocks (28) + schoolbook (16)."""
    n, m, p = 2, 3, 8
    total_rank = 44
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    S_U, S_V, S_W = _strassen_2x2()
    
    for block_idx in range(4):
        base_col = 2 * block_idx
        for s in range(7):
            c = block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, c] = S_U[i * 2 + j, s]
            for j in range(2):
                for dk in range(2):
                    k = base_col + dk
                    V[j * 8 + k, c] = S_V[j * 2 + dk, s]
            for dk in range(2):
                k = base_col + dk
                for i in range(2):
                    W[k * 2 + i, c] = S_W[dk * 2 + i, s]
    
    base = 28
    for i in range(2):
        for k in range(8):
            c = base + i * 8 + k
            U[i * 3 + 2, c] = 1
            V[2 * 8 + k, c] = 1
            W[k * 2 + i, c] = 1
    
    return U, V, W


def _build_2x3x8_rank43():
    """
    Try to get rank 43 by sharing one Strassen term between two blocks.
    The idea: if two 2x2x2 blocks share a Strassen pattern, we might merge them.
    """
    n, m, p = 2, 3, 8
    # Start with rank 44 and try to merge
    U44, V44, W44 = _build_2x3x8_rank44()
    
    # Try to find if any two columns are linearly dependent
    # This is a simple check - if it works, we can drop one
    rank_found = 44
    for c1 in range(44):
        for c2 in range(c1 + 1, 44):
            # Check if column c2 is a scalar multiple of column c1 in all three matrices
            try:
                # Check U columns
                if np.allclose(U44[:, c2], U44[:, c1] * (U44[0, c2] / U44[0, c1]) if U44[0, c1] != 0 else False):
                    # Check V columns
                    if np.allclose(V44[:, c2], V44[:, c1] * (V44[0, c2] / V44[0, c1]) if V44[0, c1] != 0 else False):
                        # Check W columns
                        if np.allclose(W44[:, c2], W44[:, c1] * (W44[0, c2] / W44[0, c1]) if W44[0, c1] != 0 else False):
                            rank_found = 43
                            break
            except:
                continue
        if rank_found == 43:
            break
    
    if rank_found == 43:
        return U44[:, :43], V44[:, :43], W44[:, :43]
    
    return None


def _als_reduce(U, V, W, n, m, p, target_rank, max_iters=50, tol=1e-8):
    """
    Try to reduce rank using ALS (Alternating Least Squares).
    Start from a valid decomposition and try to project to lower rank.
    """
    rank = U.shape[1]
    if rank <= target_rank:
        return U, V, W
    
    # Random projection to target_rank
    np.random.seed(123)
    P = np.random.randn(rank, target_rank)
    U_new = U @ P
    V_new = V @ P
    W_new = W @ P
    
    # ALS iterations
    for it in range(max_iters):
        # Update U
        for c in range(target_rank):
            # Solve for U[:, c] given V[:, c] and W[:, c]
            # This is a linear system - simplified version
            pass
        
        # Update V
        for c in range(target_rank):
            pass
        
        # Update W
        for c in range(target_rank):
            pass
        
        if it % 10 == 0:
            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return None


def _structured_2x3x8_improved():
    """
    Try to build a lower-rank decomposition.
    Strategy: Use Strassen on the 2x2 part (4 blocks x 7 = 28), 
    then optimize the 2x1x8 part (16 terms) to see if we can save.
    """
    n, m, p = 2, 3, 8
    
    # First try rank 43
    result = _build_2x3x8_rank43()
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    # Try ALS reduction from rank 44
    U44, V44, W44 = _build_2x3x8_rank44()
    for target in [43, 42, 41]:
        result = _als_reduce(U44, V44, W44, n, m, p, target)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    return None


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Try improved decomposition first
    if time.time() < deadline - 30:
        result = _structured_2x3x8_improved()
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to structured rank-44
    if time.time() < deadline - 10:
        U, V, W = _build_2x3x8_rank44()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook (rank 48)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
