"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication."""
    n_val, m_val, p_val = n, m, p
    np.random.seed(42)
    A = np.random.randn(n_val, m_val)
    B = np.random.randn(m_val, p_val)
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float) if not isinstance(U, np.ndarray) else U.astype(float)
    V_arr = np.array(V, dtype=float) if not isinstance(V, np.ndarray) else V.astype(float)
    W_arr = np.array(W, dtype=float) if not isinstance(W, np.ndarray) else W.astype(float)
    
    C_computed = np.zeros((n_val, p_val))
    rank = U_arr.shape[1]
    for c in range(rank):
        X = np.zeros(n_val)
        for i in range(n_val):
            for j in range(m_val):
                X[i] += U_arr[i * m_val + j, c] * A[i, j]
        Y = np.zeros(p_val)
        for k in range(p_val):
            for j in range(m_val):
                Y[k] += V_arr[j * p_val + k, c] * B[j, k]
        for i in range(n_val):
            for k in range(p_val):
                C_computed[i, k] += X[i] * Y[k] * W_arr[k * n_val + i, c]
    
    return np.allclose(C_computed, C_expected, atol=tol)


def _strassen_2x2():
    """Return Strassen's 7-rank decomposition for 2x2x2 matrix multiplication.
    
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00*(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11*(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _search_2x2x4_rank13(max_time=20):
    """Try to find a rank-13 decomposition for 2x2x4 matrix multiplication.
    
    Uses a local search starting from the rank-14 Strassen-based solution.
    If found, return (U, V, W) with shapes (4,13), (8,13), (8,13).
    Otherwise return None.
    """
    start_time = time.time()
    n, m, p = 2, 2, 4
    target_rank = 13
    
    S_U, S_V, S_W = _strassen_2x2()
    
    # Build rank-14: two independent Strassen for columns 0-1 and 2-3
    U14 = np.zeros((4, 14))
    V14 = np.zeros((8, 14))
    W14 = np.zeros((8, 14))
    
    for block in range(2):
        base_k = 2 * block
        for s in range(7):
            c = block * 7 + s
            for i in range(2):
                for j in range(2):
                    U14[i*2+j, c] = S_U[i*2+j, s]
            for j in range(2):
                for dk in range(2):
                    k = base_k + dk
                    V14[j*4+k, c] = S_V[j*2+dk, s]
            for dk in range(2):
                k = base_k + dk
                for i in range(2):
                    W14[k*2+i, c] = S_W[dk*2+i, s]
    
    def _verify_small(U, V, W, n, m, p, tol=1e-9):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        rank = U.shape[1]
        for c in range(rank):
            X = np.array([sum(U[i*m+j, c]*A[i,j] for j in range(m)) for i in range(n)])
            Y = np.array([sum(V[j*p+k, c]*B[j,k] for j in range(m)) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_comp[i,k] += X[i]*Y[k]*W[k*n+i, c]
        return np.allclose(C_comp, C_exp, atol=tol)
    
    if not _verify_small(U14, V14, W14, n, m, p):
        return None
    
    # Try to eliminate one term by expressing it as a combination of others
    # This is a hard problem. Let's try a simple heuristic:
    # Try all pairs of terms and see if one can be absorbed
    
    # For now, return None as we don't have an efficient search
    return None


def _structured_2x3x8():
    """Build a structured 44-rank decomposition for 2x3x8 matrix multiplication.
    
    Split the middle dimension (size 3) into 2+1:
    - 2x2x8 subproblem: use Strassen on each 2x2x2 block (4 blocks × 7 = 28)
    - 2x1x8 subproblem: schoolbook (2*1*8 = 16)
    Total: 44
    """
    n, m, p = 2, 3, 8
    total_rank = 28 + 16  # 44
    
    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)
    
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


def _structured_2x3x8_improved():
    """Try to build a lower-rank decomposition. Currently returns None."""
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Get the structured rank-44 solution
    U, V, W = _structured_2x3x8()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 48)
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
