import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_2x3x3_rank14():
    """
    Build a rank 14 decomposition for 2x3x3.
    U shape (6, 14), V shape (9, 14), W shape (6, 14).
    U[i*3+j, r], V[j*3+k, r], W[k*2+i, r].
    """
    # This is a placeholder. I need to construct this.
    # For now, I'll use schoolbook rank 18 to ensure correctness, then try to optimize.
    # But rank 18 leads to total rank 114, which is worse.
    # I must find a better one.
    
    # Let's try to construct rank 14.
    # A is 2x3, B is 3x3.
    # Split B into B1 (3x2) and B2 (3x1).
    # A B1 is 2x3x2.
    # A B2 is 2x3x1.
    
    # 2x3x1 rank 6.
    # 2x3x2 rank 14?
    # 2x3x2 is 2x2x3.
    # 2x2x3 rank 14?
    # A2x2 B2x3.
    # Split B into B2x2 and B2x1.
    # A B2x2 rank 7.
    # A B2x1 rank 4.
    # Total 11.
    # So 2x3x2 rank 11.
    # Total 2x3x3 rank 6 + 11 = 17.
    
    # This is still too high.
    
    # I will use the rank 10 decomposition if I can find it.
    # Otherwise, I will fall back to the 108 solution.
    
    # For now, I'll return None to indicate failure.
    return None

def _build_4x5x6_split23():
    """
    Build 4x5x6 decomposition using 2+3 split.
    """
    n, m, p = 4, 5, 6
    
    # Part 1: 4x2x6
    # 6 blocks of 2x2x2
    U2, V2, W2 = _build_strassen_2x2()
    rank1 = 6 * 7
    U1 = np.zeros((4 * 2, rank1)) # 8x42
    V1 = np.zeros((2 * 6, rank1)) # 12x42
    W1 = np.zeros((6 * 4, rank1)) # 24x42
    
    # A is 4x2. Split into A1 (2x2), A2 (2x2).
    # B is 2x6. Split into B1, B2, B3 (2x2).
    # A B = A1 B1 + A1 B2 + A1 B3 + A2 B1 + A2 B2 + A2 B3.
    
    for ai in range(2): # 0 for A1, 1 for A2
        for bj in range(3): # 0, 1, 2 for B1, B2, B3
            block_idx = ai * 3 + bj
            base = block_idx * 7
            
            # U: A part
            # A is 4x2. Rows 0-3.
            # A1 is rows 0-1, A2 is rows 2-3.
            # U2 is 4x7. U2[i*2+j, r].
            for i_local in range(2):
                for j_local in range(2):
                    i_global = ai * 2 + i_local
                    j_global = j_local
                    u_row = i_global * 2 + j_global
                    U1[u_row, base:base+7] = U2[i_local * 2 + j_local, :]
                    
            # V: B part
            # B is 2x6. Rows 0-1.
            # B1 is cols 0-1, B2 is cols 2-3, B3 is cols 4-5.
            # V2 is 4x7. V2[j*2+k, r].
            for j_local in range(2):
                for k_local in range(2):
                    j_global = j_local
                    k_global = bj * 2 + k_local
                    v_row = j_global * 6 + k_global
                    V1[v_row, base:base+7] = V2[j_local * 2 + k_local, :]
                    
            # W: C part
            # C is 4x6.
            # C11 (2x2) from A1 B1.
            # C12 (2x2) from A1 B2.
            # C13 (2x2) from A1 B3.
            # C21 (2x2) from A2 B1.
            # C22 (2x2) from A2 B2.
            # C23 (2x2) from A2 B3.
            # W2 is 4x7. W2[k*2+i, r].
            for k_local in range(2):
                for i_local in range(2):
                    k_global = bj * 2 + k_local
                    i_global = ai * 2 + i_local
                    w_row = k_global * 4 + i_global
                    W1[w_row, base:base+7] = W2[k_local * 2 + i_local, :]
                    
    # Part 2: 4x3x6
    # 4 blocks of 2x3x3
    # I need a 2x3x3 decomposition.
    # For now, I'll use schoolbook rank 18.
    # Total rank 42 + 4*18 = 114.
    # This is worse than 108.
    
    # I will return None to indicate failure.
    return None

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
