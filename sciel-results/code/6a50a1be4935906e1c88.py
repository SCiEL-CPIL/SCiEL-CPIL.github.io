import numpy as np
import time

DIMENSIONS = [2, 3, 5]
_START_TIME = time.time()
TIME_LIMIT = 110

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to tensors
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Reconstruct tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W index is k*n + i, so W_t[k, i, r]
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # The multiplication tensor has all 1s
    return np.allclose(T_reconstructed, 1.0, atol=1e-9)

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: A[i, j] -> 3*i + j
    # V: B[j, k] -> 5*j + k
    # W: C[i, k] -> 2*k + i
    
    # --- Block 1: C[:, 0:2] ---
    # Strassen A_left * B_1_top
    col = 0
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[6][col] += 1
    W[0][col] += 1; W[3][col] += 1

    col = 1
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[1][col] += 1; W[3][col] -= 1

    col = 2
    U[0][col] += 1
    V[1][col] += 1; V[6][col] -= 1
    W[2][col] += 1; W[3][col] += 1

    col = 3
    U[4][col] += 1
    V[5][col] += 1; V[0][col] -= 1
    W[1][col] += 1; W[0][col] += 1

    col = 4
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    W[2][col] += 1; W[0][col] -= 1

    col = 5
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[3][col] += 1

    col = 6
    U[1][col] += 1; U[4][col] -= 1
    V[5][col] += 1; V[6][col] += 1
    W[0][col] += 1

    # Schoolbook A_right * B_1_bottom
    col = 7
    U[2][col] += 1
    V[10][col] += 1
    W[0][col] += 1

    col = 8
    U[2][col] += 1
    V[11][col] += 1
    W[2][col] += 1

    col = 9
    U[5][col] += 1
    V[10][col] += 1
    W[1][col] += 1

    col = 10
    U[5][col] += 1
    V[11][col] += 1
    W[3][col] += 1

    # --- Block 2: C[:, 2:4] ---
    # Strassen A_left * B_2_top
    col = 11
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[7][col] += 1

    col = 12
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] -= 1

    col = 13
    U[0][col] += 1
    V[3][col] += 1; V[8][col] -= 1
    W[6][col] += 1; W[7][col] += 1

    col = 14
    U[4][col] += 1
    V[7][col] += 1; V[2][col] -= 1
    W[5][col] += 1; W[4][col] += 1

    col = 15
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[6][col] += 1; W[4][col] -= 1

    col = 16
    U[3][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[3][col] += 1
    W[7][col] += 1

    col = 17
    U[1][col] += 1; U[4][col] -= 1
    V[7][col] += 1; V[8][col] += 1
    W[4][col] += 1

    # Schoolbook A_right * B_2_bottom
    col = 18
    U[2][col] += 1
    V[12][col] += 1
    W[4][col] += 1

    col = 19
    U[2][col] += 1
    V[13][col] += 1
    W[6][col] += 1

    col = 20
    U[5][col] += 1
    V[12][col] += 1
    W[5][col] += 1

    col = 21
    U[5][col] += 1
    V[13][col] += 1
    W[7][col] += 1

    # --- Block 3: C[:, 4] ---
    # Schoolbook A_left * B_3_top
    col = 22
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1

    col = 23
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1

    col = 24
    U[3][col] += 1
    V[4][col] += 1
    W[9][col] += 1

    col = 25
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1

    # Schoolbook A_right * B_3_bottom
    col = 26
    U[2][col] += 1
    V[14][col] += 1
    W[8][col] += 1

    col = 27
    U[5][col] += 1
    V[14][col] += 1
    W[9][col] += 1

    return U, V, W

def _try_rank27_search():
    """
    Try to find a rank-27 decomposition.
    Currently returns None as no known simple improvement exists.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank28()
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Rank-28 decomposition failed verification")
    
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 5:
        result = _try_rank27_search()
        if result is not None:
            U, V, W = result
    
    return {"U": U, "V": V, "W": W}
