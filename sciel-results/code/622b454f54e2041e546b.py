import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return U, V, W for Strassen 2x2x2 decomposition (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    # U vectors for each of the 7 terms
    U = [
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [-1, 0, 1, 0],  # M6: -A11+A21
        [0, 1, 0, -1],  # M7: A12-A22
    ]
    V = [
        [1, 0, 0, 1],   # M1: B11+B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12-B22
        [-1, 0, 1, 0],  # M4: -B11+B21
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, 1],   # M7: B21+B22
    ]
    W = [
        [1, 0, 0, 1],   # M1: C11+C22
        [0, 0, 1, -1],  # M2: C21-C22
        [0, 1, 0, 1],   # M3: C12+C22
        [1, 0, 1, 0],   # M4: C11+C21
        [-1, 1, 0, 0],  # M5: -C11+C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    return np.array(U, dtype=int), np.array(V, dtype=int), np.array(W, dtype=int)


def _build_4x4x4_strassen():
    """Build U, V, W for 4x4x4 matrix multiplication using Strassen (rank 49).
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    U index: i*4+j, V index: j*4+k, W index: k*4+i
    """
    n = 4
    U2, V2, W2 = _strassen_2x2_terms()
    
    # Block coefficients for the 7 Strassen terms at the block level
    # Each term t has coefficients for blocks A11, A12, A21, A22
    block_A = [
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [-1, 0, 1, 0],  # M6: -A11+A21
        [0, 1, 0, -1],  # M7: A12-A22
    ]
    block_B = [
        [1, 0, 0, 1],   # M1: B11+B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12-B22
        [-1, 0, 1, 0],  # M4: -B11+B21
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, 1],   # M7: B21+B22
    ]
    block_C = [
        [1, 0, 0, 1],   # M1: C11+C22
        [0, 0, 1, -1],  # M2: C21-C22
        [0, 1, 0, 1],   # M3: C12+C22
        [1, 0, 1, 0],   # M4: C11+C21
        [-1, 1, 0, 0],  # M5: -C11+C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for t in range(7):  # block-level Strassen term
        for s in range(7):  # sub-2x2 Strassen term
            col = t * 7 + s
            # U: for each block
            for b in range(4):  # block index 0=A11, 1=A12, 2=A21, 3=A22
                coeff = block_A[t][b]
                if coeff != 0:
                    # Block b corresponds to rows in U
                    # A11: rows 0-3, A12: rows 4-7, A21: rows 8-11, A22: rows 12-15
                    row_start = b * 4
                    for si in range(4):
                        U[row_start + si, col] += coeff * U2[s, si]
            
            # V: for each block
            for b in range(4):
                coeff = block_B[t][b]
                if coeff != 0:
                    row_start = b * 4
                    for si in range(4):
                        V[row_start + si, col] += coeff * V2[s, si]
            
            # W: for each block
            for b in range(4):
                coeff = block_C[t][b]
                if coeff != 0:
                    row_start = b * 4
                    for si in range(4):
                        W[row_start + si, col] += coeff * W2[s, si]
    
    return U, V, W


def _check_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    deadline = time.time() + 100
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Build 4x4x1 decomposition for the 5th column (rank 16)
    # U[i*4+j, c] = 1, V[j*5+4, c] = 1, W[4*4+i, c] = 1
    U441 = np.zeros((16, 16), dtype=int)
    V441 = np.zeros((20, 16), dtype=int)
    W441 = np.zeros((20, 16), dtype=int)
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U441[i * 4 + j, c] = 1
            V441[j * 5 + 4, c] = 1
            W441[4 * 4 + i, c] = 1
    
    # Combine: total rank = 49 + 16 = 65
    U = np.hstack([U444, U441])
    V = np.hstack([V444, V441])
    W = np.hstack([W444, W441])
    
    # Verify
    if not _check_decomp(n, m, p, U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook
        rank = n * m * p
        Us = np.zeros((n * m, rank), dtype=int)
        Vs = np.zeros((m * p, rank), dtype=int)
        Ws = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, col] = 1
                    Vs[j * p + k, col] = 1
                    Ws[k * n + i, col] = 1
                    col += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
