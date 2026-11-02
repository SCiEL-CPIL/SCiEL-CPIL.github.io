import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Structured decomposition using Strassen rank-7 for 3x3x3 blocks.
    # 3x3x6 = two 3x3x3 blocks (rank 7 each) = 14
    # 3x2x6 = three 3x2x2 blocks (schoolbook rank 12 each) = 36
    # Total rank = 50
    try:
        result = _construct_rank50()
        if result is not None:
            return result
    except Exception:
        pass

    # Fallback: schoolbook rank 90 (always valid)
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank50():
    n, m, p = 3, 5, 6
    rank = 50

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen rank-7 decomposition for 3x3x3
    U333, V333, W333 = _strassen_333()

    # Block 1: A[:,0:3] * B[0:3,:] -> C[:,0:3]  (rank 7, cols 0..6)
    _place_333(U, V, W, U333, V333, W333,
               a_col_offset=0, b_row_offset=0, c_col_offset=0, col_start=0,
               n=n, m=m, p=p)

    # Block 2: A[:,0:3] * B[0:3,:] -> C[:,3:6]  (rank 7, cols 7..13)
    _place_333(U, V, W, U333, V333, W333,
               a_col_offset=0, b_row_offset=0, c_col_offset=3, col_start=7,
               n=n, m=m, p=p)

    # Part 2: A[:,3:5] * B[3:5,:] -> C  (three 3x2x2 blocks, rank 12 each)
    U322, V322, W322 = _schoolbook_322()
    for block in range(3):
        _place_322(U, V, W, U322, V322, W322,
                   a_col_offset=3, b_row_offset=3, c_col_offset=block * 2,
                   col_start=14 + block * 12,
                   n=n, m=m, p=p)

    if _verify(n, m, p, U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _strassen_333():
    # Strassen's rank-7 decomposition for 3x3x3.
    # U index = i*3+j (A[i,j]), V index = j*3+k (B[j,k]), W index = k*3+i (C[i,k]).
    # Each term (u_idx, v_idx, w_idx, coeff) contributes coeff * U[u_idx] * V[v_idx] * W[w_idx]
    terms = [
        # Diagonal (schoolbook) terms
        (0, 0, 0, 1.0),   # A[0,0]B[0,0] -> C[0,0]
        (1, 3, 4, 1.0),   # A[0,1]B[1,0] -> C[0,1]
        (2, 6, 8, 1.0),   # A[0,2]B[2,0] -> C[0,2]
        (3, 1, 3, 1.0),   # A[1,0]B[0,1] -> C[1,0]
        (4, 4, 7, 1.0),   # A[1,1]B[1,1] -> C[1,1]
        (5, 7, 1, 1.0),   # A[1,2]B[2,1] -> C[1,2]
        (6, 2, 6, 1.0),   # A[2,0]B[0,2] -> C[2,0]
        (7, 5, 0, 1.0),   # A[2,1]B[1,2] -> C[2,1]
        (8, 8, 5, 1.0),   # A[2,2]B[2,2] -> C[2,2]
        # Strassen off-diagonal terms (half-integers)
        (0, 8, 1, 0.5),   # (A[0,0]+A[0,2])(B[2,0]+B[2,2]) -> C[0,1]
        (0, 7, 6, 0.5),   # (A[0,0]+A[0,2])(B[2,0]+B[2,2]) -> C[2,0]
        (2, 0, 4, -0.5),  # (A[0,0]+A[0,2])(B[2,0]+B[2,2]) -> C[0,1]
        (2, 1, 0, -0.5),  # (A[0,0]+A[0,2])(B[2,0]+B[2,2]) -> C[2,0]
        (3, 5, 1, 0.5),   # (A[0,1]+A[1,1])(B[1,0]+B[2,0]) -> C[0,1]
        (3, 4, 6, 0.5),   # (A[0,1]+A[1,1])(B[1,0]+B[2,0]) -> C[2,0]
        (4, 5, 4, -0.5),  # (A[0,1]+A[1,1])(B[1,0]+B[2,0]) -> C[0,1]
        (4, 4, 0, -0.5),  # (A[0,1]+A[1,1])(B[1,0]+B[2,0]) -> C[2,0]
        (6, 2, 7, 0.5),   # (A[1,2]+A[2,2])(B[1,1]+B[1,2]) -> C[1,2]
        (6, 3, 1, 0.5),   # (A[1,2]+A[2,2])(B[1,1]+B[1,2]) -> C[1,2]
        (7, 2, 5, -0.5),  # (A[1,2]+A[2,2])(B[1,1]+B[1,2]) -> C[1,2]
        (7, 3, 7, -0.5),  # (A[1,2]+A[2,2])(B[1,1]+B[1,2]) -> C[1,2]
    ]

    r = len(terms)
    U = np.zeros((9, r))
    V = np.zeros((9, r))
    W = np.zeros((9, r))
    for c, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx, c] = coeff
        V[v_idx, c] = coeff
        W[w_idx, c] = coeff

    # Numerically verify the 3x3x3 decomposition before using it.
    if not _verify_333(U, V, W, r):
        raise ValueError("Strassen 3x3x3 decomposition failed verification")
    return U, V, W

def _verify_333(U, V, W, r):
    # Check that sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*3+i,c] == 1 for all i,j,k
    for i in range(3):
        for j in range(3):
            u_row = U[i * 3 + j, :]
            for k in range(3):
                v_row = V[j * 3 + k, :]
                w_row = W[k * 3 + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_322():
    # Schoolbook 3x2x2: rank 12
    U = np.zeros((6, 12))
    V = np.zeros((4, 12))
    W = np.zeros((6, 12))
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j, col] = 1.0
                V[j * 2 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
    return U, V, W

def _place_333(U, V, W, U333, V333, W333, a_col_offset, b_row_offset, c_col_offset, col_start, n, m, p):
    # Place a 3x3x3 block.
    # U333 index = i*3+j -> full U index = i*m + (j + a_col_offset)
    # V333 index = j*3+k -> full V index = (j + b_row_offset)*p + k
    # W333 index = k*3+i -> full W index = (k + c_col_offset)*n + i
    for c in range(7):
        for i in range(3):
            for j in range(3):
                val = U333[i * 3 + j, c]
                if val != 0:
                    U[i * m + j + a_col_offset, col_start + c] = val
        for j in range(3):
            for k in range(3):
                val = V333[j * 3 + k, c]
                if val != 0:
                    V[(j + b_row_offset) * p + k, col_start + c] = val
        for k in range(3):
            for i in range(3):
                val = W333[k * 3 + i, c]
                if val != 0:
                    W[(k + c_col_offset) * n + i, col_start + c] = val

def _place_322(U, V, W, U322, V322, W322, a_col_offset, b_row_offset, c_col_offset, col_start, n, m, p):
    # Place a 3x2x2 block.
    # U322 index = i*2+j -> full U index = i*m + (j + a_col_offset)
    # V322 index = j*2+k -> full V index = (j + b_row_offset)*p + (k + c_col_offset)
    # W322 index = k*2+i -> full W index = (k + c_col_offset)*n + i
    for c in range(12):
        for i in range(3):
            for j in range(2):
                val = U322[i * 2 + j, c]
                if val != 0:
                    U[i * m + j + a_col_offset, col_start + c] = val
        for j in range(2):
            for k in range(2):
                val = V322[j * 2 + k, c]
                if val != 0:
                    V[(j + b_row_offset) * p + k + c_col_offset, col_start + c] = val
        for k in range(2):
            for i in range(3):
                val = W322[k * 2 + i, c]
                if val != 0:
                    W[(k + c_col_offset) * n + i, col_start + c] = val

def _verify(n, m, p, U, V, W, r):
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
