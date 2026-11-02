# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0
_MARGIN = 10.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-9):
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
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


def _rank_26_decomposition():
    """
    Explicit rank-26 decomposition using block structure:
    Split 3x3 into 2x2 block (indices 0,1) and 1x1 (index 2).
    
    The 2x2 block uses Strassen (rank 7) for its internal products.
    Cross terms and corner use direct multiplication.
    
    Total: 7 (Strassen for 2x2) + 1 (corner A22*B22) + 18 (cross terms) = 26
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Index mapping:
    # U[i*3+j, r] for A[i,j]
    # V[j*3+k, r] for B[j,k]
    # W[k*3+i, r] for C[i,k]
    
    # Strassen for 2x2 block (indices 0,1 in both A and B):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1.0; U[1, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[1, 0] = 1.0; W[3, 0] = 1.0; W[4, 0] = 1.0
    
    # Term 1: M2 = (A10+A11)B00
    U[3, 1] = 1.0; U[4, 1] = 1.0
    V[0, 1] = 1.0
    W[3, 1] = 1.0; W[4, 1] = 1.0
    
    # Term 2: M3 = A00(B01-B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[4, 2] = -1.0
    W[1, 2] = 1.0; W[4, 2] = 1.0
    
    # Term 3: M4 = A11(B10-B00)
    U[4, 3] = 1.0
    V[3, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[3, 3] = 1.0
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[4, 4] = 1.0
    W[1, 4] = 1.0; W[4, 4] = 1.0
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[3, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0; W[4, 5] = 1.0
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1.0; U[4, 6] = -1.0
    V[3, 6] = 1.0; V[4, 6] = 1.0
    W[0, 6] = 1.0; W[1, 6] = 1.0
    
    # Term 7: Corner A22*B22
    U[8, 7] = 1.0
    V[8, 7] = 1.0
    W[8, 7] = 1.0
    
    # Cross terms: A[0:2, :] * B[:, 2] (A rows 0,1 times B col 2)
    # This gives C[0:2, 2]
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # Term 8: A00*B02
    U[0, 8] = 1.0
    V[2, 8] = 1.0
    W[6, 8] = 1.0  # W[k*n+i] = W[2*3+0] = W[6]
    
    # Term 9: A01*B12
    U[1, 9] = 1.0
    V[5, 9] = 1.0
    W[6, 9] = 1.0
    
    # Term 10: A02*B22
    U[2, 10] = 1.0
    V[8, 10] = 1.0
    W[6, 10] = 1.0
    
    # Term 11: A10*B02
    U[3, 11] = 1.0
    V[2, 11] = 1.0
    W[7, 11] = 1.0  # W[2*3+1] = W[7]
    
    # Term 12: A11*B12
    U[4, 12] = 1.0
    V[5, 12] = 1.0
    W[7, 12] = 1.0
    
    # Term 13: A12*B22
    U[5, 13] = 1.0
    V[8, 13] = 1.0
    W[7, 13] = 1.0
    
    # Cross terms: A[2, :] * B[:, 0:2] (A row 2 times B cols 0,1)
    # This gives C[2, 0:2]
    # C[2,0] = A20*B00 + A21*B10 + A22*B20
    # C[2,1] = A20*B01 + A21*B11 + A22*B21
    
    # Term 14: A20*B00
    U[6, 14] = 1.0
    V[0, 14] = 1.0
    W[0, 14] = 1.0  # W[0*3+2] = W[2]... wait
    
    # Let me re-check W indexing: W[k*n+i]
    # For C[2,0]: k=0, i=2, so W[0*3+2] = W[2]
    # For C[2,1]: k=1, i=2, so W[1*3+2] = W[5]
    
    U[6, 14] = 1.0
    V[0, 14] = 1.0
    W[2, 14] = 1.0
    
    # Term 15: A21*B10
    U[7, 15] = 1.0
    V[3, 15] = 1.0
    W[2, 15] = 1.0
    
    # Term 16: A22*B20
    U[8, 16] = 1.0
    V[6, 16] = 1.0
    W[2, 16] = 1.0
    
    # Term 17: A20*B01
    U[6, 17] = 1.0
    V[1, 17] = 1.0
    W[5, 17] = 1.0
    
    # Term 18: A21*B11
    U[7, 18] = 1.0
    V[4, 18] = 1.0
    W[5, 18] = 1.0
    
    # Term 19: A22*B21
    U[8, 19] = 1.0
    V[7, 19] = 1.0
    W[5, 19] = 1.0
    
    # Cross terms: A[0:2, 2] * B[2, 0:2] (A col 2, rows 0,1 times B row 2, cols 0,1)
    # This gives C[0:2, 0:2]
    # C[0,0] += A02*B20
    # C[0,1] += A02*B21
    # C[1,0] += A12*B20
    # C[1,1] += A12*B21
    
    # Term 20: A02*B20
    U[2, 20] = 1.0
    V[6, 20] = 1.0
    W[0, 20] = 1.0  # C[0,0]: k=0,i=0
    
    # Term 21: A02*B21
    U[2, 21] = 1.0
    V[7, 21] = 1.0
    W[1, 21] = 1.0  # C[0,1]: k=1,i=0
    
    # Term 22: A12*B20
    U[5, 22] = 1.0
    V[6, 22] = 1.0
    W[3, 22] = 1.0  # C[1,0]: k=0,i=1
    
    # Term 23: A12*B21
    U[5, 23] = 1.0
    V[7, 23] = 1.0
    W[4, 23] = 1.0  # C[1,1]: k=1,i=1
    
    # Cross terms: A[2, 0:2] * B[0:2, 2] (A row 2, cols 0,1 times B col 2, rows 0,1)
    # This gives C[2, 0:2]... wait, already handled above
    # Actually C[2,0] and C[2,1] are already done in terms 14-19
    
    # We need: A[0:2, 2] * B[2, 0:2] done (terms 20-23)
    # And: A[2, 0:2] * B[0:2, 2] done (terms 14-19)
    # And: A[2, 2] * B[2, 2] done (term 7)
    
    # What about A[0:2, 0:2] * B[0:2, 0:2]? That's the 2x2 block, done by Strassen (terms 0-6)
    
    # Let me verify we have all 27 terms:
    # C[0,0]: A00*B00, A01*B10, A02*B20 -> terms 0 (part), 6 (part), 20
    # C[0,1]: A00*B01, A01*B11, A02*B21 -> terms 0 (part), 4 (part), 21
    # C[0,2]: A00*B02, A01*B12, A02*B22 -> terms 8, 9, 10
    # C[1,0]: A10*B00, A11*B10, A12*B20 -> terms 1 (part), 6 (part), 22
    # C[1,1]: A10*B01, A11*B11, A12*B21 -> terms 1 (part), 4 (part), 23
    # C[1,2]: A10*B02, A11*B12, A12*B22 -> terms 11, 12, 13
    # C[2,0]: A20*B00, A21*B10, A22*B20 -> terms 14, 15, 16
    # C[2,1]: A20*B01, A21*B11, A22*B21 -> terms 17, 18, 19
    # C[2,2]: A20*B02, A21*B12, A22*B22 -> terms 24?, 25?, 7
    
    # I'm missing C[2,2] terms for A20*B02 and A21*B12!
    # Term 24: A20*B02
    U[6, 24] = 1.0
    V[2, 24] = 1.0
    W[8, 24] = 1.0  # C[2,2]: k=2,i=2
    
    # Term 25: A21*B12
    U[7, 25] = 1.0
    V[5, 25] = 1.0
    W[8, 25] = 1.0
    
    return U, V, W


def construct():
    n, m, p = 3, 3, 3
    
    # Try rank 26 explicit decomposition
    try:
        U26, V26, W26 = _rank_26_decomposition()
        if _verify(U26, V26, W26, tol=1e-9):
            return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 27
    U_s, V_s, W_s = _schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
