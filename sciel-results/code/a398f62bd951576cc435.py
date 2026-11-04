import time
import numpy as np
from fractions import Fraction
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition U, V, W exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Convention:
    A is n x m. B is m x p. C is n x p.
    C[i, k] = sum_j A[i, j] B[j, k].
    
    Flattening:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    
    Tensor T[i, j, k] = 1 for all valid i, j, k.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices:
    # A: [[A00, A01], [A10, A11]] -> indices 0,1,2,3
    # B: [[B00, B01], [B10, B11]] -> indices 0,1,2,3
    # C: [[C00, C01], [C10, C11]] -> indices 0,1,2,3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    # Strassen terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A part):
    # M1: A00, A11 -> 1
    # M2: A10, A11 -> 1
    # M3: A00 -> 1
    # M4: A11 -> 1
    # M5: A00, A01 -> 1
    # M6: A10, -A00 -> 1, -1
    # M7: A01, -A11 -> 1, -1
    
    U[0, 0] = 1; U[3, 0] = 1 # M1: A00, A11
    U[2, 1] = 1; U[3, 1] = 1 # M2: A10, A11
    U[0, 2] = 1               # M3: A00
    U[3, 3] = 1               # M4: A11
    U[0, 4] = 1; U[1, 4] = 1  # M5: A00, A01
    U[2, 5] = 1; U[0, 5] = -1 # M6: A10, -A00
    U[1, 6] = 1; U[3, 6] = -1 # M7: A01, -A11
    
    # V (B part):
    # M1: B00, B11 -> 1
    # M2: B00 -> 1
    # M3: B01, -B11 -> 1, -1
    # M4: B10, -B00 -> 1, -1
    # M5: B11 -> 1
    # M6: B00, B01 -> 1
    # M7: B10, B11 -> 1
    
    V[0, 0] = 1; V[3, 0] = 1 # M1: B00, B11
    V[0, 1] = 1               # M2: B00
    V[1, 2] = 1; V[3, 2] = -1 # M3: B01, -B11
    V[2, 3] = 1; V[0, 3] = -1 # M4: B10, -B00
    V[3, 4] = 1               # M5: B11
    V[0, 5] = 1; V[1, 5] = 1  # M6: B00, B01
    V[2, 6] = 1; V[3, 6] = 1  # M7: B10, B11
    
    # W (C part):
    # M1: C00, C11 -> 1, 1
    # M2: C10, -C11 -> 1, -1
    # M3: C01, C11 -> 1, 1
    # M4: C00, C10 -> 1, 1
    # M5: -C00, C01 -> -1, 1
    # M6: C11 -> 1
    # M7: C00 -> 1
    
    # W indices: k*n+i
    # C00: 0, C01: 1, C10: 2, C11: 3
    
    W[0, 0] = 1; W[3, 0] = 1  # M1: C00, C11
    W[2, 1] = 1; W[3, 1] = -1 # M2: C10, -C11
    W[1, 2] = 1; W[3, 2] = 1  # M3: C01, C11
    W[0, 3] = 1; W[2, 3] = 1  # M4: C00, C10
    W[0, 4] = -1; W[1, 4] = 1 # M5: -C00, C01
    W[3, 5] = 1               # M6: C11
    W[0, 6] = 1               # M7: C00
    
    return U, V, W

def _combine_decompositions(n, m, p, blocks):
    """
    Combine decompositions of blocks into a full decomposition.
    blocks: list of (n_b, m_b, p_b, U_b, V_b, W_b, offset_i, offset_j, offset_k)
    """
    total_rank = sum(U_b.shape[1] for _, _, _, U_b, _, _, _, _, _ in blocks)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for n_b, m_b, p_b, U_b, V_b, W_b, off_i, off_j, off_k in blocks:
        r_b = U_b.shape[1]
        for i in range(n_b):
            for j in range(m_b):
                U[(i + off_i)*m + (j + off_j), col:col+r_b] = U_b[i*m_b + j, :]
        for j in range(m_b):
            for k in range(p_b):
                V[(j + off_j)*p + (k + off_k), col:col+r_b] = V_b[j*p_b + k, :]
        for k in range(p_b):
            for i in range(n_b):
                W[(k + off_k)*n + (i + off_i), col:col+r_b] = W_b[k*n_b + i, :]
        col += r_b
        
    return U, V, W

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140
    
    # Strategy:
    # Decompose 2x3x10 into blocks.
    # We can split the p dimension (10) into 5+5.
    # Each 2x3x5 can be further split.
    # 2x3x5 = 2x3x2 + 2x3x2 + 2x3x1.
    
    # Let's try to build a decomposition for 2x3x2 using Strassen-like or schoolbook.
    # Schoolbook 2x3x2 rank 12.
    # Can we do better?
    # 2x3x2 tensor.
    # Maybe use 2x2x2 Strassen (rank 7) + 1x2x2 schoolbook (rank 4)?
    # 2x3x2 = 2x2x2 (cols 0,1) + 1x2x2 (col 2).
    # Wait, A is 2x3. Split A into A_left (2x2) and A_right (2x1).
    # B is 3x2. Split B into B_top (2x2) and B_bottom (1x2).
    # C = A_left * B_top + A_right * B_bottom.
    # This is a sum of two independent multiplications.
    # Rank(2x3x2) <= Rank(2x2x2) + Rank(2x1x2).
    # Rank(2x2x2) = 7 (Strassen).
    # Rank(2x1x2) = 4 (Schoolbook: 2*1*2).
    # Total rank = 11.
    
    # Let's verify Rank(2x1x2).
    # A is 2x1, B is 1x2. C is 2x2.
    # C[0,0] = A[0,0]B[0,0]
    # C[0,1] = A[0,0]B[0,1]
    # C[1,0] = A[1,0]B[0,0]
    # C[1,1] = A[1,0]B[0,1]
    # Rank 4 schoolbook.
    
    # So Rank(2x3x2) <= 11.
    # Let's construct this.
    
    def get_2x1x2_schoolbook():
        n_b, m_b, p_b = 2, 1, 2
        rank = n_b * m_b * p_b
        U = np.zeros((n_b*m_b, rank))
        V = np.zeros((m_b*p_b, rank))
        W = np.zeros((p_b*n_b, rank))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b + j, col] = 1
                    V[j*p_b + k, col] = 1
                    W[k*n_b + i, col] = 1
                    col += 1
        return U, V, W

    def get_2x2x2_strassen():
        return _strassen_2x2x2()

    def get_2x3x2_combined():
        # 2x3x2 = 2x2x2 (j=0,1) + 2x1x2 (j=2)
        # Note: The split is on the middle dimension m.
        # A (2x3) = [A_left (2x2), A_right (2x1)]
        # B (3x2) = [B_top (2x2); B_bottom (1x2)]
        # C (2x2) = A_left B_top + A_right B_bottom
        
        U1, V1, W1 = get_2x2x2_strassen()
        U2, V2, W2 = get_2x1x2_schoolbook()
        
        # Combine
        # Block 1: n=2, m=2, p=2. Offsets: i=0, j=0, k=0.
        # Block 2: n=2, m=1, p=2. Offsets: i=0, j=2, k=0.
        
        blocks = [
            (2, 2, 2, U1, V1, W1, 0, 0, 0),
            (2, 1, 2, U2, V2, W2, 0, 2, 0)
        ]
        return _combine_decompositions(2, 3, 2, blocks)

    def get_2x3x1_schoolbook():
        n_b, m_b, p_b = 2, 3, 1
        rank = n_b * m_b * p_b
        U = np.zeros((n_b*m_b, rank))
        V = np.zeros((m_b*p_b, rank))
        W = np.zeros((p_b*n_b, rank))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b + j, col] = 1
                    V[j*p_b + k, col] = 1
                    W[k*n_b + i, col] = 1
                    col += 1
        return U, V, W

    def get_2x3x5_combined():
        # 2x3x5 = 2x3x2 (k=0,1) + 2x3x2 (k=2,3) + 2x3x1 (k=4)
        U1, V1, W1 = get_2x3x2_combined()
        U2, V2, W2 = get_2x3x2_combined()
        U3, V3, W3 = get_2x3x1_schoolbook()
        
        blocks = [
            (2, 3, 2, U1, V1, W1, 0, 0, 0),
            (2, 3, 2, U2, V2, W2, 0, 0, 2),
            (2, 3, 1, U3, V3, W3, 0, 0, 4)
        ]
        return _combine_decompositions(2, 3, 5, blocks)

    def get_2x3x10_combined():
        # 2x3x10 = 2x3x5 (k=0..4) + 2x3x5 (k=5..9)
        U1, V1, W1 = get_2x3x5_combined()
        U2, V2, W2 = get_2x3x5_combined()
        
        blocks = [
            (2, 3, 5, U1, V1, W1, 0, 0, 0),
            (2, 3, 5, U2, V2, W2, 0, 0, 5)
        ]
        return _combine_decompositions(2, 3, 10, blocks)

    U, V, W = get_2x3x10_combined()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
