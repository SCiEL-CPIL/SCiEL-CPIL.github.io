import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Returns U, V, W for schoolbook decomposition of n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W

def _strassen_222():
    """Returns U, V, W for Strassen decomposition of 2 x 2 x 2."""
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Indices: 00, 01, 10, 11
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _laderman_333_approx():
    """
    Returns U, V, W for a rank-22 decomposition of 3 x 3 x 3.
    Uses 2x2 Strassen + schoolbook for the rest.
    """
    # Split 3x3x3 into:
    # 1. 2x2x2 (Strassen, rank 7)
    # 2. 2x1x2 (Schoolbook, rank 4) -> A[0:2, 2] * B[2, 0:2]
    # 3. 2x1x1 (Schoolbook, rank 2) -> A[0:2, 2] * B[2, 2]
    # 4. 1x2x2 (Schoolbook, rank 4) -> A[2, 0:2] * B[0:2, 0:2]
    # 5. 1x2x1 (Schoolbook, rank 2) -> A[2, 0:2] * B[0:2, 2]
    # 6. 1x1x2 (Schoolbook, rank 2) -> A[2, 2] * B[2, 0:2]
    # 7. 1x1x1 (Schoolbook, rank 1) -> A[2, 2] * B[2, 2]
    
    U_list = []
    V_list = []
    W_list = []
    
    # 1. 2x2x2
    U1, V1, W1 = _strassen_222()
    # Embed into 3x3 (indices 0,1)
    Ue = np.zeros((9, 7)); Ve = np.zeros((9, 7)); We = np.zeros((9, 7))
    Ue[0:4, :] = U1
    Ve[0:4, :] = V1
    We[0:4, :] = W1
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    # 2. 2x1x2 (A rows 0,1; col 2; B row 2; cols 0,1)
    U2, V2, W2 = _schoolbook(2, 1, 2)
    Ue = np.zeros((9, 4)); Ve = np.zeros((9, 4)); We = np.zeros((9, 4))
    # A indices: 0,1 (rows) x 2 (col) -> 0*1+2=2, 1*1+2=3
    Ue[2:4, :] = U2
    # B indices: 2 (row) x 0,1 (cols) -> 2*2+0=4, 2*2+1=5
    Ve[4:6, :] = V2
    # W indices: 0,1 (cols) x 0,1 (rows) -> 0*3+0=0, 1*3+0=3, 0*3+1=1, 1*3+1=4
    # W is p x n. p=2, n=2.
    # W2 is 2x2.
    We[0:4, :] = W2
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    # 3. 2x1x1 (A rows 0,1; col 2; B row 2; col 2)
    U3, V3, W3 = _schoolbook(2, 1, 1)
    Ue = np.zeros((9, 2)); Ve = np.zeros((9, 2)); We = np.zeros((9, 2))
    Ue[2:4, :] = U3
    Ve[6:7, :] = V3 # B[2,2] is index 2*3+2=8? No, B is 3x3. m=1, p=1.
    # Wait, _schoolbook(2,1,1) returns V of size 1x1.
    # B index in 3x3: row 2, col 2 -> 2*3+2 = 8.
    Ve[8:9, :] = V3
    We[6:7, :] = W3 # W[2, 0:2] -> 2*3+0=6, 2*3+1=7.
    # W3 is 1x2.
    We[6:8, :] = W3
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    # 4. 1x2x2 (A row 2; cols 0,1; B rows 0,1; cols 0,1)
    U4, V4, W4 = _schoolbook(1, 2, 2)
    Ue = np.zeros((9, 4)); Ve = np.zeros((9, 4)); We = np.zeros((9, 4))
    # A[2, 0:2] -> 2*3+0=6, 2*3+1=7
    Ue[6:8, :] = U4
    # B[0:2, 0:2] -> 0,1,3,4
    Ve[0:5, :] = 0
    Ve[0:2, :] = V4[0:2, :]
    Ve[3:5, :] = V4[2:4, :]
    We[0:4, :] = W4
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    # 5. 1x2x1 (A row 2; cols 0,1; B rows 0,1; col 2)
    U5, V5, W5 = _schoolbook(1, 2, 1)
    Ue = np.zeros((9, 2)); Ve = np.zeros((9, 2)); We = np.zeros((9, 2))
    Ue[6:8, :] = U5
    Ve[2:4, :] = 0
    Ve[2:3, :] = V5[0:1, :]
    Ve[5:6, :] = V5[1:2, :]
    We[6:7, :] = W5
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    # 6. 1x1x2 (A row 2; col 2; B row 2; cols 0,1)
    U6, V6, W6 = _schoolbook(1, 1, 2)
    Ue = np.zeros((9, 2)); Ve = np.zeros((9, 2)); We = np.zeros((9, 2))
    Ue[8:9, :] = U6
    Ve[4:6, :] = 0
    Ve[4:5, :] = V6[0:1, :]
    Ve[5:6, :] = V6[1:2, :]
    We[0:2, :] = W6
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    # 7. 1x1x1 (A row 2; col 2; B row 2; col 2)
    U7, V7, W7 = _schoolbook(1, 1, 1)
    Ue = np.zeros((9, 1)); Ve = np.zeros((9, 1)); We = np.zeros((9, 1))
    Ue[8:9, :] = U7
    Ve[8:9, :] = V7
    We[8:9, :] = W7
    U_list.append(Ue); V_list.append(Ve); W_list.append(We)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    # Split m=4 into 1+3
    # Part 1: 3x1x7
    U1, V1, W1 = _schoolbook(3, 1, 7)
    U_list.append(U1)
    V_list.append(V1)
    W_list.append(W1)
    
    # Part 2: 3x3x7
    # Split p=7 into 3+4
    # Part 2a: 3x3x3
    U2a, V2a, W2a = _laderman_333_approx()
    U_list.append(U2a)
    V_list.append(V2a)
    W_list.append(W2a)
    
    # Part 2b: 3x3x4
    # Split m=3 into 1+2
    # Part 2b-i: 3x1x4
    U2b_i, V2b_i, W2b_i = _schoolbook(3, 1, 4)
    U_list.append(U2b_i)
    V_list.append(V2b_i)
    W_list.append(W2b_i)
    
    # Part 2b-ii: 3x2x4
    # Split n=3 into 1+2
    # Part 2b-ii-a: 1x2x4
    U2b_ii_a, V2b_ii_a, W2b_ii_a = _schoolbook(1, 2, 4)
    U_list.append(U2b_ii_a)
    V_list.append(V2b_ii_a)
    W_list.append(W2b_ii_a)
    
    # Part 2b-ii-b: 2x2x4
    # Two 2x2x2 Strassen
    U_s1, V_s1, W_s1 = _strassen_222()
    U_s2, V_s2, W_s2 = _strassen_222()
    U_list.append(U_s1)
    V_list.append(V_s1)
    W_list.append(W_s1)
    U_list.append(U_s2)
    V_list.append(V_s2)
    W_list.append(W_s2)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
