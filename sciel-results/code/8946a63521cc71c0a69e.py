import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140

def _embed(U, V, W, m, n, p, offset_m, offset_n, offset_p, M, N, P):
    """
    Embed a decomposition of size m x n x p into a larger M x N x P space.
    U: (m*n, rank)
    V: (n*p, rank)
    W: (p*m, rank)
    Returns new U, V, W of size (M*N, rank), (N*P, rank), (P*M, rank)
    """
    rank = len(U)
    new_U = [[0] * rank for _ in range(M * N)]
    new_V = [[0] * rank for _ in range(N * P)]
    new_W = [[0] * rank for _ in range(P * M)]
    
    # Map U
    for i in range(m):
        for j in range(n):
            global_i = offset_m + i
            global_j = offset_n + j
            local_i = i * n + j
            global_i_j = global_i * N + global_j
            for r in range(rank):
                new_U[global_i_j][r] = U[local_i][r]
                
    # Map V
    for j in range(n):
        for k in range(p):
            global_j = offset_n + j
            global_k = offset_p + k
            local_j = j * p + k
            global_j_k = global_j * P + global_k
            for r in range(rank):
                new_V[global_j_k][r] = V[local_j][r]
                
    # Map W
    # W index convention: W[k*m + i, r]
    for k in range(p):
        for i in range(m):
            global_k = offset_p + k
            global_i = offset_m + i
            local_k = k * m + i
            global_k_i = global_k * M + global_i
            for r in range(rank):
                new_W[global_k_i][r] = W[local_k][r]
                
    return new_U, new_V, new_W

def _concat_decomps(lists):
    """
    Concatenate a list of (U, V, W) tuples.
    Assumes they are already embedded in the same global space.
    """
    U = []
    V = []
    W = []
    for u, v, w in lists:
        U.extend(u)
        V.extend(v)
        W.extend(w)
    return U, V, W

def _strassen_222():
    # Rank 7 decomposition for 2x2x2
    # U (4x7), V (4x7), W (4x7)
    # Indices for U, V: i*2+j. W: k*2+i
    U = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    V = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [-1, 0, 1, 0],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    W = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, -1, 0, -1],# M2: C10+C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 0, 0, 1],# M5: C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    # Convert to list of lists if needed, but they are already
    return U, V, W

def _schoolbook_mnp(m, n, p):
    """
    Schoolbook decomposition for m x n x p.
    Rank m*n*p.
    """
    rank = m * n * p
    U = [[0] * rank for _ in range(m * n)]
    V = [[0] * rank for _ in range(n * p)]
    W = [[0] * rank for _ in range(p * m)]
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j][col] = 1
                V[j * p + k][col] = 1
                W[k * m + i][col] = 1
                col += 1
    return U, V, W

def _decompose_223():
    """
    Decompose 2x2x3 using Strassen for first 2 cols of B and Schoolbook for last 1 col.
    Rank 7 + 4 = 11.
    """
    U_s, V_s, W_s = _strassen_222()
    U_b, V_b, W_b = _schoolbook_mnp(2, 2, 1)
    
    # Embed into 2x2x3 space
    # Strassen is for B[:, 0:2]
    U1, V1, W1 = _embed(U_s, V_s, W_s, 2, 2, 2, 0, 0, 0, 2, 2, 3)
    # Schoolbook is for B[:, 2:3]
    U2, V2, W2 = _embed(U_b, V_b, W_b, 2, 2, 1, 0, 0, 2, 2, 2, 3)
    
    return _concat_decomps([(U1, V1, W1), (U2, V2, W2)])

def _decompose_423():
    """
    Decompose 4x2x3 using two 2x2x3 decompositions.
    Rank 11 + 11 = 22.
    """
    U1, V1, W1 = _decompose_223()
    # Embed first 2x2x3 into 4x2x3 at rows 0-1
    Ua, Va, Wa = _embed(U1, V1, W1, 2, 2, 3, 0, 0, 0, 4, 2, 3)
    # Embed second 2x2x3 into 4x2x3 at rows 2-3
    Ub, Vb, Wb = _embed(U1, V1, W1, 2, 2, 3, 2, 0, 0, 4, 2, 3)
    
    return _concat_decomps([(Ua, Va, Wa), (Ub, Vb, Wb)])

def _decompose_412():
    """
    Decompose 4x1x2. Rank 8.
    """
    return _schoolbook_mnp(4, 1, 2)

def _decompose_413():
    """
    Decompose 4x1x3. Rank 12.
    """
    return _schoolbook_mnp(4, 1, 3)

def _decompose_432():
    """
    Decompose 4x3x2.
    Split n=3 into 1+2.
    M_4,1,2 (Rank 8) + M_4,2,2 (Rank 14).
    Total 22.
    """
    U1, V1, W1 = _decompose_412()
    # Embed 4x1x2 into 4x3x2 at col 0
    Ua, Va, Wa = _embed(U1, V1, W1, 4, 1, 2, 0, 0, 0, 4, 3, 2)
    
    # M_4,2,2 is two 2x2x2 Strassen
    U_s, V_s, W_s = _strassen_222()
    # Embed first 2x2x2 at rows 0-1, cols 1-2
    Ub1, Vb1, Wb1 = _embed(U_s, V_s, W_s, 2, 2, 2, 0, 1, 0, 4, 3, 2)
    # Embed second 2x2x2 at rows 2-3, cols 1-2
    Ub2, Vb2, Wb2 = _embed(U_s, V_s, W_s, 2, 2, 2, 2, 1, 0, 4, 3, 2)
    
    return _concat_decomps([(Ua, Va, Wa), (Ub1, Vb1, Wb1), (Ub2, Vb2, Wb2)])

def _decompose_422():
    """
    Decompose 4x2x2. Rank 14.
    """
    U_s, V_s, W_s = _strassen_222()
    Ua, Va, Wa = _embed(U_s, V_s, W_s, 2, 2, 2, 0, 0, 0, 4, 2, 2)
    Ub, Vb, Wb = _embed(U_s, V_s, W_s, 2, 2, 2, 2, 0, 0, 4, 2, 2)
    return _concat_decomps([(Ua, Va, Wa), (Ub, Vb, Wb)])

def _decompose_452():
    """
    Decompose 4x5x2.
    Split n=5 into 2+3.
    M_4,2,2 (Rank 14) + M_4,3,2 (Rank 22).
    Total 36.
    """
    U1, V1, W1 = _decompose_422()
    Ua, Va, Wa = _embed(U1, V1, W1, 4, 2, 2, 0, 0, 0, 4, 5, 2)
    
    U2, V2, W2 = _decompose_432()
    Ub, Vb, Wb = _embed(U2, V2, W2, 4, 3, 2, 0, 2, 0, 4, 5, 2)
    
    return _concat_decomps([(Ua, Va, Wa), (Ub, Vb, Wb)])

def _decompose_433():
    """
    Decompose 4x3x3.
    Split n=3 into 1+2.
    M_4,1,3 (Rank 12) + M_4,2,3 (Rank 22).
    Total 34.
    """
    U1, V1, W1 = _decompose_413()
    Ua, Va, Wa = _embed(U1, V1, W1, 4, 1, 3, 0, 0, 0, 4, 3, 3)
    
    U2, V2, W2 = _decompose_423()
    Ub, Vb, Wb = _embed(U2, V2, W2, 4, 2, 3, 0, 1, 0, 4, 3, 3)
    
    return _concat_decomps([(Ua, Va, Wa), (Ub, Vb, Wb)])

def _decompose_453():
    """
    Decompose 4x5x3.
    Split n=5 into 2+3.
    M_4,2,3 (Rank 22) + M_4,3,3 (Rank 34).
    Total 56.
    """
    U1, V1, W1 = _decompose_423()
    Ua, Va, Wa = _embed(U1, V1, W1, 4, 2, 3, 0, 0, 0, 4, 5, 3)
    
    U2, V2, W2 = _decompose_433()
    Ub, Vb, Wb = _embed(U2, V2, W2, 4, 3, 3, 0, 2, 0, 4, 5, 3)
    
    return _concat_decomps([(Ua, Va, Wa), (Ub, Vb, Wb)])

def _decompose_455():
    """
    Decompose 4x5x5.
    Split p=5 into 2+3.
    M_4,5,2 (Rank 36) + M_4,5,3 (Rank 56).
    Total 92.
    """
    U1, V1, W1 = _decompose_452()
    Ua, Va, Wa = _embed(U1, V1, W1, 4, 5, 2, 0, 0, 0, 4, 5, 5)
    
    U2, V2, W2 = _decompose_453()
    Ub, Vb, Wb = _embed(U2, V2, W2, 4, 5, 3, 0, 0, 2, 4, 5, 5)
    
    return _concat_decomps([(Ua, Va, Wa), (Ub, Vb, Wb)])

def construct():
    global _start_time
    _start_time = time.time()
    
    # Dimensions 4x5x5
    U, V, W = _decompose_455()
    
    return {"U": U, "V": V, "W": W}
