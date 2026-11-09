"""Rewritten program for matrix multiplication 4x5 by 5x5 tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def get_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank is 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Mapping:
    # A indices: 00(0), 01(1), 10(2), 11(3)
    # B indices: 00(0), 01(1), 10(2), 11(3)
    # C indices: 00(0), 01(1), 10(2), 11(3)
    
    # M1
    # A: A11(0)+A22(3), B: B11(0)+B22(3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C: C11(0)+C22(3)
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    # A: A21(2)+A22(3), B: B11(0)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C: C21(2)+C22(3) -> C21 gets +M2, C22 gets -M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    # A: A11(0), B: B21(2)-B22(3)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C: C12(1)+C22(3)
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    # A: A22(3), B: B12(1)-B11(0)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C: C11(0)+C21(2)
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    # A: A11(0)+A12(1), B: B22(3)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C: C11(0)-C12(1)? No, C11 = M1+M4-M5+M7. C12 = M3+M5.
    # So C11 gets -M5, C12 gets +M5.
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    # A: A21(2)-A11(0), B: B11(0)+B12(1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C: C22(3)
    W[3, 5] = 1
    
    # M7
    # A: A12(1)-A22(3), B: B21(2)+B22(3)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C: C11(0)
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def get_trivial(n, m, p):
    """
    Returns U, V, W for trivial decomposition of n x m x p.
    Rank is n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def decompose(n, m, p):
    """
    Recursively decompose n x m x p matrix multiplication.
    Returns U, V, W as lists of lists.
    """
    # Base case: 2x2x2 use Strassen
    if n == 2 and m == 2 and p == 2:
        return get_strassen_222()
    
    # Base case: if any dimension is 1, trivial is optimal (rank = product of others)
    # Actually, if m=1, rank = n*p. Trivial gives n*1*p = n*p.
    # If n=1, rank = m*p.
    # If p=1, rank = n*m.
    if m == 1 or n == 1 or p == 1:
        return get_trivial(n, m, p)
    
    # Heuristic for splitting
    # Try to split m if m > 2
    # Prefer splits that result in smaller sub-problems that might use Strassen
    # For m=5, split 2, 3
    # For m=3, split 2, 1 (1 is trivial)
    # For m=4, split 2, 2 (both Strassen)
    
    splits_m = []
    if m >= 2:
        # Try to split into 2 and rest, or balanced
        if m == 5:
            splits_m = [(2, 3), (3, 2)]
        elif m == 3:
            splits_m = [(2, 1), (1, 2)]
        elif m == 4:
            splits_m = [(2, 2)]
        elif m > 4:
            # Try 2, m-2 or 3, m-3
            splits_m = [(2, m-2), (3, m-3)]
            
    best_rank = float('inf')
    best_result = None
    
    # Try splitting m
    for m1, m2 in splits_m:
        if m1 <= 0 or m2 <= 0: continue
        U1, V1, W1 = decompose(n, m1, p)
        U2, V2, W2 = decompose(n, m2, p)
        
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        total_rank = r1 + r2
        
        if total_rank < best_rank:
            # Embed U1, V1, W1 and U2, V2, W2 into global U, V, W
            global_U = [[0] * total_rank for _ in range(n * m)]
            global_V = [[0] * total_rank for _ in range(m * p)]
            global_W = [[0] * total_rank for _ in range(p * n)]
            
            # Embed Part 1 (j in [0, m1))
            for i in range(n):
                for j in range(m1):
                    local_u_idx = i * m1 + j
                    global_u_idx = i * m + j
                    for c in range(r1):
                        global_U[global_u_idx][c] = U1[local_u_idx][c]
            
            for j in range(m1):
                for k in range(p):
                    local_v_idx = j * p + k
                    global_v_idx = j * p + k
                    for c in range(r1):
                        global_V[global_v_idx][c] = V1[local_v_idx][c]
            
            for k in range(p):
                for i in range(n):
                    local_w_idx = k * n + i
                    global_w_idx = k * n + i
                    for c in range(r1):
                        global_W[global_w_idx][c] = W1[local_w_idx][c]
                        
            # Embed Part 2 (j in [m1, m))
            offset_c = r1
            for i in range(n):
                for j in range(m2):
                    local_u_idx = i * m2 + j
                    global_u_idx = i * m + (m1 + j)
                    for c in range(r2):
                        global_U[global_u_idx][c + offset_c] = U2[local_u_idx][c]
            
            for j in range(m2):
                for k in range(p):
                    local_v_idx = j * p + k
                    global_v_idx = (m1 + j) * p + k
                    for c in range(r2):
                        global_V[global_v_idx][c + offset_c] = V2[local_v_idx][c]
            
            for k in range(p):
                for i in range(n):
                    local_w_idx = k * n + i
                    global_w_idx = k * n + i
                    for c in range(r2):
                        global_W[global_w_idx][c + offset_c] = W2[local_w_idx][c]
                        
            best_rank = total_rank
            best_result = (global_U, global_V, global_W)
            
    # If no split of m was better or possible, try splitting n or p?
    # For now, if m split didn't work, fall back to trivial
    if best_result is None:
        return get_trivial(n, m, p)
        
    return best_result

def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x5 matrix by a 5x5 matrix.
    """
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
