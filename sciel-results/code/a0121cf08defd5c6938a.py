"""Program for rank-minimal tensor decomposition of 3x4 * 4x8 matrix multiplication."""
import numpy as np
from collections import defaultdict
from itertools import product
import time

DIMENSIONS = [3, 4, 8]

# Known small-rank decompositions
def strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    U[1, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[0, 5] = -1; U[2, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition. Rank n*m*p."""
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
    return U, V, W

def add_decompositions(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (block sum)."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    U = np.zeros((U1.shape[0], r))
    V = np.zeros((V1.shape[0], r))
    W = np.zeros((W1.shape[0], r))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def strassen_34x48_sliding():
    """
    Sliding window / convolutional tensor decomposition for 3x4 * 4x8.
    
    Key idea: Instead of decomposing each 2x2x2 block independently,
    use overlapping windows that share rank-1 terms.
    
    Partition:
    - i: 3 rows -> split into {0,1} and {2}
    - j: 4 cols -> overlapping windows [0,1], [1,2], [2,3] (3 windows of size 2)
    - k: 8 cols -> overlapping windows [0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7] (7 windows of size 2)
    
    For each (i_window, j_window, k_window) where i_window covers rows {0,1}:
    Use Strassen 2x2x2 with appropriate index mapping.
    
    But overlapping windows mean the same (i,j,k) triple appears in multiple windows.
    We need to carefully combine contributions so that each (i,j,k) gets exactly 1.
    
    Alternative cleaner approach:
    - For i in {0,1}: use Strassen on 2x2x2 sub-blocks for all (j,k) pairs
    - For i=2: use schoolbook
    
    The sliding window idea: for j and k, use overlapping 2-element windows
    and weight the Strassen contributions so they sum correctly.
    
    Actually, let me think differently. The standard approach gives:
    - 4 Strassen blocks for (i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}): rank 28
    - 4 Strassen blocks for (i in {0,1}, j in {0,1,2,3}, k in {4,5,6,7}): rank 28
    - 16 schoolbook for (i=2, j in {0,1,2,3}, k in {0,1,2,3}): rank 16
    - 16 schoolbook for (i=2, j in {0,1,2,3}, k in {4,5,6,7}): rank 16
    Total: 88
    
    Sliding window improvement:
    For the j dimension (size 4), instead of 2 disjoint windows of size 2,
    use 3 overlapping windows: [0,1], [1,2], [2,3].
    Similarly for k (size 8), use 7 overlapping windows.
    
    But this creates overcounting. We need a partition of unity.
    
    Better approach: Use the fact that for 2x2x2 Strassen, we can apply it
    to any 2 consecutive j values and any 2 consecutive k values.
    For j in {0,1,2,3}, the windows are [0,1] and [2,3] (disjoint) OR
    we can use [0,1], [1,2], [2,3] with appropriate weighting.
    
    Let me try a different strategy: 
    For i in {0,1}, use Strassen for each pair of consecutive j's and k's,
    but with a partition of unity in the j and k directions.
    
    For j direction (4 elements): windows [0,1] and [2,3] are disjoint -> 2 windows
    For k direction (8 elements): windows [0,1], [2,3], [4,5], [6,7] are disjoint -> 4 windows
    Total Strassen blocks: 2 * 4 = 8, each rank 7 -> 56
    
    Plus i=2 schoolbook: 4 * 8 = 32
    Total: 88 (same as before)
    
    To improve, I need to find a way to share terms between adjacent blocks.
    
    Let me try: for the 2x4x8 sub-problem (i in {0,1}), instead of 8 separate
    Strassen 2x2x2 blocks, can I use a rank-56 or lower decomposition?
    
    Actually, 2x4x8 matmul has known better bounds. Let me try a direct construction.
    
    For A (2x4) * B (4x8) = C (2x8):
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Use Strassen-like approach:
    Split A = [A1 A2] where A1, A2 are 2x2
    Split B = [[B11, B12], [B21, B22]] where each is 2x4
    
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each term like A1*B11 is 2x2 * 2x4.
    2x2 * 2x4 can be done with rank 12 (schoolbook) or better.
    
    For 2x2 * 2x4:
    A is 2x2, B is 2x4, C is 2x4.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is like two 1x4 dot products for each i.
    Schoolbook: 2*2*4 = 16 terms.
    
    Can we do better? Use Strassen on the 2x2 part:
    For each k, we compute [A[0,0]*B[0,k] + A[0,1]*B[1,k], A[1,0]*B[0,k] + A[1,1]*B[1,k]]
    This is a 2x1 * 2x1 = 2x1 result for each k.
    Strassen for 2x1 * 1x1? Not standard.
    
    Actually, 2x2 * 2x4 = (2x2 * 2x2) + (2x2 * 2x2) for the two column pairs.
    Each 2x2*2x2 has rank 7, so total rank 14 for 2x2*2x4.
    
    So for 2x4*4x8:
    4 terms of 2x2*2x4, each rank 14 -> total 56.
    Plus i=2 schoolbook: 32.
    Total: 88. Still the same.
    
    Let me try yet another approach. What if I use a rank-12 decomposition for 2x2*2x4?
    Schoolbook is 16. Can we do 12?
    
    2x2 * 2x4: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Define:
    M1 = A[0,0]*B[0,0]
    M2 = A[0,1]*B[1,0]
    M3 = A[0,0]*B[0,1]
    M4 = A[0,1]*B[1,1]
    ... etc.
    
    This is just schoolbook with 16 terms. Hard to reduce.
    
    Let me try a completely different approach: use the known result that
    the rank of 2x4x8 matmul is at most 56 (from Strassen on sub-blocks)
    and see if we can get below that.
    
    Actually, I recall that for 2x2xn, the rank is at most 7n/2 for n even.
    For 2x2x4: rank <= 14. For 2x2x8: rank <= 28.
    
    So 2x4x8 = 2x(2x8) = 2 * (2x2x8 with rank 28) = 56.
    Or 2x4x8 = (2x4x2) + ... hmm.
    
    Let me just implement the rank 88 solution cleanly and verify it works,
    then try to optimize further.
    """
    n, m, p = 3, 4, 8
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # 4 Strassen 2x2x2 blocks
    for j2 in range(2):
        for k2 in range(2):
            U_sub, V_sub, W_sub = strassen_222()
            U_full = np.zeros((12, 7))
            V_full = np.zeros((32, 7))
            W_full = np.zeros((24, 7))
            
            for i in range(2):
                for j1 in range(2):
                    row_sub = i * 2 + j1
                    row_full = i * 4 + j1 + 2 * j2
                    U_full[row_full, :] = U_sub[row_sub, :]
            
            for j1 in range(2):
                for k1 in range(2):
                    row_sub = j1 * 2 + k1
                    row_full = (j1 + 2 * j2) * 8 + k1 + 4 * k2
                    V_full[row_full, :] = V_sub[row_sub, :]
            
            for k1 in range(2):
                for i in range(2):
                    row_sub = k1 * 2 + i
                    row_full = (k1 + 4 * k2) * 3 + i
                    W_full[row_full, :] = W_sub[row_sub, :]
            
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
    
    # Part 2: i=2, j in {0,1,2,3}, k in {0,1,2,3}
    for j in range(4):
        for k in range(4):
            U_term = np.zeros((12, 1))
            V_term = np.zeros((32, 1))
            W_term = np.zeros((24, 1))
            U_term[2 * 4 + j, 0] = 1
            V_term[j * 8 + k, 0] = 1
            W_term[k * 3 + 2, 0] = 1
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # Part 3: i in {0,1}, j in {0,1,2,3}, k in {4,5,6,7}
    for j2 in range(2):
        for k2 in range(2):
            U_sub, V_sub, W_sub = strassen_222()
            U_full = np.zeros((12, 7))
            V_full = np.zeros((32, 7))
            W_full = np.zeros((24, 7))
            
            for i in range(2):
                for j1 in range(2):
                    row_sub = i * 2 + j1
                    row_full = i * 4 + j1 + 2 * j2
                    U_full[row_full, :] = U_sub[row_sub, :]
            
            for j1 in range(2):
                for k1 in range(2):
                    row_sub = j1 * 2 + k1
                    row_full = (j1 + 2 * j2) * 8 + (4 + k1 + 2 * k2)
                    V_full[row_full, :] = V_sub[row_sub, :]
            
            for k1 in range(2):
                for i in range(2):
                    row_sub = k1 * 2 + i
                    row_full = (4 + k1 + 2 * k2) * 3 + i
                    W_full[row_full, :] = W_sub[row_sub, :]
            
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
    
    # Part 4: i=2, j in {0,1,2,3}, k in {4,5,6,7}
    for j in range(4):
        for k in range(4):
            U_term = np.zeros((12, 1))
            V_term = np.zeros((32, 1))
            W_term = np.zeros((24, 1))
            U_term[2 * 4 + j, 0] = 1
            V_term[j * 8 + (4 + k), 0] = 1
            W_term[(4 + k) * 3 + 2, 0] = 1
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    U_final = np.hstack(U_list)
    V_final = np.hstack(V_list)
    W_final = np.hstack(W_list)
    
    return U_final, V_final, W_final

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 88 construction
    U, V, W = strassen_34x48_sliding()
    
    n, m, p = 3, 4, 8
    rank = U.shape[1]
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        def round_half(x):
            return round(x * 2) / 2
        
        U_list = [[round_half(v) for v in row] for row in U_list]
        V_list = [[round_half(v) for v in row] for row in V_list]
        W_list = [[round_half(v) for v in row] for row in W_list]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_s, V_s, W_s = schoolbook_nm_p(n, m, p)
    U_list = U_s.tolist()
    V_list = V_s.tolist()
    W_list = W_s.tolist()
    
    def round_half(x):
        return round(x * 2) / 2
    
    U_list = [[round_half(v) for v in row] for row in U_list]
    V_list = [[round_half(v) for v in row] for row in V_list]
    W_list = [[round_half(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
