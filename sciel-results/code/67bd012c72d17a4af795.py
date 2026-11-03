"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build the tensor from decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of 7 entries each, where each entry is a flat 2x2 matrix (4 elements).
    """
    U = []
    V = []
    W = []
    
    # Component 1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    U.append([1, 0, 0, 1])   # A: [[1,0],[0,1]]
    V.append([1, 0, 0, 1])   # B: [[1,0],[0,1]]
    W.append([1, 0, 0, 1])   # C: [[1,0],[0,1]]
    
    # Component 2: (A21+A22)B11 -> C21 += 1
    U.append([0, 0, 1, 1])   # A: [[0,0],[1,1]]
    V.append([1, 0, 0, 0])   # B: [[1,0],[0,0]]
    W.append([0, 0, 1, 0])   # C: [[0,0],[1,0]]
    
    # Component 3: A11(B21-B22) -> C12 += 1
    U.append([1, 0, 0, 0])   # A: [[1,0],[0,0]]
    V.append([0, 0, 1, -1])  # B: [[0,0],[1,-1]]
    W.append([0, 1, 0, 0])   # C: [[0,1],[0,0]]
    
    # Component 4: A22(B12-B11) -> C11 += 1, C21 += 1
    U.append([0, 0, 0, 1])   # A: [[0,0],[0,1]]
    V.append([-1, 1, 0, 0])  # B: [[-1,1],[0,0]]
    W.append([1, 0, 1, 0])   # C: [[1,0],[1,0]]
    
    # Component 5: (A11+A12)B22 -> C11 -= 1, C12 += 1
    U.append([1, 1, 0, 0])   # A: [[1,1],[0,0]]
    V.append([0, 0, 0, 1])   # B: [[0,0],[0,1]]
    W.append([-1, 1, 0, 0])  # C: [[-1,1],[0,0]]
    
    # Component 6: (A21-A11)(B11+B12) -> C22 += 1
    U.append([-1, 0, 1, 0])  # A: [[-1,0],[1,0]]
    V.append([1, 1, 0, 0])   # B: [[1,1],[0,0]]
    W.append([0, 0, 0, 1])   # C: [[0,0],[0,1]]
    
    # Component 7: (A12-A22)(B21+B22) -> C11 += 1
    U.append([0, 1, -1, 0])  # A: [[0,1],[-1,0]]
    V.append([0, 0, 1, 1])   # B: [[0,0],[1,1]]
    W.append([1, 0, 0, 0])   # C: [[1,0],[0,0]]
    
    return U, V, W

def _block_decomposition_356():
    """
    Block decomposition of 3x5x6 matrix multiplication.
    
    Split: n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    
    This gives 3x3 blocks in the output, 3x3 blocks in A (rows), 3x3 blocks in B (cols).
    
    Block (i,j,k) has dimensions n_i x m_j x p_k.
    
    For each block, if it's 2x2x2, use Strassen (rank 7).
    Otherwise, use schoolbook.
    
    Blocks:
    (0,0,0): 2x2x2 -> Strassen, rank 7
    (0,0,1): 2x2x2 -> Strassen, rank 7
    (0,0,2): 2x2x2 -> Strassen, rank 7
    (0,1,0): 2x2x2 -> Strassen, rank 7
    (0,1,1): 2x2x2 -> Strassen, rank 7
    (0,1,2): 2x2x2 -> Strassen, rank 7
    (0,2,0): 2x1x2 -> schoolbook, rank 4
    (0,2,1): 2x1x2 -> schoolbook, rank 4
    (0,2,2): 2x1x2 -> schoolbook, rank 4
    (1,0,0): 1x2x2 -> schoolbook, rank 4
    (1,0,1): 1x2x2 -> schoolbook, rank 4
    (1,0,2): 1x2x2 -> schoolbook, rank 4
    (1,1,0): 1x2x2 -> schoolbook, rank 4
    (1,1,1): 1x2x2 -> schoolbook, rank 4
    (1,1,2): 1x2x2 -> schoolbook, rank 4
    (1,2,0): 1x1x2 -> schoolbook, rank 2
    (1,2,1): 1x1x2 -> schoolbook, rank 2
    (1,2,2): 1x1x2 -> schoolbook, rank 2
    
    Total rank: 6*7 + 6*4 + 6*4 + 3*2 = 42 + 24 + 24 + 6 = 96? That's worse.
    
    Wait, let me reconsider. The splits should be:
    n=3: [2,1]
    m=5: [2,2,1]  
    p=6: [2,2,2]
    
    Number of blocks: 2*3*3 = 18 blocks.
    
    For 2x2x2 blocks: use Strassen (rank 7)
    For other blocks: use schoolbook.
    
    Let me count more carefully:
    Blocks with n_i=2, m_j=2, p_k=2: these are 2x2x2, use Strassen.
    How many? i in {0}, j in {0,1}, k in {0,1,2} -> 1*2*3 = 6 blocks, each rank 7 -> 42
    
    Blocks with n_i=2, m_j=1, p_k=2: 2x1x2, schoolbook rank 4.
    i in {0}, j in {2}, k in {0,1,2} -> 1*1*3 = 3 blocks, each rank 4 -> 12
    
    Blocks with n_i=1, m_j=2, p_k=2: 1x2x2, schoolbook rank 4.
    i in {1}, j in {0,1}, k in {0,1,2} -> 1*2*3 = 6 blocks, each rank 4 -> 24
    
    Blocks with n_i=1, m_j=1, p_k=2: 1x1x2, schoolbook rank 2.
    i in {1}, j in {2}, k in {0,1,2} -> 1*1*3 = 3 blocks, each rank 2 -> 6
    
    Total: 42 + 12 + 24 + 6 = 84
    
    That's better than 90!
    """
    n_splits = [2, 1]  # n=3 split into 2+1
    m_splits = [2, 2, 1]  # m=5 split into 2+2+1
    p_splits = [2, 2, 2]  # p=6 split into 2+2+2
    
    n_total = sum(n_splits)  # 3
    m_total = sum(m_splits)  # 5
    p_total = sum(p_splits)  # 6
    
    # Compute offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    # Collect all blocks and their decompositions
    U_cols = []  # Each entry: list of (row_idx, value) for this component
    V_cols = []
    W_cols = []
    
    for i_idx, n_i in enumerate(n_splits):
        for j_idx, m_j in enumerate(m_splits):
            for k_idx, p_k in enumerate(p_splits):
                n_off = n_offsets[i_idx]
                m_off = m_offsets[j_idx]
                p_off = p_offsets[k_idx]
                
                if n_i == 2 and m_j == 2 and p_k == 2:
                    # Use Strassen
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for c in range(7):
                        u_flat = U_s[c]  # 4 elements: [u00, u01, u10, u11]
                        v_flat = V_s[c]
                        w_flat = W_s[c]
                        
                        U_col = [0] * (n_total * m_total)
                        V_col = [0] * (m_total * p_total)
                        W_col = [0] * (p_total * n_total)
                        
                        # Fill U: n_i x m_j block at offset (n_off, m_off)
                        U_col[(n_off + 0) * m_total + (m_off + 0)] = u_flat[0]
                        U_col[(n_off + 0) * m_total + (m_off + 1)] = u_flat[1]
                        U_col[(n_off + 1) * m_total + (m_off + 0)] = u_flat[2]
                        U_col[(n_off + 1) * m_total + (m_off + 1)] = u_flat[3]
                        
                        # Fill V: m_j x p_k block at offset (m_off, p_off)
                        V_col[(m_off + 0) * p_total + (p_off + 0)] = v_flat[0]
                        V_col[(m_off + 0) * p_total + (p_off + 1)] = v_flat[1]
                        V_col[(m_off + 1) * p_total + (p_off + 0)] = v_flat[2]
                        V_col[(m_off + 1) * p_total + (p_off + 1)] = v_flat[3]
                        
                        # Fill W: p_k x n_i block at offset (p_off, n_off)
                        W_col[(p_off + 0) * n_total + (n_off + 0)] = w_flat[0]
                        W_col[(p_off + 0) * n_total + (n_off + 1)] = w_flat[1]
                        W_col[(p_off + 1) * n_total + (n_off + 0)] = w_flat[2]
                        W_col[(p_off + 1) * n_total + (n_off + 1)] = w_flat[3]
                        
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                else:
                    # Schoolbook for this block
                    for ii in range(n_i):
                        for jj in range(m_j):
                            for kk in range(p_k):
                                U_col = [0] * (n_total * m_total)
                                V_col = [0] * (m_total * p_total)
                                W_col = [0] * (p_total * n_total)
                                
                                U_col[(n_off + ii) * m_total + (m_off + jj)] = 1
                                V_col[(m_off + jj) * p_total + (p_off + kk)] = 1
                                W_col[(p_off + kk) * n_total + (n_off + ii)] = 1
                                
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n_total * m_total)]
    V = [[0] * rank for _ in range(m_total * p_total)]
    W = [[0] * rank for _ in range(p_total * n_total)]
    
    for c in range(rank):
        for row, val in enumerate(U_cols[c]):
            if val != 0:
                U[row][c] = val
        for row, val in enumerate(V_cols[c]):
            if val != 0:
                V[row][c] = val
        for row, val in enumerate(W_cols[c]):
            if val != 0:
                W[row][c] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Try block decomposition first
    result = _block_decomposition_356()
    
    # Verify
    if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
