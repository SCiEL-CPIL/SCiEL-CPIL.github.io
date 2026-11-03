"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 150s kill

    # Strategy: Split the 2x4x7 multiplication into independent sub-problems.
    # We split the middle dimension (m=4) into two blocks of size 2: j in {0,1} and j in {2,3}.
    # We split the output dimension (p=7) into three blocks: k in {0,1,2}, {3,4,5}, {6}.
    # 
    # For each j-block (size 2) and each k-block:
    #   - If k-block size is 3: we have a 2x2x3 sub-problem.
    #   - If k-block size is 1: we have a 2x2x1 sub-problem.
    #
    # Wait, the tensor is T_{i,j,k} = sum_{j} A_{i,j} B_{j,k}.
    # The decomposition is: sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = delta_{j,j'}
    # Actually, the identity is:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    # This means the "middle" index j in U and V must match, and it's summed over c.
    # But the delta is on j and j'. This is the standard matrix multiplication tensor.
    #
    # Let's use a Kronecker-like product structure or just direct block decomposition.
    #
    # Current best known: Rank 50.
    # Let's try to see if we can do better.
    #
    # Alternative: Use the fact that 2x4x7 = (2x2x2) x (2x1x1) x (1x1x1) ? No.
    #
    # Let's stick to the block decomposition which is known to be correct.
    # j-blocks: [0,2), [2,4)
    # k-blocks: [0,3), [3,6), [6,7)
    #
    # For j-block [0,2) and k-block [0,3): sub-problem is 2x2x3.
    #   Rank of 2x2x3?
    #   We can split k into 2+1.
    #   2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    #   Or maybe there's a better 2x2x3?
    #   The trivial rank is 12.
    #   Using Strassen for 2x2x2 and schoolbook for 2x2x1 gives 7+4=11.
    #
    # For j-block [0,2) and k-block [3,6): sub-problem is 2x2x3.
    #   Similarly, rank 11.
    #
    # For j-block [0,2) and k-block [6,7): sub-problem is 2x2x1.
    #   Rank 4.
    #
    # Total for j-block [0,2): 11 + 11 + 4 = 26.
    #
    # For j-block [2,4) and k-block [0,3): sub-problem is 2x2x3.
    #   Rank 11.
    #
    # For j-block [2,4) and k-block [3,6): sub-problem is 2x2x3.
    #   Rank 11.
    #
    # For j-block [2,4) and k-block [6,7): sub-problem is 2x2x1.
    #   Rank 4.
    #
    # Total for j-block [2,4): 11 + 11 + 4 = 26.
    #
    # Total Rank = 26 + 26 = 52.
    #
    # This is worse than the previous 50. Why?
    # Previous:
    # j-blocks: [0,2), [2,4)
    # k-blocks: [0,2), [2,4), [4,6), [6,7)
    # Sizes: 2, 2, 2, 1.
    #
    # For each j-block:
    #   3 sub-problems of 2x2x2 (rank 7 each) -> 21
    #   1 sub-problem of 2x2x1 (rank 4) -> 4
    #   Total per j-block: 25.
    #   Total: 50.
    #
    # So the previous split was better. Let's use the previous split.
    #
    # j-blocks: [0,2), [2,4)
    # k-blocks: [0,2), [2,4), [4,6), [6,7)
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    # Pre-compute sub-decompositions
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, k_type in k_blocks:
            if k_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            
            # Place U
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            # Place V
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            # Place W
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    # Verify
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails
            r = n * m * p
            U_fb = [[0] * r for _ in range(n * m)]
            V_fb = [[0] * r for _ in range(m * p)]
            W_fb = [[0] * r for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

# EVOLVE-BLOCK-END
