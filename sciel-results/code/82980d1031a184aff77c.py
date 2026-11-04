import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication with rank 7."""
    # Strassen's algorithm decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-c)(e+g)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U maps A indices (i, j) to factor indices
    # V maps B indices (j, k) to factor indices
    # W maps C indices (i, k) to factor indices
    
    # Indices:
    # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Factor 1: (a+d)(e+h) -> C11, -C21, C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1 # C11, C22 (indices 0, 3 in flat C? No, C is 2x2. 0,1,2,3)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    W[0, 0] = 1
    W[3, 0] = 1
    
    # Factor 2: (c+d)e -> C21, -C22
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # Factor 3: a(f-h) -> C12, C22
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # Factor 4: d(g-e) -> C11, C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # Factor 5: (a+b)h -> -C11, C12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # Factor 6: (c-d)(f+g) -> C22
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Factor 7: (b-c)(e+g) -> C11
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_2x1x3():
    """Returns U, V, W for 2x1x3 matrix multiplication with rank 6."""
    # A: 2x1, B: 1x3, C: 2x3
    # C[i, j] = A[i, 0] * B[0, j]
    # Rank 6
    U = np.zeros((2, 6))
    V = np.zeros((3, 6))
    W = np.zeros((6, 6))
    
    for i in range(2):
        for j in range(3):
            col = i * 3 + j
            U[i, col] = 1
            V[j, col] = 1
            W[i * 3 + j, col] = 1
            
    return U, V, W

def _decomp_2x2x1():
    """Returns U, V, W for 2x2x1 matrix multiplication with rank 4."""
    # A: 2x2, B: 2x1, C: 2x1
    # C[i, 0] = sum_k A[i, k] * B[k, 0]
    # Rank 4
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    for i in range(2):
        for k in range(2):
            col = i * 2 + k
            U[i * 2 + k, col] = 1
            V[k, col] = 1
            W[i, col] = 1
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Total rank calculation
    # Part 1 (j=0,1): 3 * Strassen(2x2x2) = 21
    # Part 2 (j=2,3,4): 2 * (2x3x3)
    #   2x3x3 = 2x1x3 (6) + 2x2x3
    #   2x2x3 = 2x2x1 (4) + 2x2x2 (7) = 11
    #   2x3x3 = 6 + 11 = 17
    #   Part 2 = 2 * 17 = 34
    # Total = 21 + 34 = 55
    
    total_rank = 55
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper to embed a decomposition
    def embed(U_sub, V_sub, W_sub, 
              i_offset, j_offset, k_offset, 
              n_sub, m_sub, p_sub):
        nonlocal col_offset
        rank_sub = U_sub.shape[1]
        
        # Map local indices to global indices
        # U: (i, j) -> i_offset + i, j_offset + j
        # V: (j, k) -> j_offset + j, k_offset + k
        # W: (k, i) -> k_offset + k, i_offset + i (Note: W index is k*n + i)
        
        for c in range(rank_sub):
            # U
            for i in range(n_sub):
                for j in range(m_sub):
                    if U_sub[i * m_sub + j, c] != 0:
                        U[(i_offset + i) * m + (j_offset + j), col_offset + c] = U_sub[i * m_sub + j, c]
                        
            # V
            for j in range(m_sub):
                for k in range(p_sub):
                    if V_sub[j * p_sub + k, c] != 0:
                        V[(j_offset + j) * p + (k_offset + k), col_offset + c] = V_sub[j * p_sub + k, c]
                        
            # W
            for k in range(p_sub):
                for i in range(n_sub):
                    if W_sub[k * n_sub + i, c] != 0:
                        W[(k_offset + k) * n + (i_offset + i), col_offset + c] = W_sub[k * n_sub + i, c]
                        
        col_offset += rank_sub

    # --- Part 1: j in {0, 1} (m_sub=2) ---
    # Split k into {0,1}, {2,3}, {4,5}
    for k_block in range(3):
        k_offset = k_block * 2
        if time.time() > deadline: break
        U_sub, V_sub, W_sub = _strassen_2x2x2()
        # n_sub=2, m_sub=2, p_sub=2
        # i_offset=0, j_offset=0, k_offset=k_offset
        embed(U_sub, V_sub, W_sub, 0, 0, k_offset, 2, 2, 2)

    # --- Part 2: j in {2, 3, 4} (m_sub=3) ---
    # Split k into {0,1,2} and {3,4,5}
    for k_block in range(2):
        k_offset = k_block * 3
        if time.time() > deadline: break
        
        # Decompose 2x3x3
        # Split j_sub (local 0,1,2) into j_sub=0 and j_sub in {1,2}
        
        # Sub-part 2a: j_sub=0 (m_sub=1), k_sub (p_sub=3)
        # 2x1x3
        U_sub, V_sub, W_sub = _decomp_2x1x3()
        # n_sub=2, m_sub=1, p_sub=3
        # i_offset=0, j_offset=2 (global j=2), k_offset=k_offset
        embed(U_sub, V_sub, W_sub, 0, 2, k_offset, 2, 1, 3)
        
        # Sub-part 2b: j_sub in {1, 2} (m_sub=2), k_sub (p_sub=3)
        # Split k_sub into k_sub=0 and k_sub in {1,2}
        
        # Sub-sub-part 2b1: k_sub=0 (p_sub=1)
        # 2x2x1
        U_sub, V_sub, W_sub = _decomp_2x2x1()
        # n_sub=2, m_sub=2, p_sub=1
        # i_offset=0, j_offset=3 (global j=3), k_offset=k_offset
        embed(U_sub, V_sub, W_sub, 0, 3, k_offset, 2, 2, 1)
        
        # Sub-sub-part 2b2: k_sub in {1, 2} (p_sub=2)
        # 2x2x2
        U_sub, V_sub, W_sub = _strassen_2x2x2()
        # n_sub=2, m_sub=2, p_sub=2
        # i_offset=0, j_offset=3 (global j=3), k_offset=k_offset+1
        embed(U_sub, V_sub, W_sub, 0, 3, k_offset + 1, 2, 2, 2)

    return {"U": U, "V": V, "W": W}
