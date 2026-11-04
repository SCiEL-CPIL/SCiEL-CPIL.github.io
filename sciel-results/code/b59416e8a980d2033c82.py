"""Tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact rank-71 tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses block decomposition: split the 3x5 * 5x5 into blocks and apply Strassen
    for 2x2x2 sub-problems where beneficial.
    
    The matrix C (3x5) = A (3x5) * B (5x5) can be decomposed as:
    - Split A into A_top (1x5) and A_bot (2x5)
    - Split B into B_left (5x2) and B_right (5x3)
    - C_top (1x5) = A_top * B (rank 5)
    - C_bot (2x5) = A_bot * B (rank 2*5*5 = 50 schoolbook, but we can do better)
    
    Actually, let me think more carefully. The total schoolbook rank is 3*5*5 = 75.
    
    A better approach: partition the 5x5 B matrix and the 3x5 A matrix.
    
    Let's partition:
    - A is 3x5, split into A1 (2x5) and A2 (1x5)
    - B is 5x5, split into B1 (5x2) and B2 (5x3)
    
    C = A*B = [A1; A2] * [B1 | B2]
    C_top_left (2x2) = A1 * B1  -> rank 8 schoolbook, but Strassen gives rank 7
    C_top_right (2x3) = A1 * B2  -> rank 12 schoolbook, can decompose as 2x2x2 + 2x2x1 + 2x2x1... 
        Actually 2x3 = 2x(2+1), so C_top_right = A1*B2_left(5x2) + A1*B2_right(5x1)
        A1*B2_left: 2x2x2 -> rank 7 (Strassen)
        A1*B2_right: 2x1x2 -> rank 4
        Total: 11
    C_bot_left (1x2) = A2 * B1  -> rank 4 schoolbook (1x2x2)
    C_bot_right (1x3) = A2 * B2  -> rank 6 schoolbook (1x2x3)
    
    Total: 7 + 11 + 4 + 6 = 28? That doesn't seem right. Let me reconsider.
    
    Wait, I need to be more careful. The tensor for matrix multiplication M_{n,m,p} has
    rank at most n*m*p (schoolbook). For M_{3,5,5}, that's 75.
    
    Let me use a different partition. Split the inner dimension (5) into 2+3:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    Split j into j1 (indices 0,1) and j2 (indices 2,3,4):
    C_{i,k} = sum_{j1} A_{i,j1} B_{j1,k} + sum_{j2} A_{i,j2} B_{j2,k}
    
    First term: M_{3,2,5} tensor
    Second term: M_{3,3,5} tensor
    
    M_{3,2,5}: schoolbook rank = 3*2*5 = 30
    M_{3,3,5}: schoolbook rank = 3*3*5 = 45
    Total: 75 (same as before)
    
    Let me try splitting differently. Use Strassen-like decomposition.
    
    Actually, for a proper improvement, let me try:
    Split n=3 as 2+1, m=5 as 2+3, p=5 as 2+3.
    
    This gives 8 sub-problems:
    M_{2,2,2}: rank 7 (Strassen)
    M_{2,2,3}: rank 11 (Strassen 2x2x2 + 2x2x1 = 7+4=11)
    M_{2,3,2}: rank 11 (Strassen 2x2x2 + 2x1x2 = 7+4=11)
    M_{2,3,3}: rank 17 (Strassen 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1 = 7+4+4+2=17)
    M_{1,2,2}: rank 4
    M_{1,2,3}: rank 6
    M_{1,3,2}: rank 6
    M_{1,3,3}: rank 9
    
    Total: 7+11+11+17+4+6+6+9 = 71
    
    This gives rank 71, which is better than 75!
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Partition sizes
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Build Strassen 2x2x2 decomposition
    # Strassen factors for M_{2,2,2}
    # U is 4x7, V is 4x7, W is 4x7
    # Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i
    
    def strassen_222():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = np.zeros((4, 7), dtype=np.int8)
        V = np.zeros((4, 7), dtype=np.int8)
        W = np.zeros((4, 7), dtype=np.int8)
        
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U factors (for A):
        # M1: A11+A22 -> U[:,0] = [1,0,0,1]
        # M2: A21+A22 -> U[:,1] = [0,0,1,1]
        # M3: A11     -> U[:,2] = [1,0,0,0]
        # M4: A22     -> U[:,3] = [0,0,0,1]
        # M5: A11+A12 -> U[:,4] = [1,1,0,0]
        # M6: A21-A11 -> U[:,5] = [-1,0,1,0]
        # M7: A12-A22 -> U[:,6] = [0,1,-1,0]
        
        U[0, 0] = 1; U[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        U[0, 2] = 1
        U[3, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        U[0, 5] = -1; U[2, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        
        # V factors (for B):
        # M1: B11+B22 -> V[:,0] = [1,0,0,1]
        # M2: B11     -> V[:,1] = [1,0,0,0]
        # M3: B12-B22 -> V[:,2] = [0,1,0,-1]
        # M4: B21-B11 -> V[:,3] = [-1,0,1,0]
        # M5: B22     -> V[:,4] = [0,0,0,1]
        # M6: B11+B12 -> V[:,5] = [1,1,0,0]
        # M7: B21+B22 -> V[:,6] = [0,0,1,1]
        
        V[0, 0] = 1; V[3, 0] = 1
        V[0, 1] = 1
        V[1, 2] = 1; V[3, 2] = -1
        V[0, 3] = -1; V[2, 3] = 1
        V[3, 4] = 1
        V[0, 5] = 1; V[1, 5] = 1
        V[2, 6] = 1; V[3, 6] = 1
        
        # W factors (for C):
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # W row = k*2+i, so C11 is row 0*2+0=0, C12 is row 0*2+1=1, C21 is row 1*2+0=2, C22 is row 1*2+1=3
        
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
        W[1, 2] = 1; W[1, 4] = 1
        W[2, 1] = 1; W[2, 3] = 1
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W
    
    def trivial_block(n_b, m_b, p_b):
        """Return U, V, W for trivial n_b x m_b x p_b decomposition."""
        rank = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, rank), dtype=np.int8)
        V = np.zeros((m_b * p_b, rank), dtype=np.int8)
        W = np.zeros((p_b * n_b, rank), dtype=np.int8)
        
        i_idx, j_idx, k_idx = np.indices((n_b, m_b, p_b))
        cols = np.arange(rank)
        u_rows = (i_idx * m_b + j_idx).ravel()
        v_rows = (j_idx * p_b + k_idx).ravel()
        w_rows = (k_idx * n_b + i_idx).ravel()
        
        U[u_rows, cols] = 1
        V[v_rows, cols] = 1
        W[w_rows, cols] = 1
        
        return U, V, W
    
    def strassen_plus_trivial(n_b, m_b, p_b):
        """
        Decompose M_{n_b, m_b, p_b} where one dimension is 2 and others are 2 or 3.
        Uses Strassen for the 2x2x2 part and trivial for the rest.
        """
        if n_b == 2 and m_b == 2 and p_b == 2:
            return strassen_222()
        
        # For cases like 2x2x3, 2x3x2, 2x3x3, 1x2x2, etc.
        # We decompose by splitting the larger dimension
        
        if n_b == 2 and m_b == 2 and p_b == 3:
            # M_{2,2,3} = M_{2,2,2} + M_{2,2,1}
            U1, V1, W1 = strassen_222()
            U2, V2, W2 = trivial_block(2, 2, 1)
            
            rank = 7 + 4
            U = np.zeros((4, rank), dtype=np.int8)
            V = np.zeros((6, rank), dtype=np.int8)
            W = np.zeros((6, rank), dtype=np.int8)
            
            U[:, :7] = U1
            U[:, 7:] = U2
            
            V[:4, :7] = V1
            V[:4, 7:] = V2
            V[4:6, 7:] = V2
            
            W[:4, :7] = W1
            W[:4, 7:] = W2
            W[4:6, 7:] = W2
            
            return U, V, W
        
        if n_b == 2 and m_b == 3 and p_b == 2:
            # M_{2,3,2} = M_{2,2,2} + M_{2,1,2}
            U1, V1, W1 = strassen_222()
            U2, V2, W2 = trivial_block(2, 1, 2)
            
            rank = 7 + 4
            U = np.zeros((6, rank), dtype=np.int8)
            V = np.zeros((6, rank), dtype=np.int8)
            W = np.zeros((4, rank), dtype=np.int8)
            
            U[:4, :7] = U1
            U[4:6, 7:] = U2
            
            V[:4, :7] = V1
            V[4:6, 7:] = V2
            
            W[:, :7] = W1
            W[:, 7:] = W2
            
            return U, V, W
        
        if n_b == 2 and m_b == 3 and p_b == 3:
            # M_{2,3,3} = M_{2,2,2} + M_{2,2,1} + M_{2,1,2} + M_{2,1,1}
            U1, V1, W1 = strassen_222()
            U2, V2, W2 = trivial_block(2, 2, 1)
            U3, V3, W3 = trivial_block(2, 1, 2)
            U4, V4, W4 = trivial_block(2, 1, 1)
            
            rank = 7 + 4 + 4 + 2
            U = np.zeros((6, rank), dtype=np.int8)
            V = np.zeros((9, rank), dtype=np.int8)
            W = np.zeros((6, rank), dtype=np.int8)
            
            U[:4, :7] = U1
            U[:4, 7:11] = U2
            U[4:6, 11:15] = U3
            U[4:6, 15:] = U4
            
            V[:4, :7] = V1
            V[:4, 7:11] = V2
            V[4:6, 11:15] = V3
            V[6:9, 15:] = V4
            
            W[:4, :7] = W1
            W[:4, 7:11] = W2
            W[4:6, 11:15] = W3
            W[4:6, 15:] = W4
            
            return U, V, W
        
        # Trivial cases
        return trivial_block(n_b, m_b, p_b)
    
    # Build the full decomposition
    # Block indices: (ni, mi, pi) where ni in {0,1}, mi in {0,1}, pi in {0,1}
    # ni=0 -> n1=2 rows, ni=1 -> n2=1 row
    # mi=0 -> m1=2 cols, mi=1 -> m2=3 cols
    # pi=0 -> p1=2 cols, pi=1 -> p2=3 cols
    
    blocks = []
    for ni in range(2):
        for mi in range(2):
            for pi in range(2):
                nb = n1 if ni == 0 else n2
                mb = m1 if mi == 0 else m2
                pb = p1 if pi == 0 else p2
                U_b, V_b, W_b = strassen_plus_trivial(nb, mb, pb)
                blocks.append((ni, mi, pi, nb, mb, pb, U_b, V_b, W_b))
    
    # Compute total rank
    total_rank = sum(U_b.shape[1] for _, _, _, _, _, _, U_b, _, _ in blocks)
    
    # Build global matrices
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)
    
    col_offset = 0
    for ni, mi, pi, nb, mb, pb, U_b, V_b, W_b in blocks:
        rank_b = U_b.shape[1]
        
        # Row offsets for U: i in [0, nb), j in [0, mb)
        # Global i = (n1 if ni==0 else 0) + i... no wait
        # ni=0: i in [0, 2), global i = i
        # ni=1: i in [0, 1), global i = 2 + i
        i_offset = n1 if ni == 0 else 0
        if ni == 0:
            i_offset = 0
        else:
            i_offset = n1
        
        # For U: row = i*m + j
        # i ranges over [i_offset, i_offset+nb)
        # j ranges over [j_offset, j_offset+mb)
        j_offset = m1 if mi == 0 else 0
        if mi == 0:
            j_offset = 0
        else:
            j_offset = m1
        
        for i_local in range(nb):
            for j_local in range(mb):
                u_row = (i_offset + i_local) * m + (j_offset + j_local)
                u_b_row = i_local * mb + j_local
                U[u_row, col_offset:col_offset+rank_b] = U_b[u_b_row]
        
        # For V: row = j*p + k
        # j ranges over [j_offset, j_offset+mb)
        # k ranges over [k_offset, k_offset+pb)
        k_offset = p1 if pi == 0 else 0
        if pi == 0:
            k_offset = 0
        else:
            k_offset = p1
        
        for j_local in range(mb):
            for k_local in range(pb):
                v_row = (j_offset + j_local) * p + (k_offset + k_local)
                v_b_row = j_local * pb + k_local
                V[v_row, col_offset:col_offset+rank_b] = V_b[v_b_row]
        
        # For W: row = k*n + i
        # k ranges over [k_offset, k_offset+pb)
        # i ranges over [i_offset, i_offset+nb)
        for k_local in range(pb):
            for i_local in range(nb):
                w_row = (k_offset + k_local) * n + (i_offset + i_local)
                w_b_row = k_local * nb + i_local
                W[w_row, col_offset:col_offset+rank_b] = W_b[w_b_row]
        
        col_offset += rank_b
    
    # Verify deadline
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
