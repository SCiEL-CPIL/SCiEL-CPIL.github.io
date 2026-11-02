"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy: Decompose into blocks using Kronecker product structure.
    - Split m=4 into two blocks of size 2
    - Split p=5 into one block of size 2 and one block of size 1 (or 2+2+1, etc.)
    
    For each (n_block, m_block, p_block) sub-problem, use known optimal decompositions:
    - 2x2x2: Strassen, rank 7
    - 2x2x1: schoolbook, rank 4 (or can we do better? 2*2*1=4 is schoolbook)
    - 2x1x2: schoolbook, rank 4
    - 1x2x2: schoolbook, rank 4
    - 2x1x1: rank 2
    - 1x2x1: rank 2
    - 1x1x2: rank 2
    - 1x1x1: rank 1
    
    For 2x4x5:
    Option A: Split m into 2+2, p into 2+2+1
    - Block (2,2,2): rank 7 (Strassen)
    - Block (2,2,2): rank 7 (Strassen)  
    - Block (2,2,1): rank 4 (schoolbook)
    Total: 7+7+4 = 18
    
    Wait, let me reconsider. The matrix multiplication A(2x4) * B(4x5) = C(2x5).
    
    We can think of this as: for each i in 0..1, j in 0..3, k in 0..4:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor has indices (i, j, k) with i in 0..1, j in 0..3, k in 0..4.
    
    We can partition the j dimension and k dimension.
    
    Let's partition j into two groups of size 2: J1 = {0,1}, J2 = {2,3}
    Let's partition k into groups: K1 = {0,1}, K2 = {2,3}, K3 = {4}
    
    Then we have sub-tensors:
    - (i in 0..1, j in J1, k in K1): 2x2x2 problem -> Strassen rank 7
    - (i in 0..1, j in J1, k in K2): 2x2x2 problem -> Strassen rank 7
    - (i in 0..1, j in J1, k in K3): 2x2x1 problem -> schoolbook rank 4
    - (i in 0..1, j in J2, k in K1): 2x2x2 problem -> Strassen rank 7
    - (i in 0..1, j in J2, k in K2): 2x2x2 problem -> Strassen rank 7
    - (i in 0..1, j in J2, k in K3): 2x2x1 problem -> schoolbook rank 4
    
    Total rank: 7+7+4+7+7+4 = 36
    
    But we can do better by sharing some computations. Actually, the standard approach is:
    
    For 2x4x5, we can use:
    - Split the 4 dimension into 2+2
    - For each 2x2x5 sub-problem, further split 5 into 2+2+1
    
    2x2x5: Split 5 into 2+2+1
    - 2x2x2: Strassen rank 7
    - 2x2x2: Strassen rank 7
    - 2x2x1: schoolbook rank 4
    Total for 2x2x5: 7+7+4 = 18
    
    Two such sub-problems: 18*2 = 36
    
    Hmm, that's the same. Let me think about whether there's a better decomposition.
    
    Actually, for 2x2x1, the optimal rank is 4 (it's just a dot product of two 2-vectors with a scalar output, so 2*2*1=4).
    
    Wait, 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    This is rank 4 schoolbook. Can we do better? The border rank of 2x2x1 is 4, so no.
    
    So 36 seems to be the answer with this approach. But maybe we can do better with a different partition.
    
    Let me try: split m=4 into 1+1+1+1, and use 2x1x5 blocks.
    2x1x5: A is 2x1, B is 1x5, C is 2x5.
    C[i,k] = A[i,0]*B[0,k] for i=0,1 and k=0..4
    This is rank 2*1*5 = 10 schoolbook.
    Four such: 40. Worse.
    
    Let me try: split m=4 into 2+2, and keep p=5 whole.
    2x2x5: What's the optimal rank?
    2x2x5: A is 2x2, B is 2x5, C is 2x5.
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    We can think of this as two 2x1x5 dot products combined.
    For fixed i: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    This is a sum of two rank-1 matrices (outer products of A[i,:] with rows of B).
    
    The tensor 2x2x5 has rank... Let me think. 
    Actually, 2x2x5 can be seen as: for each k, we have a 2x2 matrix multiplication where B is a 2x1 column.
    
    Alternative: view 2x2x5 as 5 copies of 2x2x1, but with shared A.
    
    Hmm, let me just use the block decomposition approach and aim for rank 36.
    
    Actually, I recall that for matrix multiplication, the rank of m_{n,p,q} where one dimension is small can sometimes be improved.
    
    Let me just implement the 36-rank decomposition and verify it works.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We'll build the decomposition using the block structure:
    # Split m into two blocks of size 2: M1 = {0,1}, M2 = {2,3}
    # Split p into three blocks: P1 = {0,1}, P2 = {2,3}, P3 = {4}
    
    # For each (M_block, P_block), we have a 2x2x|P_block| sub-problem.
    # - 2x2x2: Strassen (rank 7)
    # - 2x2x1: schoolbook (rank 4)
    
    # Let's define the Strassen 2x2x2 decomposition.
    # Strassen for C = A*B where A, B, C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor form, the U, V, W matrices for Strassen 2x2x2:
    # U has rows indexed by (i,j) for i,j in 0,1 -> 4 rows
    # V has rows indexed by (j',k) for j',k in 0,1 -> 4 rows  
    # W has rows indexed by (k',i') for k',i' in 0,1 -> 4 rows
    # Each has 7 columns.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Column 0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # Column 1: M2 = (A10+A11)B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10
        
        # Column 2: M3 = A00(B10-B11)
        U[0][2] = 1  # A00
        V[2][2] = 1  # B10
        V[3][2] = -1 # B11
        W[1][2] = 1  # C01
        W[3][2] = 1  # C11
        
        # Column 3: M4 = A11(B01-B00)
        U[3][3] = 1  # A11
        V[1][3] = 1  # B01
        V[0][3] = -1 # B00
        W[0][3] = 1  # C00
        W[3][3] = 1  # C11
        
        # Column 4: M5 = (A00+A01)B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        W[0][4] = -1 # C00
        W[1][4] = 1  # C01
        
        # Column 5: M6 = (A10-A00)(B00+B10)
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[2][5] = 1  # B10
        W[3][5] = 1  # C11
        
        # Column 6: M7 = (A01-A11)(B01+B11)
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[1][6] = 1  # B01
        V[3][6] = 1  # B11
        W[0][6] = 1  # C00
        W[1][6] = 1  # C01
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Return U, V, W for 2x2x1 schoolbook decomposition."""
        # A is 2x2, B is 2x1, C is 2x1
        # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # Col 0: A[0,0]*B[0,0]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Col 1: A[0,1]*B[1,0]
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        # Col 2: A[1,0]*B[0,0]
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1
        
        # Col 3: A[1,1]*B[1,0]
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1
        
        return U, V, W
    
    # Now assemble the full 2x4x5 decomposition.
    # U: 8 x rank (rows indexed by i*4+j, i in 0..1, j in 0..3)
    # V: 20 x rank (rows indexed by j*5+k, j in 0..3, k in 0..4)
    # W: 10 x rank (rows indexed by k*2+i, k in 0..4, i in 0..1)
    
    blocks = []
    # (m_offset, p_offset, p_size, is_strassen)
    # M1 = j in {0,1}, M2 = j in {2,3}
    # P1 = k in {0,1}, P2 = k in {2,3}, P3 = k in {4}
    for m_off in [0, 2]:
        for p_off, p_size in [(0, 2), (2, 2), (4, 1)]:
            if p_size == 2:
                blocks.append((m_off, p_off, 2, True))
            else:
                blocks.append((m_off, p_off, 1, False))
    
    total_rank = sum(7 if is_strassen else 4 for (_, _, _, is_strassen) in blocks)
    
    U = [[0]*total_rank for _ in range(8)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(10)]
    
    col = 0
    for m_off, p_off, p_size, is_strassen in blocks:
        if is_strassen:
            Us, Vs, Ws = strassen_2x2x2()
            # Map local indices to global
            # Local U row: i*2 + j_local, j_local in 0..1 -> global j = m_off + j_local
            # Local V row: j_local*2 + k_local, k_local in 0..1 -> global j = m_off + j_local, k = p_off + k_local
            # Local W row: k_local*2 + i -> global k = p_off + k_local, i
            for i in range(2):
                for j_local in range(2):
                    local_u = i * 2 + j_local
                    global_j = m_off + j_local
                    global_u = i * 4 + global_j
                    for c in range(7):
                        U[global_u][col + c] = Us[local_u][c]
            for j_local in range(2):
                for k_local in range(2):
                    local_v = j_local * 2 + k_local
                    global_j = m_off + j_local
                    global_k = p_off + k_local
                    global_v = global_j * 5 + global_k
                    for c in range(7):
                        V[global_v][col + c] = Vs[local_v][c]
            for k_local in range(2):
                for i in range(2):
                    local_w = k_local * 2 + i
                    global_k = p_off + k_local
                    global_w = global_k * 2 + i
                    for c in range(7):
                        W[global_w][col + c] = Ws[local_w][c]
            col += 7
        else:
            Us, Vs, Ws = schoolbook_2x2x1()
            # Local U row: i*2 + j_local, j_local in 0..1 -> global j = m_off + j_local
            # Local V row: j_local -> global j = m_off + j_local, k = p_off
            # Local W row: i -> global k = p_off, i
            for i in range(2):
                for j_local in range(2):
                    local_u = i * 2 + j_local
                    global_j = m_off + j_local
                    global_u = i * 4 + global_j
                    for c in range(4):
                        U[global_u][col + c] = Us[local_u][c]
            for j_local in range(2):
                local_v = j_local
                global_j = m_off + j_local
                global_k = p_off
                global_v = global_j * 5 + global_k
                for c in range(4):
                    V[global_v][col + c] = Vs[local_v][c]
            for i in range(2):
                local_w = i
                global_k = p_off
                global_w = global_k * 2 + i
                for c in range(4):
                    W[global_w][col + c] = Ws[local_w][c]
            col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
