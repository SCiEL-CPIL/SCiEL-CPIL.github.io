import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = DIMENSIONS
    
    # Global dimensions
    rank_total = 0
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We will build lists of non-zero entries for U, V, W
    # Or just build the matrices directly since rank is small (~51)
    
    # Helper to add a term to the decomposition
    # term: (u_indices, v_indices, w_indices, u_vals, v_vals, w_vals)
    # Actually, it's easier to define the local decompositions and map them.
    
    def get_local_decomp(n_loc, m_loc, p_loc):
        """
        Returns U, V, W for a local block of size n_loc x m_loc x p_loc.
        U shape: (n_loc*m_loc, rank_loc)
        V shape: (m_loc*p_loc, rank_loc)
        W shape: (p_loc*n_loc, rank_loc)
        """
        if n_loc == 2 and m_loc == 2 and p_loc == 2:
            # Strassen rank 7
            rank_loc = 7
            U = [[0]*rank_loc for _ in range(4)]
            V = [[0]*rank_loc for _ in range(4)]
            W = [[0]*rank_loc for _ in range(4)]
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            
            # M1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2: (A10+A11)B00 -> C10, C11
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = 1
            
            # M3: A00(B10-B11) -> C01
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[1][2] = 1
            
            # M4: A11(B01-B00) -> C00, C10
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            
            # M5: (A00+A01)B11 -> C00, C01
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1
            
            # M6: (A10-A00)(B00+B01) -> C11
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7: (A01-A11)(B10+B11) -> C00, C11
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[3][6] = 1
            
            return U, V, W

        else:
            # Schoolbook
            rank_loc = n_loc * m_loc * p_loc
            U = [[0]*rank_loc for _ in range(n_loc*m_loc)]
            V = [[0]*rank_loc for _ in range(m_loc*p_loc)]
            W = [[0]*rank_loc for _ in range(p_loc*n_loc)]
            col = 0
            for i in range(n_loc):
                for j in range(m_loc):
                    for k in range(p_loc):
                        U[i*m_loc+j][col] = 1
                        V[j*p_loc+k][col] = 1
                        W[k*n_loc+i][col] = 1
                        col += 1
            return U, V, W

    def get_split_decomp(n_loc, m_loc, p_loc):
        """
        Recursive or simple split to reduce rank further if possible.
        For now, we use the specific splits identified.
        """
        # If it's 2x2x2, use Strassen
        if n_loc == 2 and m_loc == 2 and p_loc == 2:
            return get_local_decomp(2, 2, 2)
        
        # Try to split if dimensions are composite
        # For this task, we pre-defined the splits.
        # Block 1: 2x3x4 -> Split m: 2,1; p: 2,2
        if n_loc == 2 and m_loc == 3 and p_loc == 4:
            # Split into (2,2,2), (2,1,2), (2,2,2), (2,1,2)
            # Indices:
            # m: 0,1 (m1), 2 (m2)
            # p: 0,1 (p1), 2,3 (p2)
            
            U_list = []
            V_list = []
            W_list = []
            
            # Sub 1: n=2, m=2, p=2 (m1, p1)
            U1, V1, W1 = get_local_decomp(2, 2, 2)
            # Map to global local indices
            # U rows: n*m = 2*3 = 6. Rows 0-3 correspond to m=0,1.
            # V rows: m*p = 3*4 = 12. Rows for m=0,1 and p=0,1.
            # W rows: p*n = 4*2 = 8. Rows for p=0,1 and n=0,1.
            
            # This mapping is complex. Easier to just use schoolbook for the non-2x2x2 blocks?
            # No, the gain from Strassen is significant.
            # Let's implement a generic mapper.
            
            pass # Fallback to schoolbook if mapping is too hard to code quickly
            return get_local_decomp(n_loc, m_loc, p_loc)

    # To ensure correctness and speed, I will use the pre-calculated block ranks 
    # and simply use schoolbook for the sub-blocks that are not 2x2x2, 
    # BUT I will manually construct the 2x2x2 blocks using Strassen.
    
    # Actually, implementing the index mapping for split blocks is the safest way to get the rank down.
    # Let's do the mapping for the 4 main blocks.
    
    # Global U, V, W
    U_global = []
    V_global = []
    W_global = []
    
    def add_block(n_off, m_off, p_off, n_sz, m_sz, p_sz, U_loc, V_loc, W_loc):
        rank_loc = len(U_loc[0])
        for r in range(rank_loc):
            u_col = [0] * (n_sz * m_sz)
            v_col = [0] * (m_sz * p_sz)
            w_col = [0] * (p_sz * n_sz)
            
            for i in range(n_sz):
                for j in range(m_sz):
                    val = U_loc[i*m_sz+j][r]
                    if val != 0:
                        u_col[i*m_sz+j] = val
                        
            for j in range(m_sz):
                for k in range(p_sz):
                    val = V_loc[j*p_sz+k][r]
                    if val != 0:
                        v_col[j*p_sz+k] = val
                        
            for k in range(p_sz):
                for i in range(n_sz):
                    val = W_loc[k*n_sz+i][r]
                    if val != 0:
                        w_col[k*n_sz+i] = val
                        
            # Extend global lists
            U_global.append(u_col)
            V_global.append(v_col)
            W_global.append(w_col)

    # We need to pad the local columns to the global size?
    # No, the interface requires U to be (n*m, rank).
    # So each column in U_global must have length n*m.
    
    U_global = []
    V_global = []
    W_global = []
    
    def add_block_global(n_off, m_off, p_off, n_sz, m_sz, p_sz, U_loc, V_loc, W_loc):
        rank_loc = len(U_loc[0])
        n, m, p = DIMENSIONS
        
        for r in range(rank_loc):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for i in range(n_sz):
                for j in range(m_sz):
                    val = U_loc[i*m_sz+j][r]
                    if val != 0:
                        global_i = n_off + i
                        global_j = m_off + j
                        u_col[global_i * m + global_j] = val
                        
            for j in range(m_sz):
                for k in range(p_sz):
                    val = V_loc[j*p_sz+k][r]
                    if val != 0:
                        global_j = m_off + j
                        global_k = p_off + k
                        v_col[global_j * p + global_k] = val
                        
            for k in range(p_sz):
                for i in range(n_sz):
                    val = W_loc[k*n_sz+i][r]
                    if val != 0:
                        global_k = p_off + k
                        global_i = n_off + i
                        w_col[global_k * n + global_i] = val
                        
            U_global.append(u_col)
            V_global.append(v_col)
            W_global.append(w_col)

    # Block 1: n=2, m=3, p=4. Offsets: n=0, m=0, p=0.
    # Split m: 2, 1. p: 2, 2.
    # Sub 1.1: n=2, m=2, p=2. Offsets: n=0, m=0, p=0.
    U11, V11, W11 = get_local_decomp(2, 2, 2)
    add_block_global(0, 0, 0, 2, 2, 2, U11, V11, W11)
    
    # Sub 1.2: n=2, m=1, p=2. Offsets: n=0, m=2, p=0.
    U12, V12, W12 = get_local_decomp(2, 1, 2)
    add_block_global(0, 2, 0, 2, 1, 2, U12, V12, W12)
    
    # Sub 1.3: n=2, m=2, p=2. Offsets: n=0, m=0, p=2.
    U13, V13, W13 = get_local_decomp(2, 2, 2)
    add_block_global(0, 0, 2, 2, 2, 2, U13, V13, W13)
    
    # Sub 1.4: n=2, m=1, p=2. Offsets: n=0, m=2, p=2.
    U14, V14, W14 = get_local_decomp(2, 1, 2)
    add_block_global(0, 2, 2, 2, 1, 2, U14, V14, W14)
    
    # Block 2: n=2, m=2, p=3. Offsets: n=0, m=3, p=0.
    # Wait, Block 2 is C[0:2, 4:7]? No.
    # C is 3x7.
    # Block 1 covered C[0:2, 0:4].
    # Block 2 covers C[0:2, 4:7].
    # A rows 0:2. B cols 4:7.
    # A cols for Block 2: m=3:5 (indices 3,4).
    # B rows for Block 2: m=3:5 (indices 3,4).
    # So n=2, m=2, p=3.
    # Split p: 2, 1.
    # Sub 2.1: n=2, m=2, p=2. Offsets: n=0, m=3, p=4.
    U21, V21, W21 = get_local_decomp(2, 2, 2)
    add_block_global(0, 3, 4, 2, 2, 2, U21, V21, W21)
    
    # Sub 2.2: n=2, m=2, p=1. Offsets: n=0, m=3, p=6.
    U22, V22, W22 = get_local_decomp(2, 2, 1)
    add_block_global(0, 3, 6, 2, 2, 1, U22, V22, W22)
    
    # Block 3: n=1, m=3, p=4. Offsets: n=2, m=0, p=0.
    # C[2, 0:4].
    # A row 2. B cols 0:4.
    # A cols: m=0:3.
    # B rows: m=0:3.
    # n=1, m=3, p=4.
    # Schoolbook is fine.
    U3, V3, W3 = get_local_decomp(1, 3, 4)
    add_block_global(2, 0, 0, 1, 3, 4, U3, V3, W3)
    
    # Block 4: n=1, m=2, p=3. Offsets: n=2, m=3, p=4.
    # C[2, 4:7].
    U4, V4, W4 = get_local_decomp(1, 2, 3)
    add_block_global(2, 3, 4, 1, 2, 3, U4, V4, W4)
    
    rank = len(U_global)
    
    return {"U": U_global, "V": V_global, "W": W_global}

# EVOLVE-BLOCK-END
