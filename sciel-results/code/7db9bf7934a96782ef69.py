"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll use a block decomposition strategy.
    # Split n=3 into (1, 2), m=5 into (2, 3), p=7 into (3, 4).
    # This gives 4 blocks:
    # 1. 1x2x3: Rank 2 (each row of A multiplies each column of B)
    # 2. 1x3x4: Rank 3 (each row of A multiplies each column of B)
    # 3. 2x2x3: Can be split into 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11
    # 4. 2x3x4: Can be split further or use mixed strategy.
    
    # Let's try to construct explicit factors.
    
    # Block 1: n=1, m=2, p=3
    # A is 1x2, B is 2x3, C is 1x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Rank is 2 (since n=1, each k column is determined by 2 inputs)
    # Actually, rank is m=2 for 1xm xp? No, rank is n*m*p generally, but for 1xm xp, 
    # C is 1xp, each element C[0,k] = sum_j A[0,j] B[j,k].
    # The tensor is T[i,j,k] = delta_i0 * 1.
    # We can represent this with rank = p? Or m?
    # Let's use the standard schoolbook for small blocks or find optimal.
    # For 1 x m x p, rank is m (since we can fix j and sum over k? No).
    # T[i,j,k] = 1 if i=0.
    # Sum_c U[i,j,c] V[j,k,c] W[k,0,c] = delta_{i,0}
    # If we set rank = m, let c correspond to j.
    # U[0,j,j'] = delta_{j,j'}, V[j,k,j'] = B_{j,k} (but B is variable in tensor, here tensor is 1s).
    # Wait, the tensor is the multiplication tensor, T_{ijk} = 1.
    # So we need sum_c U_{ij}^c V_{jk}^c W_{ki}^c = 1.
    
    # Let's just generate schoolbook factors for all blocks and sum ranks.
    # Schoolbook rank for n x m x p is n*m*p.
    
    # Block 1: 1x2x3 -> rank 6
    # Block 2: 1x3x4 -> rank 12
    # Block 3: 2x2x3 -> rank 12
    # Block 4: 2x3x4 -> rank 24
    # Total: 54. This is better than 105 but not great.
    
    # Let's use Strassen for 2x2x2 subblocks where possible.
    
    # Strategy:
    # n=3 = 1 + 2
    # m=5 = 2 + 3
    # p=7 = 3 + 4
    
    # Block (1,2,3): 1x2x3. Schoolbook rank 6.
    # Block (1,3,4): 1x3x4. Schoolbook rank 12.
    # Block (2,2,3): 2x2x3. Split p=3 into 2+1.
    #   Subblock (2,2,2): Strassen rank 7.
    #   Subblock (2,2,1): Schoolbook rank 4.
    #   Total 11.
    # Block (2,3,4): 2x3x4. 
    #   Split m=3 into 2+1, p=4 into 2+2.
    #   Subblocks:
    #   (2,2,2): Strassen 7.
    #   (2,2,2): Strassen 7.
    #   (2,1,2): Schoolbook 4.
    #   (2,1,2): Schoolbook 4.
    #   Total 22.
    
    # Total Rank = 6 + 12 + 11 + 22 = 51.
    
    # Let's see if we can do better for 1xm xp.
    # For 1 x m x p, the tensor is T_{0jk} = 1.
    # We need U_{0j}^c V_{jk}^c W_{k0}^c = 1.
    # If we choose rank = m, let c=j.
    # U_{0j}^j = 1, others 0.
    # V_{jk}^j = 1.
    # W_{k0}^j = 1.
    # Sum_c U_{0j}^c V_{jk}^c W_{k0}^c = sum_c delta_{j,c} * 1 * 1 = 1.
    # Yes! So 1 x m x p has rank m.
    
    # Revised Ranks:
    # Block 1: 1x2x3 -> rank 2.
    # Block 2: 1x3x4 -> rank 3.
    # Block 3: 2x2x3 -> 11 (7+4).
    # Block 4: 2x3x4 -> 22 (7+7+4+4).
    # Total = 2 + 3 + 11 + 22 = 38.
    
    # This is very close to the lower bound of 35.
    
    # Let's implement this.
    
    # We need to build U, V, W of sizes (n*m, rank), (m*p, rank), (p*n, rank).
    # n=3, m=5, p=7.
    # U: 15 x 38
    # V: 35 x 38
    # W: 21 x 38
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add factors from a sub-problem to the global matrices.
    # sub_U: local U matrix (n_sub * m_sub x rank_sub)
    # sub_V: local V matrix (m_sub * p_sub x rank_sub)
    # sub_W: local W matrix (p_sub * n_sub x rank_sub)
    # offsets: i_off, j_off, k_off
    # col_off: starting column in global matrices
    
    def add_factors(sub_U, sub_V, sub_W, i_off, j_off, k_off, col_off):
        n_sub = len(sub_U) // len(sub_V[0]) if sub_V else 0
        m_sub = len(sub_V) // len(sub_W[0]) if sub_W else 0
        p_sub = len(sub_W) // len(sub_U[0]) if sub_U else 0
        
        # Actually, let's just pass shapes or infer.
        # sub_U rows: (i_sub, j_sub) -> i_sub * m_sub + j_sub
        # sub_V rows: (j_sub, k_sub) -> j_sub * p_sub + k_sub
        # sub_W rows: (k_sub, i_sub) -> k_sub * n_sub + i_sub
        
        for i_sub in range(n_sub):
            for j_sub in range(m_sub):
                gi = i_off + i_sub
                gj = j_off + j_sub
                g_row_u = gi * m + gj
                
                for c in range(rank_sub):
                    val = sub_U[i_sub * m_sub + j_sub][c]
                    if val != 0:
                        U[g_row_u][col_off + c] = val
        
        for j_sub in range(m_sub):
            for k_sub in range(p_sub):
                gj = j_off + j_sub
                gk = k_off + k_sub
                g_row_v = gj * p + gk
                
                for c in range(rank_sub):
                    val = sub_V[j_sub * p_sub + k_sub][c]
                    if val != 0:
                        V[g_row_v][col_off + c] = val
        
        for k_sub in range(p_sub):
            for i_sub in range(n_sub):
                gk = k_off + k_sub
                gi = i_off + i_sub
                g_row_w = gk * n + gi
                
                for c in range(rank_sub):
                    val = sub_W[k_sub * n_sub + i_sub][c]
                    if val != 0:
                        W[g_row_w][col_off + c] = val

    # We need to define the factors for each block type.
    
    # 1. Block 1: n=1, m=2, p=3. Rank 2.
    # T_{0jk} = 1.
    # U_{0j}^c: U is 1x2 x 2. U[0, j, c] = delta_{j,c}.
    # V_{jk}^c: V is 2x3 x 2. V[j, k, c] = 1.
    # W_{k0}^c: W is 3x1 x 2. W[k, 0, c] = 1.
    # Check: Sum_c U[0,j,c] V[j,k,c] W[k,0,c] = Sum_c delta_{j,c} * 1 * 1 = 1. Correct.
    
    U_1 = [[1, 0],
           [0, 1]]
    V_1 = [[1, 1],
           [1, 1],
           [1, 1]]
    W_1 = [[1, 1],
           [1, 1],
           [1, 1]]
    # n_sub=1, m_sub=2, p_sub=3, rank_sub=2
    # i_off=0, j_off=0, k_off=0
    
    col_off = 0
    add_factors(U_1, V_1, W_1, 0, 0, 0, col_off)
    col_off += 2
    
    # 2. Block 2: n=1, m=3, p=4. Rank 3.
    # Similar to above.
    U_2 = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
    V_2 = [[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]
    W_2 = [[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]
    # n_sub=1, m_sub=3, p_sub=4, rank_sub=3
    # i_off=0, j_off=2, k_off=3
    
    add_factors(U_2, V_2, W_2, 0, 2, 3, col_off)
    col_off += 3
    
    # 3. Block 3: n=2, m=2, p=3. Split p into 2+1.
    # Subblock 3a: n=2, m=2, p=2. Strassen. Rank 7.
    # Strassen factors for 2x2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Tensor indices: i in {0,1}, j in {0,1}, k in {0,1}.
    # U rows: (i,j) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: (j,k) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: (k,i) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (a+d)(e+h)
    # a=U[0,0], d=U[3,1] (since U[i*m+j] -> U[1*2+1]=U[3] for i=1,j=1? No.
    # U index is i*m+j.
    # i=0, j=0 -> 0. a.
    # i=1, j=1 -> 3. d.
    # V index is j*p+k.
    # j=0, k=0 -> 0. e.
    # j=1, k=1 -> 3. h.
    # W index is k*n+i.
    # k=0, i=0 -> 0.
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    
    # C11 (i=0, k=0) is at W[0,0]=0? No, W[k*n+i].
    # C[i,k].
    # C[0,0] uses W[0*2+0]=W[0].
    # C[0,1] uses W[0*2+1]=W[1].
    # C[1,0] uses W[1*2+0]=W[2].
    # C[1,1] uses W[1*2+1]=W[3].
    
    # Let's construct U, V, W for Strassen 2x2x2.
    # U (4x7), V (4x7), W (4x7).
    
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    # U[0,0]=1, U[3,0]=1 (a and d)
    # V[0,0]=1, V[3,0]=1 (e and h)
    # W[0,0]=1, W[1,0]=-1, W[2,0]=-1, W[3,0]=1 ?
    # C11 = M1 + M4 - M5 + M7 -> W[0] gets +M1
    # C12 = M3 + M5 -> W[1] gets +M1? No.
    # C21 = M2 + M4 -> W[2] gets +M1? No.
    # C22 = M1 - M2 + M3 + M6 -> W[3] gets +M1
    
    # Let's fill W for M1 (col 0):
    W_s[0][0] = 1  # C11
    W_s[1][0] = 0  # C12
    W_s[2][0] = 0  # C21
    W_s[3][0] = 1  # C22
    
    # M2 = (c+d)e
    # U[2,1]=1, U[3,1]=1 (c and d)
    # V[0,1]=1 (e)
    # C21 = M2 + M4 -> W[2] gets +M2
    # C22 = M1 - M2 ... -> W[3] gets -M2
    U_s[2][1] = 1
    U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1
    W_s[3][1] = -1
    
    # M3 = a(f-h)
    # U[0,2]=1 (a)
    # V[1,2]=1, V[3,2]=-1 (f and -h)
    # C12 = M3 + M5 -> W[1] gets +M3
    # C22 = ... + M3 ... -> W[3] gets +M3
    U_s[0][2] = 1
    V_s[1][2] = 1
    V_s[3][2] = -1
    W_s[1][2] = 1
    W_s[3][2] = 1
    
    # M4 = d(g-e)
    # U[3,3]=1 (d)
    # V[2,3]=1, V[0,3]=-1 (g and -e)
    # C11 = ... + M4 ... -> W[0] gets +M4
    # C21 = ... + M4 ... -> W[2] gets +M4
    U_s[3][3] = 1
    V_s[2][3] = 1
    V_s[0][3] = -1
    W_s[0][3] = 1
    W_s[2][3] = 1
    
    # M5 = (a+b)h
    # U[0,4]=1, U[1,4]=1 (a and b)
    # V[3,4]=1 (h)
    # C11 = ... - M5 ... -> W[0] gets -M5
    # C12 = ... + M5 ... -> W[1] gets +M5
    U_s[0][4] = 1
    U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[0][4] = -1
    W_s[1][4] = 1
    
    # M6 = (c-a)(e+f)
    # U[2,5]=1, U[0,5]=-1 (c and -a)
    # V[0,5]=1, V[1,5]=1 (e and f)
    # C22 = ... + M6 ... -> W[3] gets +M6
    U_s[2][5] = 1
    U_s[0][5] = -1
    V_s[0][5] = 1
    V_s[1][5] = 1
    W_s[3][5] = 1
    
    # M7 = (b-d)(g+h)
    # U[1,6]=1, U[3,6]=-1 (b and -d)
    # V[2,6]=1, V[3,6]=1 (g and h)
    # C11 = ... + M7 ... -> W[0] gets +M7
    U_s[1][6] = 1
    U_s[3][6] = -1
    V_s[2][6] = 1
    V_s[3][6] = 1
    W_s[0][6] = 1
    
    # Subblock 3a: n=2, m=2, p=2.
    # i_off=1, j_off=0, k_off=0
    add_factors(U_s, V_s, W_s, 1, 0, 0, col_off)
    col_off += 7
    
    # Subblock 3b: n=2, m=2, p=1. Schoolbook. Rank 4.
    # T_{ijk} = 1.
    # Rank 4.
    # U (4x4), V (2x4), W (2x4).
    U_sb = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]
    V_sb = [[1,0,0,0],
            [1,1,1,1]] # Wait, V is 2x1 x 4? No.
    # V rows: (j,k). j in {0,1}, k=0.
    # V[0,0]=1, V[1,0]=1.
    # We need 4 columns.
    # Let's just use schoolbook factors.
    # c corresponds to (i,j,k) triple? No, rank is n*m*p = 4.
    # Let c index (i,j).
    # U[i,j, (i,j)] = 1.
    # V[j,0, (i,j)] = 1.
    # W[0,i, (i,j)] = 1.
    
    U_sb = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]
    V_sb = [[1,0,1,0], # j=0: c=0(i=0), c=2(i=1)
            [0,1,0,1]] # j=1: c=1(i=0), c=3(i=1)
    # Wait, V[j,k,c]. k=0.
    # V[0,0,c] should be 1 if i is such that (i,j) is active.
    # For c=0 (i=0,j=0): V[0,0,0]=1.
    # For c=1 (i=0,j=1): V[1,0,1]=1.
    # For c=2 (i=1,j=0): V[0,0,2]=1.
    # For c=3 (i=1,j=1): V[1,0,3]=1.
    
    V_sb = [[1,0,1,0],
            [0,1,0,1]]
    
    W_sb = [[1,0,0,0], # k=0, i=0: c=0(i=0,j=0), c=1(i=0,j=1)
            [0,0,1,1]] # k=0, i=1: c=2(i=1,j=0), c=3(i=1,j=1)
    # W[k,i,c].
    # W[0,0,c] = 1 if i=0. c=0,1.
    # W[0,1,c] = 1 if i=1. c=2,3.
    
    W_sb = [[1,1,0,0],
            [0,0,1,1]]
    
    # i_off=1, j_off=0, k_off=2
    add_factors(U_sb, V_sb, W_sb, 1, 0, 2, col_off)
    col_off += 4
    
    # 4. Block 4: n=2, m=3, p=4. Split m into 2+1, p into 2+2.
    # Subblock 4a: n=2, m=2, p=2. Strassen. Rank 7.
    # i_off=1, j_off=0, k_off=0 (relative to block 4 start)
    # Block 4 global offsets: i_off=1, j_off=2, k_off=3.
    # So 4a: i_off=1, j_off=2, k_off=3.
    add_factors(U_s, V_s, W_s, 1, 2, 3, col_off)
    col_off += 7
    
    # Subblock 4b: n=2, m=2, p=2. Strassen. Rank 7.
    # i_off=1, j_off=2, k_off=5.
    add_factors(U_s, V_s, W_s, 1, 2, 5, col_off)
    col_off += 7
    
    # Subblock 4c: n=2, m=1, p=2. Schoolbook. Rank 4.
    # i_off=1, j_off=4, k_off=3.
    # n=2, m=1, p=2.
    # U (2x4), V (2x4), W (4x4).
    # c index (i,k)? No, n*m*p = 4.
    # Let c index (i,j,k) but m=1 so j=0.
    # c index (i,k).
    U_4c = [[1,0,0,0], # i=0: c=0(k=0), c=1(k=1)
            [0,1,0,0],
            [0,0,1,0], # i=1: c=2(k=0), c=3(k=1)
            [0,0,0,1]]
    V_4c = [[1,1,0,0], # j=0, k=0: c=0(i=0), c=2(i=1)
            [0,0,1,1]] # j=0, k=1: c=1(i=0), c=3(i=1)
    # Wait, V[j,k,c].
    # V[0,0,c] = 1 if i is such that (i,0,0) is active.
    # c=0 (i=0,k=0): V[0,0,0]=1.
    # c=2 (i=1,k=0): V[0,0,2]=1.
    # V[0,1,c] = 1 if i is such that (i,0,1) is active.
    # c=1 (i=0,k=1): V[0,1,1]=1.
    # c=3 (i=1,k=1): V[0,1,3]=1.
    
    V_4c = [[1,0,1,0],
            [0,1,0,1]]
    
    W_4c = [[1,0,0,0], # k=0, i=0: c=0(i=0,k=0)
            [0,0,1,0], # k=0, i=1: c=2(i=1,k=0)
            [0,1,0,0], # k=1, i=0: c=1(i=0,k=1)
            [0,0,0,1]] # k=1, i=1: c=3(i=1,k=1)
    # W[k,i,c].
    # W[0,0,0]=1.
    # W[0,1,2]=1.
    # W[1,0,1]=1.
    # W[1,1,3]=1.
    
    W_4c = [[1,0,0,0],
            [0,0,1,0],
            [0,1,0,0],
            [0,0,0,1]]
    
    add_factors(U_4c, V_4c, W_4c, 1, 4, 3, col_off)
    col_off += 4
    
    # Subblock 4d: n=2, m=1, p=2. Schoolbook. Rank 4.
    # i_off=1, j_off=4, k_off=5.
    add_factors(U_4c, V_4c, W_4c, 1, 4, 5, col_off)
    col_off += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
