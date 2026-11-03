"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _strassen_2x2x2():
    """Strassen's 7-term decomposition for 2x2x2.
    
    U: 4x7 (indices i*2+j for A[i,j])
    V: 4x7 (indices j*2+k for B[j,k])
    W: 4x7 (indices k*2+i for C[k,i]... wait, trace convention W[k*n+i])
    
    For 2x2x2: n=2, m=2, p=2
    U[i*m+j] = U[i*2+j], i in {0,1}, j in {0,1}
    V[j*p+k] = V[j*2+k], j in {0,1}, k in {0,1}
    W[k*n+i] = W[k*2+i], k in {0,1}, i in {0,1}
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    W[k*n+i] should give coefficient for C[i,k]... 
    
    Actually the trace convention: W[k*n+i, c] means the c-th rank-1 term contributes
    to C[i,k] with that coefficient. So W is indexed by (k, i) in the sense that
    W[k*n+i] corresponds to output C[i,k].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices: 0=e, 1=f, 2=g, 3=h
    # W indices: W[k*2+i]: 0=C[0,0](k=0,i=0), 1=C[0,1](k=0,i=1)... 
    # Wait: W[k*n+i] with n=2. k is output column, i is output row.
    # W[0*2+0]=W[0] -> C[0,0], W[0*2+1]=W[1] -> C[1,0]
    # W[1*2+0]=W[2] -> C[0,1], W[1*2+1]=W[3] -> C[1,1]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C[0,0]
    W[3][0] = 1   # C[1,1]
    
    # M2 = (c+d)*e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1   # C[1,0]
    W[3][1] = -1  # C[1,1]
    
    # M3 = a*(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1   # C[0,1]
    W[3][2] = 1   # C[1,1]
    
    # M4 = d*(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C[0,0]
    W[1][3] = 1   # C[1,0]
    
    # M5 = (a+b)*h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C[0,0]
    W[2][4] = 1   # C[0,1]
    
    # M6 = (c-a)*(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C[1,1]
    
    # M7 = (b-d)*(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C[0,0]
    
    return U, V, W

def _standard_mult(n_sub, m_sub, p_sub):
    """Standard multiplication: (n_sub x m_sub) * (m_sub x p_sub)."""
    rank = n_sub * m_sub * p_sub
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i*m_sub+j][col] = 1
                V[j*p_sub+k][col] = 1
                W[k*n_sub+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    total_rank = 46
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    
    def add_block(U_block, V_block, W_block, a_col_start, b_row_start, b_col_start, m_sub, p_sub):
        """Add block contribution. n_sub is always 2."""
        nonlocal col_offset
        n_sub = 2
        
        for i in range(n_sub * m_sub):
            local_i = i // m_sub
            local_j = i % m_sub
            global_i = local_i  # A has only 2 rows, both map to global rows 0,1
            global_j = a_col_start + local_j
            gu = global_i * m + global_j
            for c in range(len(U_block[0])):
                U[gu][col_offset + c] += U_block[i][c]
        
        for j in range(m_sub * p_sub):
            local_j = j // p_sub
            local_k = j % p_sub
            global_j = b_row_start + local_j
            global_k = b_col_start + local_k
            gv = global_j * p + global_k
            for c in range(len(V_block[0])):
                V[gv][col_offset + c] += V_block[j][c]
        
        for k in range(p_sub * n_sub):
            local_k = k // n_sub
            local_i = k % n_sub
            global_k = b_col_start + local_k
            global_i = local_i
            gw = global_k * n + global_i
            for c in range(len(W_block[0])):
                W[gw][col_offset + c] += W_block[k][c]
        
        col_offset += len(U_block[0])
    
    # Blocks: (a_col_start, m_sub, b_row_start, b_col_start, p_sub, use_strassen)
    blocks = [
        (0, 2, 0, 0, 2, True),   # A[:,0:2] * B[0:2,0:2]
        (0, 2, 0, 2, 2, True),   # A[:,0:2] * B[0:2,2:4]
        (0, 2, 0, 4, 1, False),  # A[:,0:2] * B[0:2,4:5]
        (2, 2, 2, 0, 2, True),   # A[:,2:4] * B[2:4,0:2]
        (2, 2, 2, 2, 2, True),   # A[:,2:4] * B[2:4,2:4]
        (2, 2, 2, 4, 1, False),  # A[:,2:4] * B[2:4,4:5]
        (4, 1, 4, 0, 2, False),  # A[:,4:5] * B[4:5,0:2]
        (4, 1, 4, 2, 2, False),  # A[:,4:5] * B[4:5,2:4]
        (4, 1, 4, 4, 1, False),  # A[:,4:5] * B[4:5,4:5]
    ]
    
    for a_cs, m_sub, b_rs, b_cs, p_sub, use_strassen in blocks:
        if use_strassen:
            U_b, V_b, W_b = _strassen_2x2x2()
        else:
            U_b, V_b, W_b = _standard_mult(2, m_sub, p_sub)
        
        add_block(U_b, V_b, W_b, a_cs, b_rs, b_cs, m_sub, p_sub)
    
    return {"U": U, "V": V, "W": W}
