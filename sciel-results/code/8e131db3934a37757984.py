import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def get_strassen_2x2():
    # Returns U, V, W for 2x2x2 matrix multiplication
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a+d)(e+h)
    # a=A[0,0]=U[0], d=A[1,1]=U[3]
    # e=B[0,0]=V[0], h=B[1,1]=V[3]
    # C[0,0] gets M1, C[1,1] gets M1
    # W[0,0]=W[0], W[1,1]=W[3]
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    # c=A[1,0]=U[2], d=A[1,1]=U[3]
    # e=B[0,0]=V[0]
    # C[1,0] gets M2
    # W[0,1]=W[2]
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    # a=A[0,0]=U[0]
    # f=B[0,1]=V[1], h=B[1,1]=V[3]
    # C[0,1] gets M3
    # W[1,0]=W[1]
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = d(g-f)
    # d=A[1,1]=U[3]
    # g=B[1,0]=V[2], f=B[0,1]=V[1]
    # C[0,0] gets M4, C[1,0] gets M4
    # W[0,0]=W[0], W[2,1]=W[2]
    
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    # a=A[0,0]=U[0], b=A[0,1]=U[1]
    # h=B[1,1]=V[3]
    # C[0,0] gets -M5, C[0,1] gets M5
    # W[0,0]=W[0], W[1,0]=W[1]
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c+d)(g+h)
    # c=A[1,0]=U[2], d=A[1,1]=U[3]
    # g=B[1,0]=V[2], h=B[1,1]=V[3]
    # C[1,1] gets M6
    # W[3,1]=W[3]
    
    U[2, 5] = 1; U[3, 5] = 1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a-c)(e+g)
    # a=A[0,0]=U[0], c=A[1,0]=U[2]
    # e=B[0,0]=V[0], g=B[1,0]=V[2]
    # C[0,0] gets M7
    # W[0,0]=W[0]
    
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Block definitions
    # I: 0,1 (size 2), 2 (size 1)
    # J: 0,1,2,3 (size 4), 4 (size 1)
    # K: 0,1,2,3 (size 4), 4,5 (size 2)
    
    blocks = [
        # (n_block, m_block, p_block, i_start, j_start, k_start)
        (2, 4, 4, 0, 0, 0),
        (2, 4, 2, 0, 0, 4),
        (2, 1, 4, 0, 4, 0),
        (2, 1, 2, 0, 4, 4),
        (1, 4, 4, 2, 0, 0),
        (1, 4, 2, 2, 0, 4),
        (1, 1, 4, 2, 4, 0),
        (1, 1, 2, 2, 4, 4),
    ]
    
    # We will accumulate U, V, W
    # Total rank is sum of ranks of blocks
    total_rank = 0
    for b in blocks:
        nb, mb, pb = b[0], b[1], b[2]
        if nb == 2 and mb == 2 and pb == 2:
            total_rank += 7
        else:
            total_rank += nb * mb * pb
            
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    current_rank = 0
    
    for b in blocks:
        nb, mb, pb, i_start, j_start, k_start = b
        
        if nb == 2 and mb == 2 and pb == 2:
            # This case doesn't happen in our block list directly as 2x2x2
            # But we have 2x4x4 which is 4x 2x2x2
            # We handle 2x4x4 by splitting into 2x2x2 blocks
            pass
        
        # We need a helper to add a block
        # If the block is 2x2x2, use Strassen
        # Else, use Schoolbook
        
        # Actually, 2x4x4 is not 2x2x2.
        # We should decompose 2x4x4 into 4x 2x2x2.
        # And 2x4x2 into 2x 2x2x2.
        
        # Let's refine the block list to only contain 2x2x2 and schoolbook blocks.
        
    # Redefine blocks to be atomic
    atomic_blocks = []
    
    # I=2 (0,1), I=1 (2)
    # J=4 (0..3), J=1 (4)
    # K=4 (0..3), K=2 (4,5)
    
    # Split J=4 into J=2 (0,1) and J=2 (2,3)
    # Split K=4 into K=2 (0,1) and K=2 (2,3)
    
    # So we have:
    # I: 2, 1
    # J: 2, 2, 1
    # K: 2, 2, 2
    
    # Combinations:
    # I=2, J=2, K=2 -> 2x2x2 (Strassen)
    # I=2, J=2, K=2 -> 2x2x2 (Strassen)
    # I=2, J=1, K=2 -> 2x1x2 (Schoolbook)
    # I=2, J=1, K=2 -> 2x1x2 (Schoolbook)
    # I=1, J=2, K=2 -> 1x2x2 (Schoolbook)
    # I=1, J=2, K=2 -> 1x2x2 (Schoolbook)
    # I=1, J=1, K=2 -> 1x1x2 (Schoolbook)
    # I=1, J=1, K=2 -> 1x1x2 (Schoolbook)
    
    # Let's generate these
    i_splits = [(2, 0), (1, 2)]
    j_splits = [(2, 0), (2, 2), (1, 4)]
    k_splits = [(2, 0), (2, 2), (2, 4)]
    
    for nb, i_start in i_splits:
        for mb, j_start in j_splits:
            for pb, k_start in k_splits:
                atomic_blocks.append((nb, mb, pb, i_start, j_start, k_start))
                
    # Calculate total rank
    total_rank = 0
    for b in atomic_blocks:
        nb, mb, pb = b[0], b[1], b[2]
        if nb == 2 and mb == 2 and pb == 2:
            total_rank += 7
        else:
            total_rank += nb * mb * pb
            
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    current_rank = 0
    
    for b in atomic_blocks:
        nb, mb, pb, i_start, j_start, k_start = b
        
        if nb == 2 and mb == 2 and pb == 2:
            U_b, V_b, W_b = get_strassen_2x2()
        else:
            U_b, V_b, W_b = get_schoolbook(nb, mb, pb)
            
        # Map indices
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        for u_i in range(nb * mb):
            # u_i corresponds to local (li, lj)
            li = u_i // mb
            lj = u_i % mb
            global_i = i_start + li
            global_j = j_start + lj
            global_u_idx = global_i * m + global_j
            U[global_u_idx, current_rank : current_rank + nb*mb*pb if nb!=2 or mb!=2 or pb!=2 else 7] = U_b[:, :] # Wait, shapes don't match
            
    # This mapping is tricky because the local U_b has shape (nb*mb, rank_b)
    # and global U has shape (n*m, total_rank)
    
    # Let's do it properly
    current_rank = 0
    for b in atomic_blocks:
        nb, mb, pb, i_start, j_start, k_start = b
        
        if nb == 2 and mb == 2 and pb == 2:
            U_b, V_b, W_b = get_strassen_2x2()
            rank_b = 7
        else:
            U_b, V_b, W_b = get_schoolbook(nb, mb, pb)
            rank_b = nb * mb * pb
            
        # Copy U
        for u_i in range(nb * mb):
            li = u_i // mb
            lj = u_i % mb
            gi = i_start + li
            gj = j_start + lj
            g_u = gi * m + gj
            U[g_u, current_rank : current_rank + rank_b] = U_b[u_i, :]
            
        # Copy V
        for v_i in range(mb * pb):
            lj = v_i // pb
            lk = v_i % pb
            gj = j_start + lj
            gk = k_start + lk
            g_v = gj * p + gk
            V[g_v, current_rank : current_rank + rank_b] = V_b[v_i, :]
            
        # Copy W
        for w_i in range(pb * nb):
            lk = w_i // nb
            li = w_i % nb
            gk = k_start + lk
            gi = i_start + li
            g_w = gk * n + gi
            W[g_w, current_rank : current_rank + rank_b] = W_b[w_i, :]
            
        current_rank += rank_b
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
