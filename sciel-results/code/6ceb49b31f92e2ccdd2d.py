"""Matrix multiplication 2x3x5 with Strassen-based rank reduction."""
DIMENSIONS = [2, 3, 5]

import time

def _strassen_222():
    """Return U, V, W for M_{2,2,2} (Strassen, rank 7).
    U shape (4,7): indexed by (i*2+j, r)
    V shape (4,7): indexed by (j*2+k, r)
    W shape (4,7): indexed by (k*2+i, r)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 7 products for 2x2 matrices:
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)*b11
    # P3 = a11*(b12-b22)
    # P4 = a22*(b21-b11)
    # P5 = (a11+a12)*b22
    # P6 = (a21-a11)*(b11+b12)
    # P7 = (a12-a22)*(b21+b22)
    #
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P7
    
    # U factors (linear combos of A entries):
    # A is 2x2: A[i,j], i in {0,1}, j in {0,1}
    # U[i*m+j, r] = coefficient for A[i,j] in product r
    # 
    # r=0 (P1): a11+a22  -> U[0,0]=1, U[3,0]=1 (a11 is i=0,j=0; a22 is i=1,j=1)
    # r=1 (P2): a21+a22 -> U[2,1]=1, U[3,1]=1
    # r=2 (P3): a11     -> U[0,2]=1
    # r=3 (P4): a22     -> U[3,3]=1
    # r=4 (P5): a11+a12 -> U[0,4]=1, U[1,4]=1
    # r=5 (P6): a21-a11 -> U[2,5]=1, U[0,5]=-1
    # r=6 (P7): a12-a22 -> U[1,6]=1, U[3,6]=-1
    
    U[0][0] = 1   # a11 in P1
    U[3][0] = 1   # a22 in P1
    U[2][1] = 1   # a21 in P2
    U[3][1] = 1   # a22 in P2
    U[0][2] = 1   # a11 in P3
    U[3][3] = 1   # a22 in P4
    U[0][4] = 1   # a11 in P5
    U[1][4] = 1   # a12 in P5
    U[2][5] = 1   # a21 in P6
    U[0][5] = -1  # -a11 in P6
    U[1][6] = 1   # a12 in P7
    U[3][6] = -1  # -a22 in P7
    
    # V factors (linear combos of B entries):
    # B is 2x2: B[j,k], j in {0,1}, k in {0,1}
    # V[j*p+k, r] = coefficient for B[j,k] in product r
    #
    # r=0 (P1): b11+b22 -> V[0,0]=1, V[3,0]=1
    # r=1 (P2): b11     -> V[0,1]=1
    # r=2 (P3): b12-b22 -> V[1,2]=1, V[3,2]=-1
    # r=3 (P4): b21-b11 -> V[2,3]=1, V[0,3]=-1
    # r=4 (P5): b22     -> V[3,4]=1
    # r=5 (P6): b11+b12 -> V[0,5]=1, V[1,5]=1
    # r=6 (P7): b21+b22 -> V[2,6]=1, V[3,6]=1
    
    V[0][0] = 1   # b11 in P1
    V[3][0] = 1   # b22 in P1
    V[0][1] = 1   # b11 in P2
    V[1][2] = 1   # b12 in P3
    V[3][2] = -1  # -b22 in P3
    V[2][3] = 1   # b21 in P4
    V[0][3] = -1  # -b11 in P4
    V[3][4] = 1   # b22 in P5
    V[0][5] = 1   # b11 in P6
    V[1][5] = 1   # b12 in P6
    V[2][6] = 1   # b21 in P7
    V[3][6] = 1   # b22 in P7
    
    # W factors (linear combos for C entries):
    # C[i,k], i in {0,1}, k in {0,1}
    # W[k*n+i, r] = coefficient of product r in C[i,k]
    #
    # C[0,0] = P1 + P4 - P5 - P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P7
    
    # W[k*2+i, r]:
    # C[0,0]: k=0, i=0 -> W[0, r]
    W[0][0] = 1   # P1
    W[0][3] = 1   # P4
    W[0][4] = -1  # -P5
    W[0][6] = -1  # -P7
    
    # C[0,1]: k=1, i=0 -> W[1, r]
    W[1][2] = 1   # P3
    W[1][4] = 1   # P5
    
    # C[1,0]: k=0, i=1 -> W[2, r]
    W[2][1] = 1   # P2
    W[2][3] = 1   # P4
    
    # C[1,1]: k=1, i=1 -> W[3, r]
    W[3][0] = 1   # P1
    W[3][1] = -1  # -P2
    W[3][2] = 1   # P3
    W[3][6] = 1   # P7
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook M_{n,m,p} with rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _add_block(global_U, global_V, global_W, rank_offset, 
               sub_U, sub_V, sub_W,
               i_off, j_off, k_off,
               n_sub, m_sub, p_sub,
               n_global, m_global, p_global):
    """Add a sub-block decomposition into the global factor matrices.
    sub_U: (n_sub*m_sub, r), sub_V: (m_sub*p_sub, r), sub_W: (p_sub*n_sub, r)
    Maps local indices to global indices with offsets.
    """
    r = len(sub_U[0])
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            gi = i_off + local_i
            gj = j_off + local_j
            global_row_U = gi * m_global + gj
            for rr in range(r):
                if sub_U[local_i*m_sub + local_j][rr] != 0:
                    global_U[global_row_U][rank_offset + rr] += sub_U[local_i*m_sub + local_j][rr]
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            gj = j_off + local_j
            gk = k_off + local_k
            global_row_V = gj * p_global + gk
            for rr in range(r):
                if sub_V[local_j*p_sub + local_k][rr] != 0:
                    global_V[global_row_V][rank_offset + rr] += sub_V[local_j*p_sub + local_k][rr]
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            gk = k_off + local_k
            gi = i_off + local_i
            global_row_W = gk * n_global + gi
            for rr in range(r):
                if sub_W[local_k*n_sub + local_i][rr] != 0:
                    global_W[global_row_W][rank_offset + rr] += sub_W[local_k*n_sub + local_i][rr]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct by checking tensor equality."""
    import numpy as np
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check on a few random matrices
    import random
    random.seed(42)
    for trial in range(3):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B directly
        C_direct = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_direct[i][k] = s
        
        # Compute C from decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) ... 
        # No wait. The standard form is:
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j too?
        # Actually the tensor decomposition of matrix mult is:
        # T[i,j,k] = delta[i,j,k] for the multiplication tensor
        # The decomposition is: for each r, the rank-1 tensor U_r * V_r * W_r
        # where U_r is a vector over (i,j), V_r over (j,k), W_r over (k,i)
        # And C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k' ... )
        
        # Let me just verify the tensor directly.
        # The multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k) in [0,n)x[0,m)x[0,p)
        # The decomposition satisfies: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1
        # Wait, that's not right either. Let me think about this more carefully.
        
        # The matrix multiplication tensor M has entries M[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
        # A rank-r decomposition gives: M[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
        # But actually for matrix mult, the tensor is defined so that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # This can be written as C[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta(i,j,k)
        # So the tensor T[i,j,k] = 1 for all (i,j,k).
        # And the decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
        
        # Let's verify this:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Total rank: use Strassen for i in [0,2), j in [0,2), k in [0,2) block (rank 7)
    # Remaining blocks:
    # i in [0,2), j in [2,3), k in [0,5): 2*1*5 = 10
    # i in [0,2), j in [0,2), k in [2,5): 2*2*3 = 12
    # Total rank = 7 + 10 + 12 = 29
    
    total_rank = 29
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Block 1: Strassen for i in [0,2), j in [0,2), k in [0,2)
    sub_U, sub_V, sub_W = _strassen_222()
    _add_block(U, V, W, 0, sub_U, sub_V, sub_W,
               i_off=0, j_off=0, k_off=0,
               n_sub=2, m_sub=2, p_sub=2,
               n_global=n, m_global=m, p_global=p)
    
    # Block 2: Schoolbook for i in [0,2), j in [2,3), k in [0,5)
    # This is M_{2,1,5}
    sb_U, sb_V, sb_W = _schoolbook(2, 1, 5)
    _add_block(U, V, W, 7, sb_U, sb_V, sb_W,
               i_off=0, j_off=2, k_off=0,
               n_sub=2, m_sub=1, p_sub=5,
               n_global=n, m_global=m, p_global=p)
    
    # Block 3: Schoolbook for i in [0,2), j in [0,2), k in [2,5)
    # This is M_{2,2,3}
    sb2_U, sb2_V, sb2_W = _schoolbook(2, 2, 3)
    _add_block(U, V, W, 17, sb2_U, sb2_V, sb2_W,
               i_off=0, j_off=0, k_off=2,
               n_sub=2, m_sub=2, p_sub=3,
               n_global=n, m_global=m, p_global=p)
    
    # Verify correctness
    if time.time() < deadline - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = [[0]*rank_fb for _ in range(n*m)]
            V_fb = [[0]*rank_fb for _ in range(m*p)]
            W_fb = [[0]*rank_fb for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col] = 1
                        V_fb[j*p+k][col] = 1
                        W_fb[k*n+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
