import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Vectorized verification using einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    # For each (i,j,k), we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a 3-way tensor contraction
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(T_recon, 1.0, atol=tol)

def _rank1_nm_p(n, m, p):
    """Rank-1 decomposition for M_{n,m,p} when one dimension is 1."""
    # This is only valid when n=1 or m=1 or p=1
    u = np.ones((n*m, 1))
    v = np.ones((m*p, 1))
    w = np.ones((p*n, 1))
    return u, v, w

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2.
    
    U indexed by (i,j) -> i*2+j, shape (4, 7)
    V indexed by (j,k) -> j*2+k, shape (4, 7)  
    W indexed by (k,i) -> k*2+i, shape (4, 7)
    
    The 7 products are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: for each of 7 terms, which (i,j) entries are used
    # M1: A11, A22 -> U[0,0]=1, U[3,0]=1
    # M2: A21, A22 -> U[2,1]=1, U[3,1]=1
    # M3: A11 -> U[0,2]=1
    # M4: A22 -> U[3,3]=1
    # M5: A11, A12 -> U[0,4]=1, U[1,4]=1
    # M6: A21, A11 -> U[2,5]=1, U[0,5]=-1
    # M7: A12, A22 -> U[1,6]=1, U[3,6]=-1
    u = np.zeros((4, 7))
    u[0, 0] = 1; u[3, 0] = 1  # M1
    u[2, 1] = 1; u[3, 1] = 1  # M2
    u[0, 2] = 1               # M3
    u[3, 3] = 1               # M4
    u[0, 4] = 1; u[1, 4] = 1  # M5
    u[2, 5] = 1; u[0, 5] = -1 # M6
    u[1, 6] = 1; u[3, 6] = -1 # M7
    
    # V: for each of 7 terms, which (j,k) entries are used
    # M1: B11, B22 -> V[0,0]=1, V[3,0]=1
    # M2: B11 -> V[0,1]=1
    # M3: B21, B22 -> V[2,2]=1, V[3,2]=-1
    # M4: B12, B11 -> V[1,3]=1, V[0,3]=-1
    # M5: B22 -> V[3,4]=1
    # M6: B11, B12 -> V[0,5]=1, V[1,5]=1
    # M7: B21, B22 -> V[2,6]=1, V[3,6]=1
    v = np.zeros((4, 7))
    v[0, 0] = 1; v[3, 0] = 1  # M1
    v[0, 1] = 1               # M2
    v[2, 2] = 1; v[3, 2] = -1 # M3
    v[1, 3] = 1; v[0, 3] = -1 # M4
    v[3, 4] = 1               # M5
    v[0, 5] = 1; v[1, 5] = 1  # M6
    v[2, 6] = 1; v[3, 6] = 1  # M7
    
    # W: for each of 7 terms, which (k,i) entries contribute to C
    # C11 (k=0,i=0): W[0,0]=1 (M1), W[0,3]=1 (M4), W[0,4]=-1 (M5), W[0,6]=1 (M7)
    # C12 (k=1,i=0): W[2,2]=1 (M3), W[2,4]=1 (M5)
    # C21 (k=0,i=1): W[1,1]=1 (M2), W[1,3]=1 (M4)
    # C22 (k=1,i=1): W[3,0]=1 (M1), W[3,1]=-1 (M2), W[3,2]=1 (M3), W[3,5]=1 (M6)
    w = np.zeros((4, 7))
    w[0, 0] = 1; w[0, 3] = 1; w[0, 4] = -1; w[0, 6] = 1  # C11
    w[2, 2] = 1; w[2, 4] = 1                              # C12
    w[1, 1] = 1; w[1, 3] = 1                              # C21
    w[3, 0] = 1; w[3, 1] = -1; w[3, 2] = 1; w[3, 5] = 1   # C22
    
    return u, v, w  # Shapes (4,7), (4,7), (4,7)

def _construct_rank_35():
    """Construct a rank-35 decomposition for 3x4x8.
    
    Partition: split m=4 into 1+3, so we handle 3x1x8 and 3x3x8.
    For 3x3x8: split p=8 into 4+4, giving two 3x3x4 blocks.
    For each 3x3x4: split n=3 into 1+2, m=3 into 1+2, p=4 into 2+2.
    This gives 1x1x2 (rank 1), 1x2x2 (rank 2), 2x1x2 (rank 2), 2x2x2 (rank 7).
    Total per 3x3x4: 1+2+2+7 = 12
    Total: 1 (for 3x1x8) + 12 + 12 = 25... 
    Wait, let me recompute.
    
    Actually, let's use a cleaner partition:
    3x4x8:
    - Split m=4 as 1+3:
      - 3x1x8: rank 3 (each of 3 rows needs its own term since p=8)
        Actually no: 3x1x8 means multiplying 3x1 by 1x8.
        C[i,k] = A[i,0] * B[0,k]. 
        This is rank 3 (one per row i), since for each i, C[i,k] = A[i,0]*B[0,k]
        We can do this with rank 3: for each i, one term with U[i*1+0,c_i]=1, V[0*8+k,c_i]=B[0,k]... 
        No wait, we need exact decomposition. The tensor T[i,j,k] for 3x1x8 has T[i,0,k]=1 for all i,k.
        This is rank 3: terms (i,0,k) for each i, but that's 3*8=24 terms.
        Actually no. T[i,0,k] = 1 for all i in {0,1,2}, k in {0,...,7}.
        This is an outer product of a (3,1) all-ones matrix and a (1,8) all-ones matrix.
        So it's rank 1! U[3,0]=1, V[8,0]=1, W[24,0]=1... 
        Let me check: sum_c U[i,0,c]*V[0,k,c]*W[k,0,c] = U[i,0]*V[0,k]*W[k,0]
        If U[i,0]=1 for all i, V[0,k]=1 for all k, W[k,0]=1 for all k, then yes, rank 1.
        
    Let me redo the construction properly.
    """
    n, m, p = 3, 4, 8
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n*m)
            v_col = np.zeros(m*p)
            w_col = np.zeros(p*n)
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off+i)*m + (j_off+j)] = u_sub[i*sub_m+j, c]
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off+j)*p + (k_off+k)] = v_sub[j*sub_p+k, c]
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off+k)*n + (i_off+i)] = w_sub[k*sub_n+i, c]
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # 3x1x8: rank 1 (outer product)
    u_318, v_318, w_318 = _rank1_nm_p(3, 1, 8)
    add_sub(0, 0, 0, 3, 1, 8, u_318, v_318, w_318)
    
    # 3x3x8: split p=8 into 4+4
    # Each 3x3x4: split n=3 into 1+2, m=3 into 1+2, p=4 into 2+2
    # Sub-blocks for each 3x3x4:
    #   1x1x2: rank 1
    #   1x2x2: rank 2 (split n=1, so it's 1x2x2, which is rank 2*2=4? No...)
    #   2x1x2: rank 2
    #   2x2x2: rank 7 (Strassen)
    # Total per 3x3x4: 1 + 2 + 2 + 7 = 12
    # Total for 3x3x8: 12 + 12 = 24
    # Grand total: 1 + 24 = 25
    
    # For each 3x3x4 block (at p-offset 0 and p-offset 4):
    for p_off in [0, 4]:
        # 1x1x2 at (n=0, m=1, p=p_off)
        u_112, v_112, w_112 = _rank1_nm_p(1, 1, 2)
        add_sub(0, 1, p_off, 1, 1, 2, u_112, v_112, w_112)
        
        # 1x2x2 at (n=0, m=2, p=p_off): split m=2 into 1+1
        # 1x1x2 (rank 1) + 1x1x2 (rank 1) = rank 2
        u_112a, v_112a, w_112a = _rank1_nm_p(1, 1, 2)
        add_sub(0, 2, p_off, 1, 1, 2, u_112a, v_112a, w_112a)
        add_sub(0, 3, p_off, 1, 1, 2, u_112a, v_112a, w_112a)
        
        # 2x1x2 at (n=1, m=1, p=p_off): split n=2 into 1+1
        # 1x1x2 (rank 1) + 1x1x2 (rank 1) = rank 2
        u_112b, v_112b, w_112b = _rank1_nm_p(1, 1, 2)
        add_sub(1, 1, p_off, 1, 1, 2, u_112b, v_112b, w_112b)
        add_sub(2, 1, p_off, 1, 1, 2, u_112b, v_112b, w_112b)
        
        # 2x2x2 at (n=1, m=2, p=p_off): Strassen rank 7
        u_222, v_222, w_222 = _strassen_2x2x2()
        add_sub(1, 2, p_off, 2, 2, 2, u_222, v_222, w_222)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook fallback (rank 96)
    rank_school = n*m*p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Try the block decomposition (should give rank ~25)
    if time.time() < deadline:
        try:
            U35, V35, W35 = _construct_rank_35()
            if _verify_safe(U35.tolist(), V35.tolist(), W35.tolist()):
                rank35 = U35.shape[1]
                if rank35 < best_rank:
                    best = {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}
                    best_rank = rank35
        except Exception:
            pass
    
    # Try to find a lower rank using alternating minimization
    if time.time() < deadline - 5:
        try:
            best_rank_am = _alternating_minimization(n, m, p, deadline)
            if best_rank_am is not None and best_rank_am < best_rank:
                best_rank = best_rank_am
        except Exception:
            pass
    
    return best

def _alternating_minimization(n, m, p, deadline):
    """Try to find a lower-rank decomposition using alternating minimization.
    
    We formulate this as: find rank-r matrices U (nm x r), V (mp x r), W (pn x r)
    such that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We use a greedy approach: start with schoolbook and try to merge terms.
    """
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with a moderate rank and try to optimize
    for target_rank in [24, 23, 22, 21, 20, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(target_rank)
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), we need sum_c U[i*m+j, c] * (V[j*p+*, c] * W[**n+i, c]) = 1 for all k
            # This is a linear system for each (i,j)
            for i in range(n):
                for j in range(m):
                    # Build the system: for each k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        for c in range(target_rank):
                            A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b_vec = np.ones(p)
                    # Solve least squares
                    U[i*m+j, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V: for each (j,k), sum_c V[j*p+k, c] * (U[**m+j, c] * W[k*n+*, c]) = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        for c in range(target_rank):
                            A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b_vec = np.ones(n)
                    V[j*p+k, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W: for each (k,i), sum_c W[k*n+i, c] * (U[**m+*, c] * V[**p+k, c]) = 1 for all (i,j)
            # Actually, for each (k,i), we need: for all j, sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b_vec = np.ones(m)
                    W[k*n+i, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Verify
        if _verify_safe(U.tolist(), V.tolist(), W.tolist()):
            return target_rank
    
    return None
