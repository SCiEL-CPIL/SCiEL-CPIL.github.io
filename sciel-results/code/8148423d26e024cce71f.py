import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def construct_strassen_2x2():
        """
        Construct a rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U, V, W of shape (4, 7) each.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: (a+d)(e+h)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1
        
        # Term 1: (c+d)e
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=1
        
        # Term 2: a(f-h)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[0,2]=1; W[1,2]=1
        
        # Term 3: c(d-b)
        U[2,3]=1; U[3,3]=1; U[1,3]=-1
        W[2,3]=1; W[3,3]=1
        
        # Term 4: (a+b)h
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=1; W[1,4]=1
        
        # Term 5: (c-a)(e+g)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[2,5]=1
        W[2,5]=1; W[3,5]=1
        
        # Term 6: (b-d)(f+h)
        U[1,6]=1; U[3,6]=-1
        V[1,6]=1; V[3,6]=1
        W[0,6]=1; W[1,6]=1
        
        return U, V, W

    def construct_laderman_3x3():
        """
        Construct a rank-15 decomposition for 3x3x3 matrix multiplication.
        Reference: Laderman, S. (1976).
        """
        U = np.zeros((9, 15))
        V = np.zeros((9, 15))
        W = np.zeros((9, 15))
        
        # Term 0: (a00+a11+a22)(b00+b11+b22)
        U[0,0]=1; U[4,0]=1; U[8,0]=1
        V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1; W[4,0]=-1; W[5,0]=1; W[6,0]=1; W[7,0]=1; W[8,0]=-1
        
        # Term 1: (a01+a12)(b00+b22)
        U[1,1]=1; U[5,1]=1
        V[0,1]=1; V[8,1]=1
        W[1,1]=1; W[2,1]=1; W[4,1]=1; W[5,1]=1
        
        # Term 2: (a02+a20)(b10+b22)
        U[2,2]=1; U[6,2]=1
        V[3,2]=1; V[8,2]=1
        W[3,2]=1; W[4,2]=1; W[6,2]=1; W[7,2]=1
        
        # Term 3: (a00+a22)(b00+b11)
        U[0,3]=1; U[8,3]=1
        V[0,3]=1; V[4,3]=1
        W[0,3]=1; W[1,3]=1; W[3,3]=1; W[4,3]=1
        
        # Term 4: (a01+a12+a22)(b10+b22)
        U[1,4]=1; U[5,4]=1; U[8,4]=1
        V[3,4]=1; V[8,4]=1
        W[3,4]=1; W[4,4]=1; W[5,4]=1; W[6,4]=1; W[7,4]=1; W[8,4]=1
        
        # Term 5: (a02+a20)(b00+b11)
        U[2,5]=1; U[6,5]=1
        V[0,5]=1; V[4,5]=1
        W[0,5]=1; W[1,5]=1; W[3,5]=1; W[4,5]=1
        
        # Term 6: (a00+a11)(b00+b11)
        U[0,6]=1; U[4,6]=1
        V[0,6]=1; V[4,6]=1
        W[0,6]=1; W[1,6]=1; W[3,6]=1; W[4,6]=1
        
        # Term 7: (a00+a11)(b00+b22)
        U[0,7]=1; U[4,7]=1
        V[0,7]=1; V[8,7]=1
        W[0,7]=1; W[1,7]=1; W[5,7]=1; W[6,7]=1
        
        # Term 8: (a01+a12)(b10+b22)
        U[1,8]=1; U[5,8]=1
        V[3,8]=1; V[8,8]=1
        W[3,8]=1; W[4,8]=1; W[6,8]=1; W[7,8]=1
        
        # Term 9: (a00+a22)(b00+b22)
        U[0,9]=1; U[8,9]=1
        V[0,9]=1; V[8,9]=1
        W[0,9]=1; W[1,9]=1; W[5,9]=1; W[6,9]=1
        
        # Term 10: (a02+a20)(b00+b11+b22)
        U[2,10]=1; U[6,10]=1
        V[0,10]=1; V[4,10]=1; V[8,10]=1
        W[0,10]=1; W[1,10]=1; W[3,10]=1; W[4,10]=1; W[5,10]=1; W[6,10]=1
        
        # Term 11: (a00+a11+a22)(b10+b22)
        U[0,11]=1; U[4,11]=1; U[8,11]=1
        V[3,11]=1; V[8,11]=1
        W[3,11]=1; W[4,11]=1; W[5,11]=1; W[6,11]=1; W[7,11]=1; W[8,11]=1
        
        # Term 12: (a01+a12+a22)(b00+b22)
        U[1,12]=1; U[5,12]=1; U[8,12]=1
        V[0,12]=1; V[8,12]=1
        W[0,12]=1; W[1,12]=1; W[5,12]=1; W[6,12]=1
        
        # Term 13: (a00+a11)(b10+b22)
        U[0,13]=1; U[4,13]=1
        V[3,13]=1; V[8,13]=1
        W[3,13]=1; W[4,13]=1; W[6,13]=1; W[7,13]=1
        
        # Term 14: (a02+a20)(b10+b22)
        U[2,14]=1; U[6,14]=1
        V[3,14]=1; V[8,14]=1
        W[3,14]=1; W[4,14]=1; W[6,14]=1; W[7,14]=1
        
        # Verify
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.dot(U[i*3+j], V[j*3+k] * W[k*3+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-8:
            return U, V, W
        else:
            return None

    def build_block_decomposition():
        """
        Builds a decomposition using Strassen for 2x2, Laderman for 3x3, and schoolbook for off-diagonals.
        """
        laderman_result = construct_laderman_3x3()
        use_laderman = laderman_result is not None
        laderman_rank = 15 if use_laderman else 27
        
        total_rank = 7 + 12 + 18 + 12 + 18 + laderman_rank
        U = np.zeros((25, total_rank))
        V = np.zeros((25, total_rank))
        W = np.zeros((25, total_rank))
        
        col_offset = 0
        
        # 1. Output block (0..1, 0..1): A(2x2) x B(2x2) -> Strassen rank 7
        U_a, V_a, W_a = construct_strassen_2x2()
        
        U[0, col_offset:col_offset+7] = U_a[0]
        U[1, col_offset:col_offset+7] = U_a[1]
        U[5, col_offset:col_offset+7] = U_a[2]
        U[6, col_offset:col_offset+7] = U_a[3]
        
        V[0, col_offset:col_offset+7] = V_a[0]
        V[1, col_offset:col_offset+7] = V_a[1]
        V[5, col_offset:col_offset+7] = V_a[2]
        V[6, col_offset:col_offset+7] = V_a[3]
        
        W[0, col_offset:col_offset+7] = W_a[0]
        W[1, col_offset:col_offset+7] = W_a[1]
        W[5, col_offset:col_offset+7] = W_a[2]
        W[6, col_offset:col_offset+7] = W_a[3]
        
        col_offset += 7
        
        # 2. Output block (0..1, 2..4): 
        #    Part A: A_top_left(2x2) x B_top_right(2x3) -> rank 12
        for i_prime in range(2):
            for j_prime in range(2):
                for k_prime in range(3):
                    t = col_offset + i_prime*6 + j_prime*3 + k_prime
                    u_idx = i_prime * 5 + j_prime
                    v_idx = j_prime * 5 + (2 + k_prime)
                    w_idx = (2 + k_prime) * 5 + i_prime
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        col_offset += 12
        
        #    Part B: A_top_right(2x3) x B_bottom_right(3x3) -> rank 18
        for i_prime in range(2):
            for j_prime in range(3):
                for k_prime in range(3):
                    t = col_offset + i_prime*9 + j_prime*3 + k_prime
                    u_idx = i_prime * 5 + (2 + j_prime)
                    v_idx = (2 + j_prime) * 5 + (2 + k_prime)
                    w_idx = (2 + k_prime) * 5 + i_prime
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        col_offset += 18
        
        # 3. Output block (2..4, 0..1):
        #    Part A: A_bottom_left(3x2) x B_top_left(2x2) -> rank 12
        for i_prime in range(3):
            for j_prime in range(2):
                for k_prime in range(2):
                    t = col_offset + i_prime*4 + j_prime*2 + k_prime
                    u_idx = (2 + i_prime) * 5 + j_prime
                    v_idx = j_prime * 5 + k_prime
                    w_idx = k_prime * 5 + (2 + i_prime)
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        col_offset += 12
        
        #    Part B: A_bottom_right(3x3) x B_bottom_left(3x2) -> rank 18
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(2):
                    t = col_offset + i_prime*6 + j_prime*2 + k_prime
                    u_idx = (2 + i_prime) * 5 + (2 + j_prime)
                    v_idx = (2 + j_prime) * 5 + k_prime
                    w_idx = k_prime * 5 + (2 + i_prime)
                    
                    U[u_idx, t] = 1
                    V[v_idx, t] = 1
                    W[w_idx, t] = 1
        col_offset += 18
        
        # 4. Output block (2..4, 2..4): A_bottom_right(3x3) x B_bottom_right(3x3)
        if use_laderman:
            U_l, V_l, W_l = laderman_result
            for t in range(15):
                for i_prime in range(3):
                    for j_prime in range(3):
                        u_local = i_prime * 3 + j_prime
                        u_global = (2 + i_prime) * 5 + (2 + j_prime)
                        U[u_global, col_offset + t] = U_l[u_local, t]
                
                for j_prime in range(3):
                    for k_prime in range(3):
                        v_local = j_prime * 3 + k_prime
                        v_global = (2 + j_prime) * 5 + (2 + k_prime)
                        V[v_global, col_offset + t] = V_l[v_local, t]
                
                for k_prime in range(3):
                    for i_prime in range(3):
                        w_local = k_prime * 3 + i_prime
                        w_global = (2 + k_prime) * 5 + (2 + i_prime)
                        W[w_global, col_offset + t] = W_l[w_local, t]
            
            col_offset += 15
        else:
            for i_prime in range(3):
                for j_prime in range(3):
                    for k_prime in range(3):
                        t = col_offset + i_prime*9 + j_prime*3 + k_prime
                        u_idx = (2 + i_prime) * 5 + (2 + j_prime)
                        v_idx = (2 + j_prime) * 5 + (2 + k_prime)
                        w_idx = (2 + k_prime) * 5 + (2 + i_prime)
                        
                        U[u_idx, t] = 1
                        V[v_idx, t] = 1
                        W[w_idx, t] = 1
            
            col_offset += 27
        
        return U, V, W, col_offset

    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    # Try block decomposition
    if time.time() < deadline - 10:
        U_block, V_block, W_block, block_rank = build_block_decomposition()
        max_err = verify(U_block, V_block, W_block, block_rank)
        if max_err < 1e-8:
            best_rank = block_rank
            best_result = (U_block, V_block, W_block)
    
    # Try alternating minimization for even lower ranks
    candidate_ranks = list(range(max(20, best_rank - 10), best_rank))
    
    def try_decomposition(rank, max_iters=150, seed=0):
        """Alternating minimization with structured initialization."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        if rank >= 25:
            U = rng.randn(n*m, rank) * (1.0 / np.sqrt(rank))
            V = rng.randn(m*p, rank) * (1.0 / np.sqrt(rank))
            W = rng.randn(p*n, rank) * (1.0 / np.sqrt(rank))
        else:
            U = rng.randn(n*m, rank) * 0.3
            V = rng.randn(m*p, rank) * 0.3
            W = rng.randn(p*n, rank) * 0.3
        
        prev_err = float('inf')
        best_local = None
        best_local_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < best_local_err:
                best_local_err = max_err
                best_local = (U.copy(), V.copy(), W.copy(), max_err)
            
            if max_err < 1e-8:
                return U, V, W, max_err
            
            if max_err > prev_err * 1.2 and it > 20:
                break
            prev_err = max_err
        
        if best_local is not None and best_local[3] < 1e-6:
            return best_local
        return None

    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        num_seeds = 8
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            result = try_decomposition(rank, max_iters=150, seed=seed + rank*1000)
            if result is not None:
                U, V, W, err = result
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                max_err = verify(U_r, V_r, W_r, rank)
                if max_err < 1e-8:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
                    break
        
        if best_rank < candidate_ranks[0]:
            break
    
    # Final polish
    if time.time() < deadline - 5:
        U, V, W = best_result
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            for i in range(n):
                for j in range(m):
                    B = np.zeros((best_rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = sol
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    A = np.zeros((best_rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = sol
                    except:
                        pass
            for k in range(p):
                for i in range(n):
                    C = np.zeros((best_rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            max_err = verify(U_r, V_r, W_r, best_rank)
            if max_err < 1e-10:
                best_result = (U_r, V_r, W_r)
                break

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
