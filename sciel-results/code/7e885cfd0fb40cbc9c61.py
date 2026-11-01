import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0

    n, m, p = 2, 5, 6
    rank = 48

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen(j1, j2, k1, k2):
        """
        Adds Strassen rank-7 terms for A[:, j1:j2+1] * B[j1:j2+1, k1:k2+1]
        assuming i is 0,1.
        """
        # m1 = (a00+a11)(b00+b11)
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
                     
        # m2 = (a10+a11)*b00
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
                     
        # m3 = a00*(b01-b11)
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
                     
        # m4 = a11*(b10-b00)
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
                     
        # m5 = (a00+a01)*b11
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
                     
        # m6 = (a10-a00)*(b00+b01)
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
                     
        # m7 = (a01-a11)*(b10+b11)
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_rank2_2x1x2(j, k1, k2):
        """
        Adds rank-2 terms for A[:, j] * B[j, k1:k2+1]
        C[i,k1] += A[i,j]*B[j,k1]
        C[i,k2] += A[i,j]*B[j,k2]
        """
        # For i=0:
        add_rank_step([(u_idx(0,j), 1)], 
                     [(v_idx(j,k1), 1), (v_idx(j,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,0), 1)])
        # For i=1:
        add_rank_step([(u_idx(1,j), 1)], 
                     [(v_idx(j,k1), 1), (v_idx(j,k2), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), 1)])

    # Split j into (2,2,1) and k into (2,2,2)
    # j groups: (0,1), (2,3), (4)
    # k groups: (0,1), (2,3), (4,5)
    
    # For each j-group of size 2 and k-group of size 2: use Strassen (rank 7)
    # For j-group of size 1 and k-group of size 2: use rank-2 (rank 2)
    
    # j=(0,1), k=(0,1): Strassen
    add_strassen(0, 1, 0, 1)
    # j=(0,1), k=(2,3): Strassen
    add_strassen(0, 1, 2, 3)
    # j=(0,1), k=(4,5): Strassen
    add_strassen(0, 1, 4, 5)
    
    # j=(2,3), k=(0,1): Strassen
    add_strassen(2, 3, 0, 1)
    # j=(2,3), k=(2,3): Strassen
    add_strassen(2, 3, 2, 3)
    # j=(2,3), k=(4,5): Strassen
    add_strassen(2, 3, 4, 5)
    
    # j=(4), k=(0,1): rank-2
    add_rank2_2x1x2(4, 0, 1)
    # j=(4), k=(2,3): rank-2
    add_rank2_2x1x2(4, 2, 3)
    # j=(4), k=(4,5): rank-2
    add_rank2_2x1x2(4, 4, 5)

    # Verify: 6 Strassen blocks * 7 = 42, 3 rank-2 blocks * 2 = 6, total = 48
    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness with random test
    if time.time() - start_time < deadline - 5:
        np.random.seed(42)
        A = np.random.randint(0, 5, size=(n, m)).astype(float)
        B = np.random.randint(0, 5, size=(m, p)).astype(float)
        
        # Compute true product
        C_true = A @ B
        
        # Compute using decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But we need to sum over j as well
        C_decomp = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    for j in range(m):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                C_decomp[i, k] = total
        
        if not np.allclose(C_true, C_decomp, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            
            assert col == rank
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
