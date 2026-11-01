import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook Rank
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Strategy: Construct a rank-54 decomposition using Strassen for 2x2x2 blocks
    # and standard multiplication for 2x1x1 blocks.
    
    # Total Rank:
    # 3 Strassen blocks (2x2x2) for A[:, 0:2] * B[0:2, :] -> 3 * 7 = 21
    # 6 Standard blocks (2x1x1) for A[:, 2:3] * B[2:3, :] -> 6 * 2 = 12
    # 3 Strassen blocks (2x2x2) for A[:, 3:5] * B[3:5, :] -> 3 * 7 = 21
    # Total = 54
    
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # Local indices 0,1 map to global:
        # i: i_off, i_off+1
        # j: j_off, j_off+1
        # k: k_off, k_off+1
        
        # r=0: M1
        # U: (0,0), (1,1)
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        # V: (0,0), (1,1)
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        # W: (0,0), (1,1) -> C[0,0], C[1,1]
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        
        # r=1: M2
        # U: (1,0), (1,1)
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        # V: (0,0)
        V_new[(j_off)*p + (k_off), current_rank] = 1
        # W: C[1,0] coeff 1, C[1,1] coeff -1
        # W[k,i]. C[1,0] is i=1, k=0 -> W[0,1]. C[1,1] is i=1, k=1 -> W[1,1].
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        
        # r=2: M3
        # U: (0,0)
        U_new[(i_off)*m + (j_off), current_rank] = 1
        # V: (0,1), (1,1)
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        # W: C[0,1] coeff 1, C[1,1] coeff 1
        # C[0,1] is i=0, k=1 -> W[1,0]. C[1,1] is i=1, k=1 -> W[1,1].
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        
        # r=3: M4
        # U: (1,1)
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        # V: (1,0), (0,0)
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        # W: C[0,0] coeff 1, C[1,0] coeff 1
        # C[0,0] is i=0, k=0 -> W[0,0]. C[1,0] is i=1, k=0 -> W[0,1].
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        
        # r=4: M5
        # U: (0,0), (0,1)
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        # V: (1,1)
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        # W: C[0,0] coeff -1, C[0,1] coeff 1
        # C[0,0] is i=0, k=0 -> W[0,0]. C[0,1] is i=0, k=1 -> W[1,0].
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        
        # r=5: M6
        # U: (1,0), (0,0)
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        # V: (0,0), (0,1)
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        # W: C[1,1] coeff 1
        # C[1,1] is i=1, k=1 -> W[1,1].
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        
        # r=6: M7
        # U: (0,1), (1,1)
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        # V: (1,0), (1,1)
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        # W: C[0,0] coeff 1
        # C[0,0] is i=0, k=0 -> W[0,0].
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        # 2x1x1 block
        # C[i, k] = A[i, j] * B[j, k]
        # r=0: for i=0
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=1: for i=1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    # j_off = 0
    # k splits: 0, 2, 4
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    # j_off = 2
    # k splits: 0, 1, 2, 3, 4, 5
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    # j_off = 3
    # k splits: 0, 2, 4
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Verify
    # U shape (10, 54), V shape (30, 54), W shape (12, 54)
    
    # Check if exact
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Our decomposition: C[i, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    # W is (p*n, rank). Index k*n+i.
    
    # We can verify by checking if the tensor matches.
    # Since it's constructed algebraically, it should be exact.
    # But let's do a quick sanity check with random matrices if time permits.
    if time.time() - start_time < DEADLINE - 5:
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_new):
                        u_val = U_new[i*m + j, r]
                        v_val = V_new[j*p + k, r]
                        w_val = W_new[k*n + i, r]
                        val += u_val * v_val * w_val
                C_calc[i, k] = val
                
        if np.allclose(C_test, C_calc):
            best_rank = rank_new
            best_U = U_new
            best_V = V_new
            best_W = W_new
        else:
            # Fallback to schoolbook if verification fails (should not happen)
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
