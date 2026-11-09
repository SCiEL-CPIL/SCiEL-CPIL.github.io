import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 36
    
    # 1. Generate Schoolbook Solution (Rank 36)
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx][col] = 1
                V_sb[v_idx][col] = 1
                W_sb[w_idx][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    def verify_fast(U, V, W, R):
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        
        U_reshaped = U_arr.reshape(n, m, R)
        V_reshaped = V_arr.reshape(m, p, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        for _ in range(5):
            A = np.random.randint(-3, 4, size=(n, m), dtype=np.int32)
            B = np.random.randint(-3, 4, size=(m, p), dtype=np.int32)
            
            C_direct = A @ B
            LA = np.einsum('ij,ijr->ir', A, U_reshaped)
            LB = np.einsum('jk,jkr->kr', B, V_reshaped)
            C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
            
            if not np.array_equal(C_direct, C_decomp):
                return False
        return True

    # 2. Try to find a lower rank using block structure
    # Rank 20 solution: two rank-10 blocks
    # We need a rank-10 decomposition for 2x3x3.
    # We will search for a valid rank-20 decomposition of the full tensor.
    
    # Initialize with a random sparse matrix of rank 20
    target_rank = 20
    max_trials = 200
    for trial in range(max_trials):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random sparse U, V, W
        U = [[random.choice([-1, 0, 1]) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(p * n)]
        
        if verify_fast(U, V, W, target_rank):
            best_rank = target_rank
            best_solution = {"U": U, "V": V, "W": W}
            return best_solution

    return best_solution
# EVOLVE-BLOCK-END
