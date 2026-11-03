import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs factor matrices for matrix multiplication 2x3 * 3x5.
    Attempts rank-27 via structured search, falls back to rank-28.
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 145
    
    # First, get the known rank-28
    result28 = _construct_rank28(n, m, p)
    
    # Try to find rank-27
    if time.time() < deadline - 20:
        result27 = _try_rank27_structured(n, m, p, deadline)
        if result27 is not None:
            return result27
    
    return result28

def _construct_rank28(n, m, p):
    """Known rank-28 construction."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(rank_offset, j_offset, k_offset):
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        U[a_idx][r0] += 1; U[d_idx][r0] += 1
        U[c_idx][r1] += 1; U[d_idx][r1] += 1
        U[a_idx][r2] += 1
        U[d_idx][r3] += 1
        U[a_idx][r4] += 1; U[b_idx][r4] += 1
        U[c_idx][r5] += 1; U[a_idx][r5] -= 1
        U[b_idx][r6] += 1; U[d_idx][r6] -= 1
        
        V[e_idx][r0] += 1; V[h_idx][r0] += 1
        V[e_idx][r1] += 1
        V[f_idx][r2] += 1; V[h_idx][r2] -= 1
        V[g_idx][r3] += 1; V[e_idx][r3] -= 1
        V[h_idx][r4] += 1
        V[e_idx][r5] += 1; V[f_idx][r5] += 1
        V[g_idx][r6] += 1; V[h_idx][r6] += 1
        
        W[k_offset * n + 0][r0] += 1
        W[k_offset * n + 0][r3] += 1
        W[k_offset * n + 0][r4] -= 1
        W[k_offset * n + 0][r6] += 1
        
        W[(k_offset + 1) * n + 0][r2] += 1
        W[(k_offset + 1) * n + 0][r4] += 1
        
        W[k_offset * n + 1][r1] += 1
        W[k_offset * n + 1][r3] += 1
        
        W[(k_offset + 1) * n + 1][r0] += 1
        W[(k_offset + 1) * n + 1][r1] -= 1
        W[(k_offset + 1) * n + 1][r2] += 1
        W[(k_offset + 1) * n + 1][r5] += 1
    
    add_strassen_block(0, 0, 0)
    add_strassen_block(7, 0, 2)
    
    # Block 3: j=0,1, k=4 (schoolbook 2x2*2x1)
    r0, r1, r2, r3 = 14, 15, 16, 17
    U[0][r0] += 1; U[1][r1] += 1
    U[3][r2] += 1; U[4][r3] += 1
    V[4][r0] += 1; V[9][r1] += 1
    V[4][r2] += 1; V[9][r3] += 1
    W[8][r0] += 1; W[8][r1] += 1
    W[9][r2] += 1; W[9][r3] += 1
    
    # Block 4: j=2, k=0..4 (schoolbook 2x1*1x5)
    rank_offset = 18
    for k in range(5):
        r_u0 = rank_offset
        r_u1 = rank_offset + 1
        rank_offset += 2
        U[2][r_u0] += 1
        U[5][r_u1] += 1
        V[10 + k][r_u0] += 1
        V[10 + k][r_u1] += 1
        W[k * n][r_u0] += 1
        W[k * n + 1][r_u1] += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_trials=50):
    """Verify using random sampling."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    for _ in range(num_trials):
        A = np.random.randint(-2, 3, size=(n, m)).astype(np.float64)
        B = np.random.randint(-2, 3, size=(m, p)).astype(np.float64)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=np.float64)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    val = sum(U_arr[i*m + j, r] * V_arr[j*p + k, r] for j in range(m))
                    total += val * W_arr[k*n + i, r]
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _try_rank27_structured(n, m, p, deadline):
    """
    Try structured modifications to reduce rank from 28 to 27.
    Focus on the schoolbook blocks where there's more flexibility.
    """
    base = _construct_rank28(n, m, p)
    U_base = [row[:] for row in base['U']]
    V_base = [row[:] for row in base['V']]
    W_base = [row[:] for row in base['W']]
    
    target_rank = 27
    
    # Strategy 1: Merge two terms in Block 4 (the 2x1*1x5 schoolbook part)
    # Block 4 uses ranks 18-27 (10 terms). Try to use only 9.
    # The last two terms (ranks 26, 27) correspond to k=4:
    # U[2][26]=1, U[5][27]=1, V[14][26]=1, V[14][27]=1, W[8][26]=1, W[9][27]=1
    
    # Try to fold rank 27 into rank 26 with a coefficient
    for coeff in [0.5, 1.0, -0.5, -1.0, 1.5, -1.5]:
        if time.time() > deadline - 5:
            break
        
        U_try = [row[:target_rank] for row in U_base]
        V_try = [row[:target_rank] for row in V_base]
        W_try = [row[:target_rank] for row in W_base]
        
        # Fold rank 27 into rank 26
        # Original: U[2][26]=1, V[14][26]=1, W[8][26]=1
        #           U[5][27]=1, V[14][27]=1, W[9][27]=1
        # Try: modify rank 26 to handle both
        # This won't work directly since they affect different W entries
        
        # Alternative: try to use a shared structure
        # Set up a new rank-27 column that combines aspects
        U_try[2][26] = U_base[2][26]  # =1
        U_try[5][26] = coeff          # was U[5][27]=1, now scaled
        V_try[14][26] = V_base[14][26]  # =1
        W_try[8][26] = W_base[8][26]    # =1
        W_try[9][26] = coeff            # was W[9][27]=1, now scaled
        
        if _verify_decomposition(U_try, V_try, W_try, n, m, p, num_trials=30):
            return {"U": U_try, "V": V_try, "W": W_try}
    
    # Strategy 2: Try to modify Block 3 (2x2*2x1, ranks 14-17)
    for coeff in [0.5, 1.0, -0.5, -1.0]:
        if time.time() > deadline - 5:
            break
        
        U_try = [row[:target_rank] for row in U_base]
        V_try = [row[:target_rank] for row in V_base]
        W_try = [row[:target_rank] for row in W_base]
        
        # Block 3: ranks 14-17
        # Try to merge rank 17 into rank 16
        U_try[4][16] = U_base[4][16] + coeff * U_base[4][17]
        V_try[9][16] = V_base[9][16] + coeff * V_base[9][17]
        W_try[9][16] = W_base[9][16] + coeff * W_base[9][17]
        
        if _verify_decomposition(U_try, V_try, W_try, n, m, p, num_trials=30):
            return {"U": U_try, "V": V_try, "W": W_try}
    
    # Strategy 3: Randomized search with half-integer constraints
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        U_try = [row[:target_rank] for row in U_base]
        V_try = [row[:target_rank] for row in V_base]
        W_try = [row[:target_rank] for row in W_base]
        
        # Perturb last 3 columns randomly
        for r in range(target_rank - 3, target_rank):
            for i in range(len(U_try)):
                if random.random() < 0.3:
                    U_try[i][r] = random.choice([-1, -0.5, 0, 0.5, 1])
            for i in range(len(V_try)):
                if random.random() < 0.3:
                    V_try[i][r] = random.choice([-1, -0.5, 0, 0.5, 1])
            for i in range(len(W_try)):
                if random.random() < 0.3:
                    W_try[i][r] = random.choice([-1, -0.5, 0, 0.5, 1])
        
        if _verify_decomposition(U_try, V_try, W_try, n, m, p, num_trials=20):
            return {"U": U_try, "V": V_try, "W": W_try}
    
    return None

BET: {mechanism: "Structured search over half-integer parameter space with Schwartz-Zippel verification may find a rank-27 decomposition by testing specific structural modifications to the rank-28 base", q: 0.12, delta: [0, 0.0357]}
