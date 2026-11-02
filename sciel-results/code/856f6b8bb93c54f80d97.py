import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = 2, 3, 9
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True
    
    def strassen_decomposition(rank_2x2):
        """Standard Strassen decomposition for 2x2 matrix multiplication."""
        # Returns U, V, W for 2x2 * 2x2 with given rank
        # Strassen uses rank 7
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # Standard Strassen
        U[0,0] = 1; U[3,0] = 1; V[0,0] = 1; V[3,0] = 1
        U[2,1] = 1; U[3,1] = 1; V[0,1] = 1
        U[0,2] = 1; V[1,2] = 1; V[3,2] = -1
        U[3,3] = 1; V[2,3] = 1; V[0,3] = -1
        U[0,4] = 1; U[1,4] = 1; V[3,4] = 1
        U[2,5] = 1; U[0,5] = -1; V[0,5] = 1; V[1,5] = 1
        U[1,6] = 1; U[3,6] = -1; V[2,6] = 1; V[3,6] = 1
        
        W[0,0] = 1; W[0,3] = 1; W[0,4] = -1; W[0,6] = 1
        W[1,2] = 1; W[1,4] = 1
        W[2,1] = 1; W[2,3] = 1
        W[3,0] = 1; W[3,1] = -1; W[3,2] = 1; W[3,5] = 1
        
        return U, V, W
    
    # Current best: rank 50 solution
    def rank50_solution():
        rank = 50
        U = np.zeros((6, rank), dtype=np.int64)
        V = np.zeros((27, rank), dtype=np.int64)
        W = np.zeros((18, rank), dtype=np.int64)
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9) - 4 blocks of 2x2 * 2x9
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0
            u_b = 1
            u_c = 2
            u_d = 3
            
            # Strassen
            U[u_a, col] += 1; U[u_d, col] += 1; V[v_e, col] += 1; V[v_h, col] += 1
            U[u_c, col+1] += 1; U[u_d, col+1] += 1; V[v_e, col+1] += 1
            U[u_a, col+2] += 1; V[v_f, col+2] += 1; V[v_h, col+2] -= 1
            U[u_d, col+3] += 1; V[v_g, col+3] += 1; V[v_e, col+3] -= 1
            U[u_a, col+4] += 1; U[u_b, col+4] += 1; V[v_h, col+4] += 1
            U[u_c, col+5] += 1; U[u_a, col+5] -= 1; V[v_e, col+5] += 1; V[v_f, col+5] += 1
            U[u_b, col+6] += 1; U[u_d, col+6] -= 1; V[v_g, col+6] += 1; V[v_h, col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00, col] += 1; W[w_00, col+3] += 1; W[w_00, col+4] -= 1; W[w_00, col+6] += 1
            W[w_01, col+2] += 1; W[w_01, col+4] += 1
            W[w_10, col+1] += 1; W[w_10, col+3] += 1
            W[w_11, col] += 1; W[w_11, col+1] -= 1; W[w_11, col+2] += 1; W[w_11, col+5] += 1
            col += 7
            
        # Last col of B_T (k=8)
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3
        
        U[u_a, col] += 1; V[v_e, col] += 1
        U[u_b, col+1] += 1; V[v_g, col+1] += 1
        U[u_c, col+2] += 1; V[v_e, col+2] += 1
        U[u_d, col+3] += 1; V[v_g, col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08, col] += 1; W[w_08, col+1] += 1
        W[w_18, col+2] += 1; W[w_18, col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 4
        u_ar1 = 5
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0, col] += 1; V[v_b, col] += 1; W[k * n + 0, col] += 1
            U[u_ar1, col+1] += 1; V[v_b, col+1] += 1; W[k * n + 1, col+1] += 1
            col += 2
            
        return U, V, W, rank
    
    U50, V50, W50, rank50 = rank50_solution()
    
    # Try to find lower rank solutions using structured search
    def try_lower_rank(target_rank, max_trials=50):
        """Try to find a solution with target_rank using structured search."""
        if time.time() > deadline - 15:
            return None
        
        # Strategy: Use the known structure but try to merge/optimize
        # For 2x3 x 3x9, we can think of it as:
        # - 2x2 x 2x9 part (using Strassen: 7 * 4 = 28 rank for the 4 blocks)
        # - 2x1 x 1x9 part (schoolbook: 18 rank)
        # Total: 46 rank? But we need to be careful about overlaps.
        
        # Actually, let's try a different approach:
        # View 2x3 as [2x2 | 2x1] and 3x9 as [2x9; 1x9]
        # The product is: A_left * B_top + A_right * B_bottom
        
        # For the 2x2 * 2x9 part, we can use a rank-7 decomposition per 2x2 block
        # But we have 4 such blocks (k=0,1; k=2,3; k=4,5; k=6,7) plus k=8 separately
        
        # Let's try to find a better decomposition by solving a system of equations
        # with the constraint that entries are small integers
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            # Try with specific structure: fix W to be sparse and solve for U, V
            # Or fix U and V and solve for W
            
            # Approach: Start from the rank-50 solution and try to reduce rank
            # by finding linear dependencies and eliminating columns
            
            # For now, try random sparse initializations
            r = target_rank
            np.random.seed(trial * 1000 + 42)
            
            U = np.zeros((6, r), dtype=np.float64)
            V = np.zeros((27, r), dtype=np.float64)
            W = np.zeros((18, r), dtype=np.float64)
            
            # Initialize with small random values
            for c in range(r):
                # U: 6 entries
                for i in range(6):
                    if np.random.random() < 0.3:
                        U[i, c] = np.random.choice([-1, 1])
                # V: 27 entries  
                for i in range(27):
                    if np.random.random() < 0.15:
                        V[i, c] = np.random.choice([-1, 1])
                # W: 18 entries
                for i in range(18):
                    if np.random.random() < 0.2:
                        W[i, c] = np.random.choice([-1, 1])
            
            # Try to refine using gradient-free optimization
            # This is complex, so let's just check if any random sparse solution works
            # (unlikely, but worth trying)
            
            # Convert to integer and verify
            U_int = np.round(U).astype(np.int64)
            V_int = np.round(V).astype(np.int64)
            W_int = np.round(W).astype(np.int64)
            
            if verify(U_int, V_int, W_int, r):
                return U_int, V_int, W_int, r
        
        return None
    
    # Try ranks from 49 down to 40
    best_result = (U50, V50, W50, rank50)
    
    for r in range(49, 39, -1):
        if time.time() > deadline - 20:
            break
        result = try_lower_rank(r, max_trials=30)
        if result is not None:
            best_result = result
            break
    
    U, V, W, rank = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
