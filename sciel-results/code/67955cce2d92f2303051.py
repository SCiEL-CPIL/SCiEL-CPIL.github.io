import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook solution as fallback
    U_best = [[0] * rank_schoolbook for _ in range(n * m)]
    V_best = [[0] * rank_schoolbook for _ in range(m * p)]
    W_best = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_best
    best_V = V_best
    best_W = W_best
    
    # Helper to verify exactness of integer/half-integer solution
    def verify_exact(U, V, W, n, m, p):
        r = len(U[0])
        # Check all constraints: sum_r W[k*n+i,r] U[i*m+j1,r] V[j2*p+k,r] == delta(j1, j2)
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for r_idx in range(r):
                            s += W[k*n+i][r_idx] * U[i*m+j1][r_idx] * V[j2*p+k][r_idx]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def convert_to_lists(U_np, V_np, W_np):
        return [
            [float(x) for x in row] for row in U_np
        ], [
            [float(x) for x in row] for row in V_np
        ], [
            [float(x) for x in row] for row in W_np
        ]

    # Try to find a lower rank solution
    # Candidate ranks: start from lower bound max(n*m, m*p, n*p) = 25
    # Known bounds for 3x5x5 might be around 30-40.
    # We'll try a range of ranks.
    min_rank = max(n*m, m*p, n*p)
    # We'll try ranks from min_rank to rank_schoolbook - 1
    # But to save time, we'll sample some ranks or go sequentially if fast.
    # ALS is O(R * (n*p*m^2 + ...)) which is small.
    
    # We'll try ranks in a specific order: lower is better.
    # But finding a solution at 25 is unlikely.
    # Let's try a few random restarts for each rank.
    
    np.random.seed(42)
    
    # Pre-compute indices for efficiency
    # For W update: for each i, k, build A_W (m*m, R)
    # For U update: for each i, j1, build A_U (p*m, R)
    # For V update: for each j2, k, build A_V (n*m, R)
    
    # We will try ranks from 30 to 60 in steps of 1 or 2.
    # If we find one, we stop.
    
    for rank in range(30, rank_schoolbook):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS Optimization
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # A_W shape (m*m, rank)
                        A_W = np.zeros((m*m, rank))
                        b = np.zeros(m*m)
                        # Fill A_W and b
                        # b is vec(I_m)
                        for j in range(m):
                            b[j*m+j] = 1.0
                        for j1 in range(m):
                            for j2 in range(m):
                                A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                        # Solve
                        w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                        W[k*n+i, :] = w
                
                # Update U
                for i in range(n):
                    for j1 in range(m):
                        # A_U shape (p*m, rank)
                        A_U = np.zeros((p*m, rank))
                        b = np.zeros(p*m)
                        for k in range(p):
                            for j2 in range(m):
                                A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                                b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                        u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                        U[i*m+j1, :] = u
                        
                # Update V
                for j2 in range(m):
                    for k in range(p):
                        # A_V shape (n*m, rank)
                        A_V = np.zeros((n*m, rank))
                        b = np.zeros(n*m)
                        for i in range(n):
                            for j1 in range(m):
                                A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                                b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                        v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                        V[j2*p+k, :] = v
                        
            # Check convergence and exactness
            # Calculate error
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = 0.0
                            for r_idx in range(rank):
                                s += W[k*n+i, r_idx] * U[i*m+j1, r_idx] * V[j2*p+k, r_idx]
                            target = 1.0 if j1 == j2 else 0.0
                            err += (s - target)**2
                            
            if err < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                U_list = convert_to_lists(U_snap, None, None)
                V_list = convert_to_lists(None, V_snap, None)
                W_list = convert_to_lists(None, None, W_snap)
                
                if verify_exact(U_list, V_list, W_list, n, m, p):
                    best_rank = rank
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    found = True
                    break
        
        if found:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}

