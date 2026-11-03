import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    def get_T_u_v_w(u, v, w, dims):
        n, m, p = dims
        i = u // m
        j = u % m
        j2 = v // p
        k = v % p
        k2 = w // n
        i2 = w % n
        if i == i2 and j == j2 and k == k2:
            return 1.0
        return 0.0

    def check_solution(U, V, W, dims):
        n, m, p = dims
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[u, r] * V[v, r] * W[w, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        # Check random zero entries
        for _ in range(100):
            u = np.random.randint(0, n*m)
            v = np.random.randint(0, m*p)
            w = np.random.randint(0, p*n)
            if get_T_u_v_w(u, v, w, dims) == 0:
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err

    def als_step(U, V, W, rank, dims):
        n, m, p = dims
        # Update U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                B = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    for r in range(rank):
                        B[k, r] = V[v, r] * W[w, r]
                U[u, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update V
        for j in range(m):
            for k in range(p):
                v = j * p + k
                B = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    for r in range(rank):
                        B[i, r] = U[u, r] * W[w, r]
                V[v, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update W
        for k in range(p):
            for i in range(n):
                w = k * n + i
                B = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    for r in range(rank):
                        B[j, r] = U[u, r] * V[v, r]
                W[w, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)

    def find_decomp(n, m, p, target_rank, max_time, restarts=50):
        """Attempts to find a decomposition of rank target_rank for n x m x p."""
        start_time = time.time()
        deadline = start_time + max_time
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(1234 + restart)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            for iter in range(300):
                if time.time() > deadline:
                    break
                als_step(U, V, W, target_rank, (n, m, p))
                
                if iter % 10 == 0 or iter == 299:
                    err = check_solution(U, V, W, (n, m, p))
                    if err < 1e-8:
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        err_snap = check_solution(U_snap, V_snap, W_snap, (n, m, p))
                        if err_snap < 1e-8:
                            return U_snap, V_snap, W_snap
                            
        return None

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W

    start_time = time.time()
    deadline = start_time + 115

    # Strategy 1: Find 3x3x3 rank 22, then extend to 3x3x4 rank 31
    # The 3x3x4 problem can be split into 3x3x3 (for k=0,1,2) and 3x3x1 (for k=3)
    # Rank 3x3x3 is 22. Rank 3x3x1 is 9. Total 31.
    
    # Try to find 3x3x3 rank 22
    # This might take time, so we give it a budget.
    # If we find it, we construct the 31-rank solution.
    
    U_333, V_333, W_333 = None, None, None
    if time.time() < deadline - 60: # Leave 60s for construction and fallback
        U_333, V_333, W_333 = find_decomp(3, 3, 3, 22, 50, restarts=20)
        
    if U_333 is not None:
        # Construct 3x3x4 solution
        # U is 9x31, V is 12x31, W is 12x31
        # First 22 columns come from 3x3x3 decomposition
        # But we need to map indices correctly.
        # In 3x3x3: u=i*3+j, v=j*3+k, w=k*3+i for k in 0..2
        # In 3x3x4: u=i*3+j, v=j*4+k, w=k*3+i for k in 0..3
        
        U_new = np.zeros((9, 31))
        V_new = np.zeros((12, 31))
        W_new = np.zeros((12, 31))
        
        # Copy 3x3x3 part (cols 0-21)
        # U_333 is 9x22. U_new[0:9, 0:22] = U_333
        U_new[:, 0:22] = U_333
        
        # V_333 is 9x22 (indices j*3+k for k=0..2).
        # V_new is 12x31 (indices j*4+k for k=0..3).
        # For k=0,1,2: V_new[j*4+k, 0:22] = V_333[j*3+k, 0:22]
        for j in range(3):
            for k in range(3):
                V_new[j*4+k, 0:22] = V_333[j*3+k, 0:22]
        
        # W_333 is 9x22 (indices k*3+i for k=0..2).
        # W_new is 12x31 (indices k*3+i for k=0..3).
        # For k=0,1,2: W_new[k*3+i, 0:22] = W_333[k*3+i, 0:22]
        W_new[0:9, 0:22] = W_333
        
        # Add 3x3x1 part (cols 22-30)
        # This handles C_{i3} = sum_j A_{ij} B_{j3}
        # Terms: A_{ij} * B_{j3} * C_{i3}
        # u = i*3+j
        # v = j*4+3
        # w = 3*3+i = 9+i
        for i in range(3):
            for j in range(3):
                col = 22 + i*3 + j
                u = i*3 + j
                v = j*4 + 3
                w = 9 + i
                U_new[u, col] = 1
                V_new[v, col] = 1
                W_new[w, col] = 1
                
        # Verify
        err = check_solution(U_new, V_new, W_new, (3, 3, 4))
        if err < 1e-8:
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    # Fallback: Try to find 3x3x4 rank 31 directly or schoolbook
    # If 3x3x3 search failed, try direct search for 3x3x4 with rank 31
    if time.time() < deadline - 10:
        res = find_decomp(3, 3, 4, 31, 10, restarts=5)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final Fallback: Schoolbook 36
    U, V, W = schoolbook(3, 3, 4)
    return {"U": U, "V": V, "W": W}
