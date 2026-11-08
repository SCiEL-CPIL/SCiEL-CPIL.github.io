"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 if valid matmul term"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        diff = np.max(np.abs(total - expected))
        return False, diff
    return True, 0.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P2 = (A_1,j0 + A_1,j1) B_j0,k0
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                
                # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                
                # P5 = (A_0,j0 + A_0,j1) B_j1,k1
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                
                # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        # Remaining j=4
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build baseline
    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    best = baseline
    best_rank = 54
    
    if time.time() > deadline - 30:
        return best
    
    # Systematic approach: fix U and V, solve for W linearly
    # For fixed U,V, for each (k,i), we need:
    #   For each j: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is: M_{k,i} @ W[k*n+i,:] = ones(m)
    # where M_{k,i}[j,r] = U[i*m+j,r] * V[j*p+k,r]
    
    import random
    random.seed(42)
    
    def _try_systematic(target_rank, n, m, p, max_attempts=100):
        """Try to find a decomposition by fixing U,V and solving for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize U and V with structured random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            
            # Strategy: make U and V have good structure
            # Each column r should have some pattern
            for r in range(target_rank):
                # For U: pick which (i,j) pairs are nonzero
                # Use a pattern based on r
                num_nonzero_u = random.randint(2, 4)
                rows_u = random.sample(range(n*m), min(num_nonzero_u, n*m))
                for row in rows_u:
                    U[row, r] = random.choice([-1, 1])
                
                # For V: pick which (j,k) pairs are nonzero
                num_nonzero_v = random.randint(2, 4)
                rows_v = random.sample(range(m*p), min(num_nonzero_v, m*p))
                for row in rows_v:
                    V[row, r] = random.choice([-1, 1])
            
            # For each (k, i), solve for W[k*n + i, :]
            W = np.zeros((p*n, target_rank))
            feasible = True
            for k in range(p):
                for i in range(n):
                    # Build system matrix M[j,r] = U[i*m+j,r] * V[j*p+k,r]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    # Solve M @ w = ones(m)
                    try:
                        # Check rank
                        rank_m = np.linalg.matrix_rank(M, tol=1e-10)
                        if rank_m < m:
                            feasible = False
                            break
                        
                        # Use least squares and check residual
                        w, residuals, rank_ls, sv = np.linalg.lstsq(M, np.ones(m), rcond=1e-10)
                        if len(residuals) > 0 and residuals[0] > 1e-8:
                            # Try pinv
                            w = np.linalg.pinv(M, rcond=1e-10) @ np.ones(m)
                            if np.linalg.norm(M @ w - np.ones(m)) > 1e-8:
                                feasible = False
                                break
                        W[k*n+i, :] = w
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            valid, diff = _verify(U_list, V_list, W_list, n, m, p)
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 53 down to 48
    for target_rank in range(53, 47, -1):
        if time.time() > deadline - 15:
            break
        if best_rank <= target_rank:
            continue
        
        result = _try_systematic(target_rank, n, m, p, max_attempts=50)
        if result:
            best = result
            best_rank = target_rank
    
    return best
# EVOLVE-BLOCK-END
