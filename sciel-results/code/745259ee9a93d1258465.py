import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]
_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE:
        # Return a trivial rank-64 (schoolbook) decomposition as fallback
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Default rank 56 solution (Strassen based)
    rank_56 = 56
    U_56 = [[0] * rank_56 for _ in range(n * m)]
    V_56 = [[0] * rank_56 for _ in range(m * p)]
    W_56 = [[0] * rank_56 for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block_56(base_rank, i_base, j_base, k_base, U, V, W):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val

    # Populate rank 56
    add_block_56(0, 0, 0, 0, U_56, V_56, W_56)
    add_block_56(7, 0, 0, 2, U_56, V_56, W_56)
    add_block_56(14, 0, 0, 4, U_56, V_56, W_56)
    add_block_56(21, 0, 0, 6, U_56, V_56, W_56)
    add_block_56(28, 0, 2, 0, U_56, V_56, W_56)
    add_block_56(35, 0, 2, 2, U_56, V_56, W_56)
    add_block_56(42, 0, 2, 4, U_56, V_56, W_56)
    add_block_56(49, 0, 2, 6, U_56, V_56, W_56)

    # Try to find a rank-13 decomposition for 2x2x4
    # If found, use it to build rank 52 for 2x4x8
    best_result = None
    target_rank = 13
    n_sub, m_sub, p_sub = 2, 2, 4
    
    def objective_224(params):
        # params: flattened U (4xR), V (4xR), W (8xR)
        U_flat = params[:4*target_rank]
        V_flat = params[4*target_rank:8*target_rank]
        W_flat = params[8*target_rank:]
        
        U = U_flat.reshape(2, 2, target_rank)
        V = V_flat.reshape(2, 2, target_rank)
        W = W_flat.reshape(4, 2, target_rank) # W is (k, i, r) -> (p_sub, n_sub, R)
        
        # Reconstruct T[i,j,k]
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        target = np.ones((2, 2, 4))
        err = np.sum((T - target)**2)
        return err

    def grad_224(params):
        # Numerical gradient is too slow. Use analytic or finite difference with small step.
        # Given time constraints, we might skip gradient and use Powell or Nelder-Mead on a reduced set?
        # Or just use L-BFGS-B with numerical gradient.
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = objective_224(params)
        for i in range(len(params)):
            old = params[i]
            params[i] = old + eps
            f1 = objective_224(params)
            params[i] = old
            grad[i] = (f1 - f0) / eps
        return grad

    found_rank_13 = False
    best_params = None
    
    # Time budget for search: 100s
    search_deadline = time.time() + 100
    np.random.seed(42)
    
    # Initial guess: Strassen 2x2x2 twice (rank 14), then drop one rank?
    # Or random init.
    
    # Let's try a few random restarts
    for restart in range(20):
        if time.time() > search_deadline:
            break
            
        # Random initialization
        params = np.random.randn(4*target_rank + 4*target_rank + 8*target_rank) * 0.1
        
        # Optimization
        try:
            res = minimize(objective_224, params, method='L-BFGS-B', jac=grad_224, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
            if res.success and res.fun < 1e-9:
                found_rank_13 = True
                best_params = res.x
                break
        except Exception:
            pass

    if found_rank_13:
        # Construct rank 52 solution
        rank_52 = 52
        U_52 = [[0] * rank_52 for _ in range(n * m)]
        V_52 = [[0] * rank_52 for _ in range(m * p)]
        W_52 = [[0] * rank_52 for _ in range(p * n)]
        
        # Extract 2x2x4 decomposition
        U_224 = best_params[:4*target_rank].reshape(2, 2, target_rank)
        V_224 = best_params[4*target_rank:8*target_rank].reshape(2, 2, target_rank)
        W_224 = best_params[8*target_rank:].reshape(4, 2, target_rank)
        
        # We need two copies: one for (i in 0,1; j in 0,1; k in 0,3) and one for (i in 0,1; j in 2,3; k in 0,3)
        # Wait, the 2x4x8 tensor is composed of two 2x2x4 blocks in the j-dimension?
        # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3).
        # B is 4x8. Split into B0 (rows 0,1) and B1 (rows 2,3).
        # C = A0 B0 + A1 B1.
        # A0 is 2x2, B0 is 2x8.
        # B0 is 2x8. Split into B00 (cols 0,1), B01 (cols 2,3), B02 (cols 4,5), B03 (cols 6,7).
        # This is 4 blocks of 2x2.
        # My 2x2x4 decomposition handles a 2x2 A and 2x4 B.
        # So I can use the 2x2x4 decomposition for A0 and B0 (which is 2x8)?
        # No, B0 is 2x8. The 2x2x4 decomposition is for 2x2 A and 2x4 B.
        # So I can use it for A0 and B0_first_half (cols 0,1,2,3)?
        # B0 is 2x8. I can split B0 into two 2x4 blocks: B0a (cols 0-3) and B0b (cols 4-7).
        # Then A0 * B0 = A0 * B0a + A0 * B0b.
        # Each term is 2x2 * 2x4.
        # So I need two 2x2x4 decompositions for the A0 part.
        # Similarly for A1 part: A1 * B1 = A1 * B1a + A1 * B1b.
        # Total 4 blocks of 2x2x4.
        # Rank = 4 * 13 = 52.
        
        # Mapping:
        # Block 1: A0 (i=0,1; j=0,1) * B0a (j=0,1; k=0,1,2,3)
        # Block 2: A0 (i=0,1; j=0,1) * B0b (j=0,1; k=4,5,6,7)
        # Block 3: A1 (i=0,1; j=2,3) * B1a (j=2,3; k=0,1,2,3)
        # Block 4: A1 (i=0,1; j=2,3) * B1b (j=2,3; k=4,5,6,7)
        
        def add_block_224(base_rank, i_base, j_base, k_base, U, V, W):
            for r in range(target_rank):
                cr = base_rank + r
                for li in range(2):
                    for lj in range(2):
                        gi = i_base + li
                        gj = j_base + lj
                        U[gi * 4 + gj][cr] += U_224[li, lj, r]
                for lj in range(2):
                    for lk in range(4):
                        gj = j_base + lj
                        gk = k_base + lk
                        V[gj * 8 + gk][cr] += V_224[lj, lk, r]
                for lk in range(4):
                    for li in range(2):
                        gk = k_base + lk
                        gi = i_base + li
                        W[gk * 2 + gi][cr] += W_224[lk, li, r]
        
        add_block_224(0, 0, 0, 0, U_52, V_52, W_52)
        add_block_224(13, 0, 0, 4, U_52, V_52, W_52)
        add_block_224(26, 0, 2, 0, U_52, V_52, W_52)
        add_block_224(39, 0, 2, 4, U_52, V_52, W_52)
        
        # Verify
        try:
            U_arr = np.array(U_52, dtype=float)
            V_arr = np.array(V_52, dtype=float)
            W_arr = np.array(W_52, dtype=float)
            U3 = U_arr.reshape(n, m, rank_52)
            V3 = V_arr.reshape(m, p, rank_52)
            W3 = W_arr.reshape(p, n, rank_52)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return {"U": U_52, "V": V_52, "W": W_52}
        except Exception:
            pass

    # Fallback to rank 56
    return {"U": U_56, "V": V_56, "W": W_56}
# EVOLVE-BLOCK-END
