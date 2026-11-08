import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Trivial Rank 64
    # Strassen Rank 56
    
    # Try to find Rank 52 using a randomized algebraic solver
    # The tensor T has shape (n*m, m*p, p*n) = (8, 32, 16)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def error_func(params, rank):
        # params is flattened U, V, W
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        U = params[:n * m * rank].reshape(n * m, rank)
        V = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
        W = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
        
        # Reconstruct tensor
        # T_reconstructed[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # This is expensive to compute fully.
        # Instead, compute the residual in a flattened way?
        # Or just compute the sum for non-zero entries?
        # The tensor is sparse.
        
        # Efficient check:
        # For each rank r, we have a rank-1 tensor U[:,r] V[:,r] W[:,r]
        # We want sum_r (U[:,r] V[:,r] W[:,r]) == T
        
        # We can compute the difference for each element
        # But T is fixed.
        
        # Let's compute the reconstructed tensor
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow for rank 52?
        # 8*32*16*52 ~ 200k operations. Fast.
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_rec - T
        return diff.flatten()

    # Try rank 52
    target_rank = 52
    # Initialize with random values
    np.random.seed(42)
    init_params = np.random.randn((n * m + m * p + p * n) * target_rank)
    
    # Use least_squares
    try:
        res = least_squares(error_func, init_params, args=(target_rank,), method='lm', max_nfev=1000)
        if res.success and res.cost < 1e-6:
            # Extract U, V, W
            U = res.x[:n * m * target_rank].reshape(n * m, target_rank)
            V = res.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
            W = res.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
            
            # Round to nearest integer/half-integer if close
            # Check if entries are close to 0.5 or integers
            # For now, return as is (floats are allowed? "integer/half-integer entries")
            # The prompt says "integer/half-integer entries".
            # If the solver gives 0.4999, we should round to 0.5.
            
            def round_half(x):
                # Round to nearest 0.5
                return np.round(x * 2) / 2.0
            
            U = round_half(U)
            V = round_half(V)
            W = round_half(W)
            
            # Verify the rounded solution
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_rec, T, atol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception:
        pass

    # Fallback to Strassen Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            u_entries, v_entries, w_entries = S[r]
            for idx, val in u_entries:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_off, lj + j_off
                U[gi * 4 + gj][cr] += val
            for idx, val in v_entries:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_off, lk + k_off
                V[gj * 8 + gk][cr] += val
            for idx, val in w_entries:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_off, li + i_off
                W[gk * 2 + gi][cr] += val
                
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(block_idx * 7, 0, j_off, k_off)
            block_idx += 1
            
    return {"U": U, "V": V, "W": W}
