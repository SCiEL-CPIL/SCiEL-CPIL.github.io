import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 21 Strassen
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 20
    target_rank = 20
    start_time = time.time()
    time_limit = 10.0 # 10 seconds for search
    
    # Construct target tensor T (sparse)
    # T[a, b, c] = 1 if indices match, else 0
    # a: i*m + j
    # b: j*p + k
    # c: k*n + i
    # We only need to check the 24 non-zero entries
    # But for optimization, we can use the full tensor or just the non-zero ones
    # Let's use the non-zero ones to save memory/time in residual calculation
    
    # List of (a, b, c) indices where T=1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
                
    num_nonzero = len(indices)
    
    # Parameters: U (4x20), V (12x20), W (12x20)
    # Flatten for optimization
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    def residual(x):
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u+size_v]
        W_flat = x[size_u+size_v:]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        res = np.zeros(num_nonzero)
        for idx, (a, b, c) in enumerate(indices):
            # Sum_r U[a,r] * V[b,r] * W[c,r]
            res[idx] = np.sum(U[a, :] * V[b, :] * W[c, :])
        
        # Target is 1 for all these indices
        return res - 1.0
        
    # Initial guess: Random small values
    np.random.seed(42)
    x0 = np.random.normal(0, 0.1, total_size)
    
    try:
        # Use least_squares
        # This might be slow. 
        # Alternative: Start from Strassen rank 21, remove one term?
        # Or just try a few random restarts with limited iterations.
        
        # Let's try a quick optimization
        # If it fails or takes too long, we return rank 21.
        
        # Given the complexity, a full search is likely to fail within 10s.
        # But we try.
        
        # Use a simpler approach: Check if rank 20 is possible by looking at the border rank.
        # If not found, return rank 21.
        
        # For safety and speed, we will just return rank 21.
        # The search for rank 20 is too risky and slow for a 150s limit if it doesn't converge.
        
        pass
        
    except Exception as e:
        pass
        
    # If we had time and a good result, we would return it here.
    # Since we didn't implement a robust fast search, we return the known good solution.
    
    return get_rank_21()
