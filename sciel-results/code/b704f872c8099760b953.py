import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target Tensor T
    # T[alpha, beta, gamma] = 1 if (i, j, k) is valid, else 0
    # alpha = i*m + j
    # beta  = j*p + k
    # gamma = k*n + i
    
    T = np.zeros((size_U, size_V, size_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Precompute A for all gammas for a given U, V
    # A_gamma is (size_U * size_V, R)
    # A_gamma[alpha * size_V + beta, r] = U[alpha, r] * V[beta, r]
    
    # We will use a random search strategy:
    # 1. Randomly generate U and V with small integer/half-integer entries.
    # 2. Solve for W using linear least squares for each gamma.
    # 3. Check if W entries are half-integers and if the decomposition is exact.
    
    # Use a wider range of values and more trials for lower ranks
    # The search space is huge. Random search is unlikely to find rank < 30.
    # However, we try.
    
    # Optimization: Vectorize the check for all gammas at once?
    # We need to solve 10 systems.
    # Let's optimize the inner loop.
    
    # Strategy:
    # For a given R, generate random U, V.
    # Construct the 10 matrices A_gamma.
    # Solve for W_gamma.
    # Check if snapped W_gamma gives zero residual.
    
    # To speed up, we can check a subset of gammas first?
    # No, all must match.
    
    # Let's try to be smarter about initialization.
    # If we fix U and V to be sparse, it might help?
    # Or use specific structures?
    
    # Let's stick to random but optimize the code.
    
    # We can precompute the indices for A construction.
    # A[alpha, beta, r] = U[alpha, r] * V[beta, r]
    # This is just an outer product of columns of U and V, stacked.
    
    # A = np.einsum('ar,br->abr', U, V).reshape(-1, R)
    # This is fast enough.
    
    # Let's try ranks 15 to 30.
    
    # Increase trials for lower ranks if time permits.
    # But random search is very inefficient.
    
    # Alternative: Local search / Hill climbing?
    # Start from schoolbook (R=30) and try to reduce rank?
    # Hard to do directly.
    
    # Let's just run the random search with a tighter time loop.
    
    possible_values = [-1, 0, 1] # Smaller values might lead to smaller W?
    # Actually, scale doesn't matter for feasibility, only ratios.
    # But integer constraints matter.
    
    # Let's use [-2, -1, 0, 1, 2] to have more variation.
    
    for R in range(15, 31):
        if time.time() > deadline:
            break
            
        # Adjust trials based on remaining time and rank
        # Lower ranks are harder, so maybe more trials?
        # But if R is close to 30, it's easier.
        
        # Let's just run as many as possible, but keep it low to avoid overhead
        num_trials = 2000 # Reduced to ensure we stay well within time limits
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            U = np.random.choice(possible_values, size=(size_U, R))
            V = np.random.choice(possible_values, size=(size_V, R))
            
            # Check if U and V are "good" enough?
            # E.g., rank(U) == R? Not necessary.
            
            W = np.zeros((size_W, R))
            valid_W = True
            
            # Precompute A for all gammas?
            # A_gamma depends on gamma only through target, not A itself.
            # Wait, A is the same for all gammas?
            # A_gamma[alpha, beta, r] = U[alpha, r] * V[beta, r]
            # Yes, A is independent of gamma!
            # The target changes, but A is constant for a fixed U, V.
            
            A = np.einsum('ar,br->abr', U, V).reshape(-1, R)
            
            # Solve for each gamma
            for gamma in range(size_W):
                target = T[:, :, gamma].flatten()
                
                try:
                    W_gamma, residuals, rank_A, s = np.linalg.lstsq(A, target, rcond=None)
                except np.linalg.LinAlgError:
                    valid_W = False
                    break
                    
                W_gamma_snapped = np.round(W_gamma * 2) / 2.0
                
                residual = A @ W_gamma_snapped - target
                if np.max(np.abs(residual)) > 1e-9:
                    valid_W = False
                    break
                    
                W[gamma, :] = W_gamma_snapped
            
            if valid_W:
                # Final verification
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-9):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
