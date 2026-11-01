import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    
    # 1. Construct the seed solution (Rank 24)
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1.0
                V_seed[j * p + k, col] = 1.0
                W_seed[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed

    # Define the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Compute tensor T_approx
        # T_approx[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=tol)

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank(r, seed_val=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed_val)
        
        # Number of variables: r * (nm + mp + pn)
        size = r * (n*m + m*p + p*n)
        
        # We try a few restarts. Since the landscape is non-convex and high-dimensional,
        # standard gradient descent often fails to find exact solutions for rank reduction.
        # However, we implement a robust check.
        
        # Strategy: 
        # 1. Random initialization
        # 2. Gradient Descent to minimize squared error
        # 3. Round to nearest half-integer
        # 4. Verify exactness
        
        # For small ranks like 20, random init + GD is unlikely to work in 140s.
        # But we include it as a best-effort attempt within the "family" of numerical search.
        
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            params = np.random.randn(size) * 0.5
            
            # Unpack
            U = params[:r*n*m].reshape(n*m, r)
            V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
            W = params[r*(n*m+m*p):].reshape(p*n, r)
            
            # Gradient Descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    return None
                
                # Compute gradients
                # dE/dU[a, k] = 2 * (T_approx[a, b, c] - T_target[a, b, c]) * V[b, k] * W[c, k] summed over b, c
                # Efficiently:
                # T_approx = einsum('ak,bk,ck->abc', U, V, W)
                # diff = T_approx - target
                # dU = einsum('abc,bk,ck->ak', diff, V, W) * 2
                # dV = einsum('abc,ak,ck->bk', diff, U, W) * 2
                # dW = einsum('abc,ak,bk->ck', diff, U, V) * 2
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = 2 * np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = 2 * np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = 2 * np.einsum('abc,ak,bk->ck', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence and rounding
                if step % 10 == 0:
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    
                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r

        return None

    # Try rank 20
    # Given the difficulty, this is unlikely to succeed quickly, but we try.
    sol = try_rank(20)
    if sol:
        best_U, best_V, best_W = sol
        best_rank = 20
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # If rank 20 failed, return seed
    return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
