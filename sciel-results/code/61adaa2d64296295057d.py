import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    n, m, p = DIMENSIONS

    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Fallback: Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verification function
    def verify(U, V, W, tol=1e-8):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T_comp[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We check against the target tensor T
        # For efficiency, we can compute the tensor and compare
        # But for large R, this is expensive. Let's do a spot check or full check depending on size.
        # Given small dims (9, 21, 21), full check is fine.
        
        # Use einsum for speed
        # T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        # This creates a 9x21x21 tensor which is 3969 elements. Very fast.
        
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check if T_comp matches T
        # T has 1s at specific indices, 0 elsewhere.
        # Due to floating point, we check allclose.
        if np.allclose(T_comp, T, atol=tol):
            return True
        
        # Sometimes numerical noise might make allclose fail if tol is too tight,
        # but for integer/half-integer solutions, it should be exact.
        # If we find a near-miss, we might try to round it?
        # For now, strict verification.
        return False

    # Strategy 1: Structured Random Search with Refinement
    # We search for low-rank decompositions.
    # We use a "block" structure inspired by the p=7 asymmetry if possible, 
    # but general random search with local refinement is robust.
    
    # Lower bound estimate: 
    # Rank >= max(n*m, m*p, p*n) is not a lower bound.
    # Rank >= n*m*p / min(n,m,p) is a loose bound? No.
    # For 3x3x3, rank is 23 (vs 27).
    # For 3x3x7, let's try ranks from 40 to 62.
    
    # We will iterate ranks from low to high.
    # For each rank, we try multiple random initializations and refine using gradient descent 
    # on the squared error.
    
    # To speed up, we can use a simpler objective or fewer iterations for higher ranks.
    
    ranks_to_try = list(range(40, 63))
    
    # Shuffle ranks slightly? No, we want the lowest rank possible.
    # But maybe rank 40 is hard. Let's try a few random ranks first to see if we can find anything quick?
    # No, the score is 1/rank (effectively, since score = seed/value and seed is rank 63? No, seed is the program score 1.0 which corresponds to rank 63? 
    # Wait, "score is the rank". "evaluator measures the raw value (lower is better)".
    # So if I return rank 63, score is 63? 
    # "score is seed/value, so 1.0 equals the seed program".
    # Seed program returns rank 63. So score = 63/63 = 1.0?
    # Or does the evaluator compute score = 63 / rank_found?
    # "score is the rank" ... "evaluator measures the raw value (lower is better)".
    # If the raw value IS the rank, then lower rank = better score.
    # If the search score is seed/value, and seed is 1.0 (which corresponds to rank 63), then value must be rank/63?
    # No, if value is rank, and seed program has rank 63, and score is 1.0, then Score = 63 / Rank?
    # If Rank=63, Score=1.
    # If Rank=50, Score=63/50 = 1.26.
    # This matches "above 1.0 is an improvement".
    
    # So we want to minimize Rank.
    
    # Let's try ranks from 40 to 62.
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        # The number of trials can be adjusted based on time remaining.
        num_trials = 5 if R < 50 else 2
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Random seed for reproducibility within the run
            np.random.seed(42 + R * 100 + trial)
            
            # Initialize U, V, W
            # Scaling matters for gradient descent.
            # If we initialize with small values, the tensor is small.
            # We want the tensor to be close to T (which has 1s).
            # Let's initialize with values around 1/sqrt(R) or similar?
            # Or just standard normal.
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Refine using L-BFGS-B
            # We flatten the parameters.
            # Objective: sum (T - T_comp)^2
            
            def objective(params):
                size_U = n * m * R
                size_V = m * p * R
                U_flat = params[0:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U_mat = U_flat.reshape(n*m, R)
                V_mat = V_flat.reshape(m*p, R)
                W_mat = W_flat.reshape(p*n, R)
                
                T_comp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                diff = T - T_comp
                return np.sum(diff * diff)
            
            # Gradient is expensive to compute analytically for each iteration if we do it naively,
            # but L-BFGS-B uses finite differences if gradient is not provided.
            # Providing gradient is much faster.
            # Let's implement the gradient.
            # L = sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # dL/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - T_comp[a,b,c]) * V[b,r] * W[c,r]
            
            def objective_and_grad(params):
                size_U = n * m * R
                size_V = m * p * R
                U_flat = params[0:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U_mat = U_flat.reshape(n*m, R)
                V_mat = V_flat.reshape(m*p, R)
                W_mat = W_flat.reshape(p*n, R)
                
                T_comp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                diff = T - T_comp
                loss = np.sum(diff * diff)
                
                # Gradients
                # grad_U[a, r] = -2 * sum_{b, c} diff[a, b, c] * V[b, r] * W[c, r]
                # This is a contraction of diff with V and W.
                # diff is (n*m, m*p, p*n)
                # We can compute this as:
                # grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
                # grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
                # grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
                
                grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
                grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
                grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                return loss, grad
            
            from scipy.optimize import minimize
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Run optimization
            # maxiter should be kept low to respect deadline.
            # We have ~140s total.
            # Let's allow 100 iterations per trial.
            try:
                res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-12})
                
                if res.fun < 1e-6:  # Close enough to consider a candidate
                    U_opt = res.x[0:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    # Verify
                    if verify(U_opt, V_opt, W_opt, tol=1e-5):
                        best_rank = R
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        break # Found a valid decomposition for this rank, try next (lower) rank? No, we are going low to high. So we found the lowest so far.
                        # Actually, we are iterating R from 40 to 62. If we find one at 45, we break the inner loop, but continue outer loop to check 44, 43... wait.
                        # My loop is `for R in range(40, 63)`. So it goes 40, 41, 42...
                        # If I find a solution at R=45, I should stop? No, I want the MINIMAL rank.
                        # So if I find one at 45, I should check 44, 43, 42, 41, 40.
                        # But I am iterating 40, 41, 42...
                        # So if I find one at 40, I can stop immediately.
                        # If I find one at 45, I have already checked 40-44 and failed. So 45 is the best.
                        # So yes, if I find a valid solution, I can break all loops and return it.
                        
                        # However, it's possible that the optimizer got stuck in a local minimum that looks valid but isn't exact?
                        # The verify function checks exactness (within tol).
                        # If it passes, it's a valid decomposition.
                        
                        # So we can break.
                        pass # Signal to break outer loop
                    else:
                        # Maybe it's close but not exact. Can we polish?
                        # Try a few more iterations with tighter tol?
                        pass
                        
            except Exception as e:
                # Optimization failed, try next trial
                continue
        
        if best_U is not None:
            break # Found minimal rank (since we went low to high)

    # If no better solution found, return schoolbook
    if best_U is None:
        result = schoolbook()
        return result
    else:
        # Convert to lists
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
