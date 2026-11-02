import time
import random
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 115

    # Rank-23 decomposition for 3x3x3 matrix multiplication
    # Based on known constructions (Landsberg, etc.)
    # U: (9, 23), V: (9, 23), W: (9, 23)
    
    # Known rank-23 algorithm for 3x3x3
    # Using a construction from the literature
    U23 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
    ]
    V23 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
    ]
    W23 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
    ]
    
    # This is just the trivial 27-term compressed to 23 - but that's wrong.
    # Let me use a proper approach: start with schoolbook and use known reductions.
    
    # Actually, let me use a verified approach: 
    # For 3x3x4, I'll use a rank-26 construction based on known results.
    
    # Strategy: Use the fact that 3x3x3 can be done in rank 23 (known),
    # and add 3 more terms for the 4th column, total rank 26.
    
    # But I need explicit coefficients. Let me use a different approach:
    # Start with schoolbook rank 36 and try to find a lower rank via optimization.
    
    # Better approach: Use a known valid decomposition.
    # For 3x3x4, the known upper bound is 26 (from 3x3x3 rank 23 + 3).
    
    # Let me construct a valid rank-26 decomposition:
    # First 23 terms handle 3x3x3, last 3 terms handle the 4th column.
    
    # For the 3x3x3 part, I'll use a simplified valid construction.
    # Actually, let me just use the schoolbook for 3x3x3 (rank 27) + 3 for 4th col = rank 30.
    # Then try to reduce.
    
    # Even better: use a known rank-23 construction.
    # Here's a valid rank-23 for 3x3x3 (simplified from literature):
    
    # Let me use a different strategy: try to find a valid decomposition
    # using gradient-based optimization starting from random initializations.
    
    def _verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def _error(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    err += (s - 1.0) ** 2
        return err

    def _gradient_descent(U, V, W, lr=0.01, max_iters=1000, time_limit=None):
        """Simple gradient descent on the sum of squared errors."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            
            # Compute gradients numerically
            U_grad = [[0.0]*rank for _ in range(n*m)]
            V_grad = [[0.0]*rank for _ in range(m*p)]
            W_grad = [[0.0]*rank for _ in range(p*n)]
            
            eps = 1e-5
            base_err = _error(U, V, W)
            
            # Only compute gradients for a random subset to save time
            if it % 10 == 0 and time.time() > deadline - 30:
                break
            
            # Sample a few entries
            for _ in range(3):
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, rank-1)
                    old = U[row][col]
                    U[row][col] = old + eps
                    err_plus = _error(U, V, W)
                    U[row][col] = old - eps
                    err_minus = _error(U, V, W)
                    U[row][col] = old
                    U_grad[row][col] = (err_plus - err_minus) / (2*eps)
                elif mat == 1:
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, rank-1)
                    old = V[row][col]
                    V[row][col] = old + eps
                    err_plus = _error(U, V, W)
                    V[row][col] = old - eps
                    err_minus = _error(U, V, W)
                    V[row][col] = old
                    V_grad[row][col] = (err_plus - err_minus) / (2*eps)
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, rank-1)
                    old = W[row][col]
                    W[row][col] = old + eps
                    err_plus = _error(U, V, W)
                    W[row][col] = old - eps
                    err_minus = _error(U, V, W)
                    W[row][col] = old
                    W_grad[row][col] = (err_plus - err_minus) / (2*eps)
            
            # Update
            for i in range(n*m):
                for r in range(rank):
                    U[i][r] -= lr * U_grad[i][r]
            for i in range(m*p):
                for r in range(rank):
                    V[i][r] -= lr * V_grad[i][r]
            for i in range(p*n):
                for r in range(rank):
                    W[i][r] -= lr * W_grad[i][r]
        
        return U, V, W, _error(U, V, W)

    def _round_and_verify(U, V, W):
        U_r = [[round(x * 2) / 2.0 for x in row] for row in U]
        V_r = [[round(x * 2) / 2.0 for x in row] for row in V]
        W_r = [[round(x * 2) / 2.0 for x in row] for row in W]
        if _verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Schoolbook rank-36 as fallback
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][c] = 1
                V_school[j*p+k][c] = 1
                W_school[k*n+i][c] = 1
                c += 1

    best_rank = n * m * p
    best_factors = (U_school, V_school, W_school)

    # Try ranks from 26 upward
    for target_rank in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
        found = False
        trials = 0
        max_trials = 30 if target_rank <= 28 else 15
        while time.time() < deadline - 10 and trials < max_trials:
            trials += 1
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W):
                best_rank = target_rank
                best_factors = (U, V, W)
                found = True
                break
            
            U_ls, V_ls, W_ls, err = _gradient_descent(U, V, W, lr=0.005, max_iters=300, time_limit=deadline - 10)
            if err < 1e-9:
                rounded = _round_and_verify(U_ls, V_ls, W_ls)
                if rounded:
                    best_rank = target_rank
                    best_factors = rounded
                    found = True
                    break
        
        if found:
            break

    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}
