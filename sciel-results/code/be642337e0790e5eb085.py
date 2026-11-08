import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute all valid (i, j, k) triples
    triples = list(product(range(n), range(m), range(p)))
    num_triples = len(triples)  # 36 for 3x3x4
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W with given rank correctly decompose the tensor."""
        for i, j, k in triples:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = sum(U[u_idx, r] * V[v_idx, r] * W[w_idx, r] for r in range(rank))
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    def try_random_search(rank, num_trials, seed_base):
        """Try random integer/half-integer initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            np.random.seed(seed_base + trial)
            
            # Use small integers and half-integers
            scale = 2  # Multiply by 2 to work with integers, then divide
            U = np.random.randint(-scale, scale + 1, (n*m, rank)) / scale
            V = np.random.randint(-scale, scale + 1, (m*p, rank)) / scale
            W = np.random.randint(-scale, scale + 1, (p*n, rank)) / scale
            
            if check_decomposition(U, V, W, rank):
                return U, V, W
        return None
    
    def try_structured_search(rank, num_trials, seed_base):
        """Try structured patterns: sparse matrices with specific structures."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            np.random.seed(seed_base + trial)
            
            # Initialize with zeros
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # For each rank column, assign non-zero entries with probability
            sparsity = 0.3 + 0.4 * (trial / num_trials)
            values = [-1.0, -0.5, 0.5, 1.0]
            
            for r in range(rank):
                # Randomly select entries to be non-zero
                for u_idx in range(n*m):
                    if np.random.random() < sparsity:
                        U[u_idx, r] = np.random.choice(values)
                for v_idx in range(m*p):
                    if np.random.random() < sparsity:
                        V[v_idx, r] = np.random.choice(values)
                for w_idx in range(p*n):
                    if np.random.random() < sparsity:
                        W[w_idx, r] = np.random.choice(values)
            
            if check_decomposition(U, V, W, rank):
                return U, V, W
        return None
    
    def try_gradient_descent(rank, max_iter, seed_base):
        """Use gradient descent on real-valued variables, then round to half-integers."""
        np.random.seed(seed_base)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        lr = 0.1
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i, j, k in triples:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                for r in range(rank):
                    current = sum(U[u_idx, rr] * V[v_idx, rr] * W[w_idx, rr] for rr in range(rank))
                    error = current - 1.0
                    
                    grad_U[u_idx, r] += error * V[v_idx, r] * W[w_idx, r]
                    grad_V[v_idx, r] += error * U[u_idx, r] * W[w_idx, r]
                    grad_W[w_idx, r] += error * U[u_idx, r] * V[v_idx, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check if we're close to integer/half-integer
            if it % 100 == 0:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if check_decomposition(U_rounded, V_rounded, W_rounded, rank):
                    return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 15 upwards
    best_result = None
    
    for rank in range(15, 37):
        if time.time() > deadline:
            break
        
        # Method 1: Random search
        result = try_random_search(rank, 100, rank * 1000)
        if result is not None:
            best_result = (rank,) + result
            break
        
        # Method 2: Structured search
        result = try_structured_search(rank, 50, rank * 2000)
        if result is not None:
            best_result = (rank,) + result
            break
        
        # Method 3: Gradient descent
        result = try_gradient_descent(rank, 500, rank * 3000)
        if result is not None:
            best_result = (rank,) + result
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: Schoolbook (rank 36)
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
