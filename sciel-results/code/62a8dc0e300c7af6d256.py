"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: trivial decomposition with rank = n*m*p
    def trivial_decomposition():
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
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    product_sum = np.sum(u_val * v_val * w_val)
                    if abs(product_sum - 1.0) > 1e-6:
                        return False
        return True
    
    def try_rank_structured(rank, time_limit):
        """Try to find a valid decomposition with given rank using structured approach.
        
        Strategy: Use the asymmetry p=7. For each of the n=3 rows of the first matrix,
        we can share structure. We'll try to find U, V, W with small integer entries.
        """
        start = time.time()
        np.random.seed(int(start * 1000) % (2**31))
        
        # Try various patterns
        for attempt in range(2000):
            if time.time() - start > time_limit:
                break
            
            # Generate random matrices with small integer entries
            # Use a mix of 0, 1, -1, 2, -2
            U = np.random.choice([-2, -1, 0, 1, 2], size=(n * m, rank))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(m * p, rank))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(p * n, rank))
            
            if verify_decomposition(U, V, W, rank):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    def try_rank_with_polish(rank, time_limit):
        """Try random initialization then polish with local search."""
        start = time.time()
        np.random.seed(int(start * 1000) % (2**31))
        
        best_score = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(500):
            if time.time() - start > time_limit:
                break
            
            # Initialize
            U = np.random.choice([-2, -1, 0, 1, 2], size=(n * m, rank))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(m * p, rank))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(p * n, rank))
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                        error += (np.sum(prod) - 1.0) ** 2
            
            if error < best_score:
                best_score = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_score < 1e-10:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        # If we found a good candidate, try local search
        if best_U is not None and best_score < 100:
            # Local search: try changing individual entries
            for _ in range(200):
                if time.time() - start > time_limit:
                    break
                
                # Try changing a random entry in U, V, or W
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row, col = np.random.randint(0, n * m), np.random.randint(0, rank)
                    old_val = best_U[row, col]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val:
                            continue
                        best_U[row, col] = new_val
                        error = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    prod = best_U[i * m + j, :] * best_V[j * p + k, :] * best_W[k * n + i, :]
                                    error += (np.sum(prod) - 1.0) ** 2
                        if error < best_score:
                            best_score = error
                        else:
                            best_U[row, col] = old_val
                elif mat == 'V':
                    row, col = np.random.randint(0, m * p), np.random.randint(0, rank)
                    old_val = best_V[row, col]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val:
                            continue
                        best_V[row, col] = new_val
                        error = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    prod = best_U[i * m + j, :] * best_V[j * p + k, :] * best_W[k * n + i, :]
                                    error += (np.sum(prod) - 1.0) ** 2
                        if error < best_score:
                            best_score = error
                        else:
                            best_V[row, col] = old_val
                else:  # W
                    row, col = np.random.randint(0, p * n), np.random.randint(0, rank)
                    old_val = best_W[row, col]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val:
                            continue
                        best_W[row, col] = new_val
                        error = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    prod = best_U[i * m + j, :] * best_V[j * p + k, :] * best_W[k * n + i, :]
                                    error += (np.sum(prod) - 1.0) ** 2
                        if error < best_score:
                            best_score = error
                        else:
                            best_W[row, col] = old_val
                
                if best_score < 1e-10:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        if best_U is not None and best_score < 1e-10:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try ranks from lower to higher
    # Theoretical lower bound for M_{3,3,7} is at least max(n*m, m*p, p*n) = 21
    
    for rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 63]:
        if time.time() > deadline - 5:
            break
        
        # Try structured random search first
        result = try_rank_structured(rank, time_limit=3.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try with polish
        result = try_rank_with_polish(rank, time_limit=5.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to trivial decomposition
    return trivial_decomposition()
# EVOLVE-BLOCK-END
