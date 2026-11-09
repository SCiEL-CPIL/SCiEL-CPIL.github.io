"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave safety margin
    
    # Schoolbook as fallback
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
    
    # Greedy approach: try to find low-rank decomposition
    # T[i,j,k] = 1 for all i,j,k
    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def try_greedy(max_rank, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Track residual for each (i,j,k)
        residual = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[(i,j,k)] = 1.0
        
        U = [[0.0] * max_rank for _ in range(n * m)]
        V = [[0.0] * max_rank for _ in range(m * p)]
        W = [[0.0] * max_rank for _ in range(p * n)]
        
        for r in range(max_rank):
            if time.time() > deadline - 10:
                break
            
            # Find best (u, v, w) that minimizes sum of squared residuals
            # Try random vectors and pick the best
            best_score = float('inf')
            best_u, best_v, best_w = None, None, None
            
            # Try a few random candidates
            for attempt in range(50):
                # Generate random sparse-ish vectors
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                # Set some random entries to 1 or -1
                for idx in range(n * m):
                    if random.random() < 0.3:
                        u[idx] = random.choice([1, -1, 0.5, -0.5])
                for idx in range(m * p):
                    if random.random() < 0.3:
                        v[idx] = random.choice([1, -1, 0.5, -0.5])
                for idx in range(p * n):
                    if random.random() < 0.3:
                        w[idx] = random.choice([1, -1, 0.5, -0.5])
                
                # Compute the contribution of this term
                contribution = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            if val != 0:
                                contribution[(i,j,k)] = val
                
                # Compute new residual sum of squares
                score = 0.0
                for (i,j,k), res in residual.items():
                    new_res = res - contribution.get((i,j,k), 0)
                    score += new_res * new_res
                
                if score < best_score:
                    best_score = score
                    best_u, best_v, best_w = u, v, w
            
            if best_u is None:
                break
            
            # Apply the best term
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = best_u[i*m+j]
            for j in range(m):
                for k in range(p):
                    V[j*p+k][r] = best_v[j*p+k]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r] = best_w[k*n+i]
            
            # Update residual
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
                        residual[(i,j,k)] -= val
            
            # Check if all residuals are close to zero
            max_res = max(abs(v) for v in residual.values())
            if max_res < 1e-9:
                # Found exact decomposition
                # Convert to integer/half-integer if possible
                U_int = [[round(val * 2) / 2 for val in row] for row in U]
                V_int = [[round(val * 2) / 2 for val in row] for row in V]
                W_int = [[round(val * 2) / 2 for val in row] for row in W]
                return {"U": U_int, "V": V_int, "W": W_int}
        
        return None
    
    # Try decreasing ranks
    for rank in range(100, 144):
        if time.time() > deadline - 20:
            break
        result = try_greedy(rank, seed=42)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
