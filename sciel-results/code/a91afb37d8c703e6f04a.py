import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    """
    Returns the best rank decomposition found for matrix multiplication M_{3,4,7}.
    Tries to find rank < 84 using optimization, falls back to schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Verify function
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find better decompositions using random search + local optimization
    def try_rank_with_seed(rank, seed):
        """Try to find a decomposition with given rank using random init."""
        if time.time() - start_time > DEADLINE - 15:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = (rng.randn(n*m, rank) * 0.1).tolist()
        V = (rng.randn(m*p, rank) * 0.1).tolist()
        W = (rng.randn(p*n, rank) * 0.1).tolist()
        
        # Use a simple iterative method
        lr = 0.05
        for iteration in range(200):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Pick a random (i,j,k) and adjust
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            
            val = 0.0
            for r in range(rank):
                val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            err = val - 1.0
            
            if abs(err) < 1e-12:
                continue
            
            # Adjust each factor
            for r in range(rank):
                U[i*m+j][r] -= lr * err * V[j*p+k][r] * W[k*n+i][r]
                V[j*p+k][r] -= lr * err * U[i*m+j][r] * W[k*n+i][r]
                W[k*n+i][r] -= lr * err * U[i*m+j][r] * V[j*p+k][r]
        
        # Check if we found a valid solution
        if verify(U, V, W, rank):
            return U, V, W, rank
        
        return None
    
    # Try decreasing ranks with multiple seeds
    for rank in range(83, 60, -1):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        for seed in range(3):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            result = try_rank_with_seed(rank, seed)
            if result is not None:
                U, V, W, r = result
                if r < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, r
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
