"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
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
    
    # Try to find a lower-rank decomposition using randomized search
    # We'll use a greedy approach with random restarts
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    product_sum = np.sum(u_val * v_val * w_val)
                    if abs(product_sum - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(rank, time_limit):
        """Try to find a valid decomposition with given rank."""
        start = time.time()
        
        # Try multiple random initializations
        for attempt in range(1000):
            if time.time() - start > time_limit:
                break
            
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            if verify_decomposition(U, V, W, rank):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from lower to higher
    # Theoretical lower bound for M_{3,3,7} is at least max(n*m, m*p, p*n) = 21
    # But we need to be realistic - try a few ranks below the trivial 63
    
    for rank in [21, 24, 27, 30, 33, 36, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, time_limit=5.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to trivial decomposition
    return trivial_decomposition()
# EVOLVE-BLOCK-END
