"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        n, m, p = DIMENSIONS
        # Check tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check that all other entries are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > 1e-9:
                        return False
        # Spot-check some non-tensor entries should be 0
        # T[a,b,c] = 0 unless (a,b,c) corresponds to (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        # For a random sample of non-matching indices:
        import random
        random.seed(42)
        for _ in range(1000):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if this is a valid tensor index
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
        return True
    
    def try_rank_reduced():
        """Try to construct a rank-reduced decomposition.
        
        Strategy: Partition the 7 columns into:
        - First 3 columns: use a rank-23 decomposition for 3x3x3
        - Remaining 4 columns: use rank-4 per column group
        
        Total target rank: 23 + 16 = 39
        """
        # For the 3x3x3 part, we can use a known rank-23 construction
        # This is based on Strassen-like methods
        
        # Let's try a simpler approach first: use the fact that for rectangular matrices,
        # we can sometimes do better than n*m*p
        
        # For 3x3 x 3x7, one approach:
        # Treat as two parts: 3x3 x 3x3 (rank 23) and 3x3 x 3x4 (rank ?)
        
        # Actually, let's try a direct construction with rank < 63
        
        # Method: Use the identity that for matrix multiplication, we can use
        # the tensor rank which is at most n*m*p, but often less.
        
        # Let's try to construct a rank-39 decomposition explicitly
        
        rank = 39
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Part 1: First 3 columns (k=0,1,2) - use rank-23 for 3x3x3
        # This is complex, so let's use a simpler approach for now
        
        # Alternative: Use a block structure
        # For each output column k, we need to compute (A @ B[:,k])
        # This is a 3x3 matrix multiplication for each k
        
        # Let's use the fact that we can share some computation
        
        # Simple improvement: For the first 3 columns, use rank 9 each = 27
        # For the last 4 columns, use rank 9 each = 36
        # Total: 63 (no improvement)
        
        # Let me try a different approach: use the known result that 
        # R(3,3,3) = 23 (or close to it)
        
        # For now, let's just return schoolbook if we can't find a better one
        # This function is a placeholder for future optimization
        
        return None
    
    # Try to find a better decomposition
    result = try_rank_reduced()
    if result is None:
        result = schoolbook()
    return result
# EVOLVE-BLOCK-END
