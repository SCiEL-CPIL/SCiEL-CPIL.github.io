"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    def make_schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Build the tensor from U, V, W
        # T[i*m+j, j*p+k, k*n+i] should equal 1 if this is the MM tensor
        # Actually, we need to check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(i,j,k) contribution
        # The MM tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
        # T should be 1 when the indices are consistent (same j in a and b, same i in a and c, same k in b and c)
        # More precisely: T[(i,j),(j,k),(k,i)] = 1 for all i,j,k
        # All other entries = 0
        
        # Check all (a,b,c) triples
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    
                    # Compute sum_c U[a,c]*V[b,c]*W[c,c]
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    
                    # Expected value
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        expected = 1
                    else:
                        expected = 0
                    
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a valid decomposition at given rank."""
        # Initialize with random values
        np.random.seed(42 + rank)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Try to refine using a simple iterative approach
        # This is a non-convex problem, so we'll use a heuristic
        
        # Convert to lists for verification
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try various ranks from low to high
    # Start with a reasonable estimate
    # For 4x4 x 4x9, let's try ranks from 70 up to schoolbook
    best_result = None
    best_rank = schoolbook_rank
    
    # First, get the schoolbook as a safe fallback
    schoolbook = make_schoolbook()
    best_result = schoolbook
    
    # Try to find lower rank decompositions
    # We'll use a simple approach: try random initializations at various ranks
    # and check if they're valid
    
    # Actually, random won't work for exact decomposition. We need a smarter approach.
    # Let's try a constructive approach based on known results.
    
    # For now, return the schoolbook decomposition
    # A proper search would require more sophisticated methods
    
    return best_result
# EVOLVE-BLOCK-END
