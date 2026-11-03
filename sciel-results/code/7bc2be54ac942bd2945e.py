"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 150.0
    n, m, p = 3, 3, 5
    
    # Return a valid schoolbook decomposition of rank 45
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
    
    # Verify a decomposition by checking that the tensor matches
    def verify(U, V, W, n, m, p, tol=1e-9):
        """
        Check if U, V, W form a valid tensor decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        should equal 1 if i,j,k are valid indices, else 0.
        """
        rank = len(U[0])
        # Build the tensor from the decomposition
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U[i * m + j][r]
                    if u_val != 0:
                        for k in range(p):
                            v_val = V[j * p + k][r]
                            if v_val != 0:
                                w_val = W[k * n + i][r]
                                if w_val != 0:
                                    T[i, j, k] += u_val * v_val * w_val
        
        # Check against the identity tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i, j, k] - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using random search
    def random_search(max_rank=44, iterations=10000):
        """
        Try to find a decomposition with rank < 45 using random search.
        This is unlikely to succeed but we try.
        """
        for rank in range(23, max_rank + 1):
            for _ in range(iterations // (max_rank - 22)):
                if time.time() > deadline - 10:
                    return None
                
                # Random integer entries in {-1, 0, 1}
                U = np.random.randint(-1, 2, size=(n * m, rank))
                V = np.random.randint(-1, 2, size=(m * p, rank))
                W = np.random.randint(-1, 2, size=(p * n, rank))
                
                if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Try structured constructions
    def strassen_3x3x3():
        """
        Attempt a rank-23 Strassen-like decomposition for 3x3x3.
        This is complex; we'll use a known construction if available.
        """
        # For now, this is a placeholder. A proper rank-23 construction
        # for 3x3x3 exists but is non-trivial to implement from scratch.
        # We'll skip this for now and rely on other methods.
        return None
    
    # Try to combine a 3x3x3 decomposition with additional terms for the 2 extra columns
    def hybrid_construction():
        """
        Try to build a decomposition by:
        1. Using a low-rank decomposition for the 3x3x3 part
        2. Adding terms for the remaining 2 columns
        """
        # This is a heuristic approach. We'll try various combinations.
        # For now, we'll just return None if we can't find something better.
        return None
    
    # Main logic: try to find a better decomposition
    # First, check if we have time
    elapsed = time.time() - start_time
    if elapsed < 120:
        # Try random search
        result = random_search(max_rank=44, iterations=5000)
        if result is not None:
            return result
        
    # If no better decomposition found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
