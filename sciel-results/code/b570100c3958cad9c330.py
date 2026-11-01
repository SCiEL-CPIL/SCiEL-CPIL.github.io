import time
import math
import random
import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    First tries to find a lower-rank decomposition via structured search,
    falls back to schoolbook rank 54.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook decomposition: rank = n*m*p = 54
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, n, m, p):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: use a structured ansatz where each rank-1 term is associated
    # with a subset of (i,j,k) indices, and we try to share terms.
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try ranks from a lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But for matrix multiplication, better bounds exist.
    # For 3x3 * 3x6, let's try various ranks
    
    # Use a greedy/structured approach:
    # Each rank-1 term (u, v, w) contributes to all (i,j,k) where
    # u[i*m+j] * v[j*p+k] * w[k*n+i] != 0
    # We need the sum over all terms to equal 1 for each (i,j,k)
    
    # Try a different approach: parameterize with rational/half-integer values
    # and search for a low-rank solution
    
    # For 3x3 matrix multiplication, the best known rank is 23 (Strassen-like)
    # For 3x3 * 3x6, we can potentially do better than 54
    
    # Let's try a structured search: assume the decomposition has a certain
    # symmetry pattern
    
    # Approach: try rank r from 18 to 53, and for each, try to find
    # a valid decomposition using a simple heuristic
    
    def try_rank(r, n, m, p, time_limit=5):
        """Try to find a rank-r decomposition."""
        t0 = time.time()
        
        # Initialize with random values
        for attempt in range(100):
            if time.time() - t0 > time_limit:
                break
            
            # Use a structured initialization
            U = [[0.0] * r for _ in range(n * m)]
            V = [[0.0] * r for _ in range(m * p)]
            W = [[0.0] * r for _ in range(p * n)]
            
            # Try a simple pattern: assign each term to a specific (i,j,k)
            # but allow sharing
            if attempt == 0:
                # Direct mapping for first min(r, n*m*p) terms
                for c in range(r):
                    idx = c % (n * m * p)
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    j = rem // p
                    k = rem % p
                    U[i * m + j][c] = 1.0
                    V[j * p + k][c] = 1.0
                    W[k * n + i][c] = 1.0
            else:
                # Random initialization with small values
                for i in range(n * m):
                    for c in range(r):
                        U[i][c] = random.choice([-1, 0, 1]) * random.uniform(0, 1)
                for j in range(m * p):
                    for c in range(r):
                        V[j][c] = random.choice([-1, 0, 1]) * random.uniform(0, 1)
                for k in range(p * n):
                    for c in range(r):
                        W[k][c] = random.choice([-1, 0, 1]) * random.uniform(0, 1)
            
            # Check if this is a valid decomposition
            if verify(U, V, W, n, m, p):
                # Convert to integer/half-integer if possible
                U_int = [[int(round(2 * U[i][c]) / 2) for c in range(r)] for i in range(n * m)]
                V_int = [[int(round(2 * V[i][c]) / 2) for c in range(r)] for i in range(m * p)]
                W_int = [[int(round(2 * W[i][c]) / 2) for c in range(r)] for i in range(p * n)]
                if verify(U_int, V_int, W_int, n, m, p):
                    return U_int, V_int, W_int
                return U, V, W
        
        return None
    
    # Try various ranks
    for r in range(18, min(rank_school, 54)):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, n, m, p, time_limit=2)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    return best_result
