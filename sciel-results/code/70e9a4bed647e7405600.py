"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, n, m, p):
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Structural search: try to find a decomposition with rank < 60
    # using a pattern-based approach
    
    # Key structural insight: the tensor T[i,j,k] = 1 for all (i,j,k)
    # We can try decompositions where the rank-1 terms have a "block" structure
    
    # Try a specific construction: use the fact that m=4 and p=5 are close
    # to exploit near-periodicity
    
    # Approach: try ranks R and for each, try to find U,V,W with entries in {0,1}
    # or {0, 1/2, 1} that satisfy the constraint
    
    # For efficiency, use a constraint-based search
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a structured search for ranks 45-59
    # Use the observation that for matrix multiplication, the tensor
    # has a specific algebraic structure that may allow for a lower-rank
    # decomposition with a specific pattern
    
    # Try pattern: U[i*m+j, r] depends on (i, j, r) in a modular way
    # Specifically, try: U[i*m+j, r] = 1 if j ≡ (i + r) mod m, else 0
    # and similarly for V and W, then check if the product sums to 1
    
    # This is a specific ansatz that may or may not work, but it's a
    # structural change from the schoolbook approach
    
    for R in range(45, 60):
        if time.time() > deadline - 5:
            break
        
        # Try the modular pattern
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Pattern 1: U[i*m+j, r] = 1 if j % R == r % m (cyclic in j)
        for i in range(n):
            for j in range(m):
                for r in range(R):
                    if j == r % m:
                        U[i * m + j][r] = 1
        
        # Pattern 2: V[j*p+k, r] = 1 if k % R == r % p (cyclic in k)
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    if k == r % p:
                        V[j * p + k][r] = 1
        
        # Pattern 3: W[k*n+i, r] = 1 if i % R == r % n (cyclic in i)
        for k in range(p):
            for i in range(n):
                for r in range(R):
                    if i == r % n:
                        W[k * n + i][r] = 1
        
        if verify(U, V, W, n, m, p):
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break
    
    # If the modular pattern didn't work, try another structural pattern
    if best_rank == rank_schoolbook and time.time() < deadline - 10:
        # Try a different pattern: based on the sum i+j+k mod R
        for R in range(48, 60):
            if time.time() > deadline - 5:
                break
            
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            
            # Pattern: U[i*m+j, r] = 1 if (i + 2*j) % R == r
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        if (i + 2 * j) % R == r:
                            U[i * m + j][r] = 1
            
            # Pattern: V[j*p+k, r] = 1 if (3*j + k) % R == r
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        if (3 * j + k) % R == r:
                            V[j * p + k][r] = 1
            
            # Pattern: W[k*n+i, r] = 1 if (4*k + 5*i) % R == r
            for k in range(p):
                for i in range(n):
                    for r in range(R):
                        if (4 * k + 5 * i) % R == r:
                            W[k * n + i][r] = 1
            
            if verify(U, V, W, n, m, p):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
    
    # Return the best result found
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
