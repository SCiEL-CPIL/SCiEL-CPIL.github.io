"""Matrix multiplication 2x3x4: rank-minimal tensor decomposition."""
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = Fraction(1)
                    V[j * p + k][column] = Fraction(1)
                    W[k * n + i][column] = Fraction(1)
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # For each rank r from 19 to 23, try to find a valid decomposition
    
    # The tensor equation: for each (i,j,k) in range(n) x range(m) x range(p),
    # sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 24 equations
    
    # For a given rank r, we have (n*m + m*p + p*n) * r = (6+12+8)*r = 26r variables
    
    # We can fix the supports (which entries are nonzero) and solve for the values
    
    # For efficiency, try binary supports first (entries are 0 or 1)
    # Then the equation becomes: for each (i,j,k), the number of columns c with
    # U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 must equal 1
    
    # This is a combinatorial design problem
    
    # For rank r, we need to assign to each column c a "pattern" of which (i,j), (j,k), (k,i) it covers
    
    # Let's try to find such a design for r=19,20,21,22,23
    
    # For each r, try random assignments and check if they work
    
    import random
    random.seed(42)
    
    for r in [19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        # Try multiple random attempts
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random binary support pattern
            U_support = [[random.random() < 0.3 for _ in range(r)] for _ in range(n * m)]
            V_support = [[random.random() < 0.3 for _ in range(r)] for _ in range(m * p)]
            W_support = [[random.random() < 0.3 for _ in range(r)] for _ in range(p * n)]
            
            # Check if this support pattern works
            # For each (i,j,k), count columns c with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        count = 0
                        for c in range(r):
                            if U_support[i * m + j][c] and V_support[j * p + k][c] and W_support[k * n + i][c]:
                                count += 1
                        if count != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Found a valid binary support pattern!
                U = [[Fraction(1) if U_support[i * m + j][c] else Fraction(0) for c in range(r)] for i in range(n) for j in range(m)]
                V = [[Fraction(1) if V_support[j * p + k][c] else Fraction(0) for c in range(r)] for j in range(m) for k in range(p)]
                W = [[Fraction(1) if W_support[k * n + i][c] else Fraction(0) for c in range(r)] for k in range(p) for i in range(n)]
                return {"U": U, "V": V, "W": W}
    
    return schoolbook()

# EVOLVE-BLOCK-END
