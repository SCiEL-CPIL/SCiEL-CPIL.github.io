"""Program to find a rank-minimal exact tensor decomposition for 2x3 by 3x4 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying 2x3 by 3x4 matrix."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Try to find a decomposition with rank R
    # Start with small R and increase
    
    def try_find_decomposition(R, max_iterations=10000):
        """Try to find a rank-R decomposition with integer/half-integer entries."""
        # Use randomized search with small integer/half-integer values
        # Values can be: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        
        # For efficiency, work with doubled integers (multiply all by 2)
        # So values are: -4, -3, -2, -1, 0, 1, 2, 3, 4 (doubled)
        
        # The condition is:
        # sum_r (U[i*m+j, r] / 2) * (V[j*p+k, r] / 2) * (W[k*n+i, r] / 2) = 1
        # => sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 8
        
        # Let's try different search strategies
        
        # Strategy 1: Random search with small integer values
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Generate random matrices with small integer values
            # Use values from {-2, -1, 0, 1, 2}
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(nm)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(mp)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(pn)]
            
            # Check if this satisfies the tensor decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if s != 8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert back to half-integers by dividing by 2
                U_out = [[U[i][r] / 2 for r in range(R)] for i in range(nm)]
                V_out = [[V[i][r] / 2 for r in range(R)] for i in range(mp)]
                W_out = [[W[i][r] / 2 for r in range(R)] for i in range(pn)]
                return {'U': U_out, 'V': V_out, 'W': W_out}
        
        return None
    
    # Try ranks from 12 upward (theoretical minimum for 2x3x4 is around 12)
    # Actually, the minimum rank for matrix multiplication of 2x3 by 3x4 is known to be 12
    for R in range(12, 25):
        if time.time() > deadline - 5:
            break
        
        result = try_find_decomposition(R, max_iterations=5000)
        if result is not None:
            return result
    
    # If we couldn't find a small rank decomposition, return the schoolbook rank-24 decomposition
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

# EVOLVE-BLOCK-END
