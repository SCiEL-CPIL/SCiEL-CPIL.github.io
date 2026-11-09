import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds limit
    
    # Try to find a low rank decomposition
    # Known ranks for small matrices suggest R(M_{2,5,6}) is likely 14-19
    # We try ranks from 14 up to 20
    for r in range(14, 21):
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts for this rank
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize U, V, W with random half-integers
            # Scale: values in [-2, 2]
            U = np.random.randint(-4, 5, size=(n*m, r)) * 0.5
            V = np.random.randint(-4, 5, size=(m*p, r)) * 0.5
            W = np.random.randint(-4, 5, size=(p*n, r)) * 0.5
            
            # Check if this decomposition works
            # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
            # And for other indices? No, the tensor is only defined for the multiplication pattern.
            # Wait, the tensor T is the identity map on the multiplication?
            # The condition is:
            # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists of floats (integers or half-integers)
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we didn't find one for this rank, try next
        # Note: Random search is unlikely to find a rank 14 decomposition quickly.
        # We might need a more structured search or known decomposition.
        
    # If no low rank found, return schoolbook
    rank = rank_schoolbook
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
