import numpy as np
import time
import random
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try ranks from 35 down to 31
    for R in range(35, 30, -1):
        if time.time() > deadline:
            break
        
        # Try to find a valid decomposition of rank R
        for attempt in range(10000):
            if time.time() > deadline:
                break
            
            # Generate random U and V with entries in {-1, 0, 1}
            # Sparsity might help
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Compute S[i, k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # U shape: (n*m, R)
            # V shape: (m*p, R)
            # We need to sum over j.
            # Reshape U to (n, m, R)
            # Reshape V to (m, p, R)
            U_reshaped = U.reshape(n, m, R)
            V_reshaped = V.reshape(m, p, R)
            
            # S[i, k, c] = sum_j U_reshaped[i, j, c] * V_reshaped[j, k, c]
            # Use einsum or manual loop
            # S shape: (n, p, R)
            S = np.zeros((n, p, R), dtype=int)
            for i in range(n):
                for k in range(p):
                    for c in range(R):
                        S[i, k, c] = np.sum(U_reshaped[i, :, c] * V_reshaped[:, k, c])
            
            # Check gcd condition for each (i, k)
            valid = True
            for i in range(n):
                for k in range(p):
                    row = S[i, k, :]
                    # Calculate gcd of the row
                    # If all zeros, invalid
                    if np.all(row == 0):
                        valid = False
                        break
                    g = reduce(gcd, map(abs, row))
                    if g == 0:
                        valid = False
                        break
                    if 2 % g != 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Construct W
            # We need to find x[i, k, c] such that sum_c S[i, k, c] x[i, k, c] = 2
            # x are integers. Then W[i, k, c] = x[i, k, c] / 2
            W = np.zeros((p * n, R), dtype=float)
            
            # Simple solution: find one non-zero element in each row
            # Since gcd divides 2, there exists a combination.
            # We can use a simple greedy or extended Euclidean.
            # For simplicity, if any S[i,k,c] is +/-1, we can set x = +/-2.
            # If any is +/-2, we can set x = +/-1.
            # Otherwise, we need a combination.
            
            for i in range(n):
                for k in range(p):
                    row = S[i, k, :]
                    # Find a simple solution
                    # Check if any element is +/-1
                    found = False
                    for c in range(R):
                        if row[c] == 1:
                            W[k * n + i, c] = 1.0  # x=2 -> w=1
                            found = True
                            break
                        elif row[c] == -1:
                            W[k * n + i, c] = -1.0 # x=-2 -> w=-1
                            found = True
                            break
                        elif row[c] == 2:
                            W[k * n + i, c] = 0.5  # x=1 -> w=0.5
                            found = True
                            break
                        elif row[c] == -2:
                            W[k * n + i, c] = -0.5 # x=-1 -> w=-0.5
                            found = True
                            break
                    
                    if not found:
                        # Need to solve general Diophantine
                        # Use a simple method: find two numbers with gcd dividing 2
                        # This is rare with random {-1,0,1} inputs, S entries are small
                        # If not found, skip this attempt
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            # C[i, k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # Should be 1 for all i, k
            C = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(R):
                            C[i, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
            
            if np.allclose(C, np.ones((n, p)), atol=1e-9):
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback to schoolbook
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
