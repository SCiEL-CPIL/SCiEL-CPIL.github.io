"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or at least valid) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Tries to find a lower-rank decomposition using random initialization
    and solving the nonlinear system, falling back to schoolbook if unsuccessful.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(nm)]
    V_school = [[0] * rank_schoolbook for _ in range(mp)]
    W_school = [[0] * rank_schoolbook for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from 50 to 83
    best_result = None
    for rank in range(50, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        np.random.seed(rank)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Try to solve the system using gradient descent or similar
        # This is a nonlinear system, so use a simple approach
        
        # For now, just check if we can find a solution with this rank
        # by using a simple iterative method
        
        # Actually, let's use a different approach: try to find a sparse solution
        # Initialize with zeros and fill in entries greedily
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Greedy: for each (i,j,k), try to assign a term
        assigned = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Try to find an existing term that covers this
                    found = False
                    for r in range(rank):
                        if U[i*m+j, r] != 0 and V[j*p+k, r] != 0 and W[k*n+i, r] != 0:
                            # Check if this term can be adjusted
                            # This is complex, skip for now
                            found = True
                            break
                    if not found:
                        # Assign a new term
                        for r in range(rank):
                            if U[i*m+j, r] == 0 and V[j*p+k, r] == 0 and W[k*n+i, r] == 0:
                                U[i*m+j, r] = 1
                                V[j*p+k, r] = 1
                                W[k*n+i, r] = 1
                                found = True
                                break
                        if not found:
                            break
                    if not found:
                        break
                if not found:
                    break
            if not found:
                break
        
        # Verify the solution
        if found:
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to list of lists with integer/half-integer entries
                U_list = [[int(U[i,j]) if abs(U[i,j] - round(U[i,j])) < 1e-10 else float(U[i,j]) for j in range(rank)] for i in range(nm)]
                V_list = [[int(V[i,j]) if abs(V[i,j] - round(V[i,j])) < 1e-10 else float(V[i,j]) for j in range(rank)] for i in range(mp)]
                W_list = [[int(W[i,j]) if abs(W[i,j] - round(W[i,j])) < 1e-10 else float(W[i,j]) for j in range(rank)] for i in range(pn)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
