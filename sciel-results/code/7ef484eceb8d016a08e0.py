"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Find minimal rank decomposition
    
    # Try ranks from low to high
    # For each rank, try to find U, V, W with small rational entries
    
    # Known: schoolbook is 72. Let's try to find something better.
    # Use a systematic approach: try rank r, set up equations
    
    # The tensor has n*m*p = 72 non-zero entries (one for each (i,j,k) triple)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # For a rank-r decomposition:
    # For each (i,j,k): sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 72 quadratic equations in 3*R variables.
    
    # Strategy: Try small R and use random initialization + local search
    # with rational rounding
    
    # Let's try a different approach: exploit the structure
    # For matrix multiplication, there are known lower bounds and constructions.
    
    # For 3x4 x 4x6, let's try to find a decomposition by search.
    
    # Start with schoolbook and try to reduce rank
    # Or try random initializations
    
    import random
    random.seed(42)
    
    best_rank = None
    best_solution = None
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actual lower bound for matrix mult is higher.
    
    for R in range(30, 73):
        # Try to find a solution of rank R
        # Use random initialization and try to solve
        
        # Initialize with random small rationals
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Random initialization
        for _ in range(50):
            # Random init with values in {-1, 0, 1, 0.5, -0.5}
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.array([[random.choice(vals) for _ in range(R)] for _ in range(n*m)])
            V = np.array([[random.choice(vals) for _ in range(R)] for _ in range(m*p)])
            W = np.array([[random.choice(vals) for _ in range(R)] for _ in range(p*n)])
            
            # Check if this satisfies the tensor equations
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = U[i*m + j]
                        vj = V[j*p + k]
                        wk = W[k*n + i]
                        val = np.dot(ui * vj, wk)
                        if abs(val - 1) > 1e-9:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            if success:
                best_rank = R
                best_solution = (U, V, W)
                break
        
        if best_solution:
            break
    
    if best_solution is None:
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
    
    U, V, W = best_solution
    # Convert to lists of integers/half-integers
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
