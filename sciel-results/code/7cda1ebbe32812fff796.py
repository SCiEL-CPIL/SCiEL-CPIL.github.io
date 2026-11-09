"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
from scipy.optimize import minimize
import random
import time

def construct():
    n, m, p = DIMENSIONS
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Find minimal rank decomposition
    
    # For a rank-r decomposition:
    # For each (i,j,k): sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: For fixed rank R, fix U and V randomly, then solve for W linearly.
    # Then verify. Use iterative refinement.
    
    random.seed(42)
    
    def schoolbook_decomposition():
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
    
    def verify_decomposition(U, V, W):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                    if abs(val - 1) > 1e-6:
                        return False
        return True
    
    # Try ranks from lower bound up
    # Lower bound is at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But for matrix multiplication, better bounds exist
    # Let's try from 30 to 72
    
    for R in range(30, 73):
        found = False
        
        # Multiple attempts with different random seeds
        for attempt in range(20):
            random.seed(42 + attempt * 1000)
            
            # Initialize U and V with random values from {-1, 0, 1}
            # This creates a structured initialization
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            
            # Use a sparse random initialization
            for idx in range(n*m):
                # Each row of U has a few non-zero entries
                num_nonzero = random.randint(1, min(3, R))
                cols = random.sample(range(R), num_nonzero)
                for c in cols:
                    U[idx, c] = random.choice([-1, 1])
            
            for idx in range(m*p):
                num_nonzero = random.randint(1, min(3, R))
                cols = random.sample(range(R), num_nonzero)
                for c in cols:
                    V[idx, c] = random.choice([-1, 1])
            
            # Now solve for W
            # For each (i,j,k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # 
            # This is a system of n*m*p = 72 equations in p*n = 18*R unknowns
            # (W has shape (p*n, R))
            #
            # Actually, the equations are coupled. Let's set up the linear system.
            # 
            # Let's index W entries as W[k*n+i, r] for k in range(p), i in range(n), r in range(R)
            # 
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # This is linear in W. Let's build the system.
            
            # Number of unknowns: p*n*R
            num_unknowns = p * n * R
            num_equations = n * m * p
            
            A = np.zeros((num_equations, num_unknowns))
            b = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            # W[k*n+i, r] is at index (k*n+i)*R + r
                            w_idx = (k*n + i) * R + r
                            A[eq_idx, w_idx] = U[i*m + j, r] * V[j*p + k, r]
                        eq_idx += 1
            
            # Solve the linear system
            # Use least squares since the system may be underdetermined
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                if residuals.size > 0:
                    if residuals[0] > 1e-6:
                        continue
                
                # Reshape W
                W = W_flat.reshape((p*n, R))
                
                # Verify the decomposition
                if verify_decomposition(U, V, W):
                    # Check if entries are close to integers or half-integers
                    # and clean them up
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    
                    # Re-verify after cleaning
                    if verify_decomposition(U_clean, V_clean, W_clean):
                        # Convert to lists
                        U_list = [[float(x) for x in row] for row in U_clean]
                        V_list = [[float(x) for x in row] for row in V_clean]
                        W_list = [[float(x) for x in row] for row in W_clean]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Use the uncleaned version if it works
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        return {"U": U_list, "V": V_list, "W": W_list}
            except np.linalg.LinAlgError:
                continue
        
        # If we found a solution for this R, return it
        # Otherwise, continue to next R
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
