"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Try to find a low-rank decomposition
    for r in range(40, 97):
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition with rank r
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            
            # Initialize U and V with random small integers
            U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
            
            # Solve for W
            # For each (i,k), we have a linear system in W[k*n+i, c] for c=0..r-1
            # The equation is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
            
            # This gives m equations for each (i,k), with r unknowns
            # If m <= r, we can solve this
            
            W = [[0] * r for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the linear system: A * x = b
                    # where A[j, c] = U[i*m+j, c] * V[j*p+k, c], b[j] = 1
                    A = [[U[i * m + j][c] * V[j * p + k][c] for c in range(r)] for j in range(m)]
                    b = [1] * m
                    
                    # Solve A * x = b
                    # Use Gaussian elimination
                    try:
                        # Convert to Fraction for exact arithmetic
                        A_frac = [[Fraction(A[j][c]) for c in range(r)] for j in range(m)]
                        b_frac = [Fraction(b[j]) for j in range(m)]
                        
                        # Gaussian elimination
                        n_eq = m
                        n_unknowns = r
                        augmented = [A_frac[j] + [b_frac[j]] for j in range(n_eq)]
                        
                        # Perform elimination
                        for col in range(n_unknowns):
                            # Find pivot
                            pivot = -1
                            for row in range(col, n_eq):
                                if augmented[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                valid = False
                                break
                            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
                            
                            # Eliminate
                            for row in range(n_eq):
                                if row != col and augmented[row][col] != 0:
                                    factor = augmented[row][col] / augmented[col][col]
                                    for c in range(col, n_unknowns + 1):
                                        augmented[row][c] -= factor * augmented[col][c]
                        
                        # Check for consistency
                        for row in range(n_eq):
                            if all(augmented[row][c] == 0 for c in range(n_unknowns)) and augmented[row][n_unknowns] != 0:
                                valid = False
                                break
                        
                        if not valid:
                            break
                        
                        # Extract solution
                        x = [Fraction(0)] * n_unknowns
                        for row in range(n_eq):
                            if augmented[row][col] != 0:
                                x[col] = augmented[row][n_unknowns] / augmented[row][col]
                        
                        # This doesn't work directly; need to back-substitute
                        # Let me use a different approach
                        
                        valid = False
                        break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                break
        
        if valid:
            # Check the decomposition
            final_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if s != 1:
                            final_valid = False
                            break
                    if not final_valid:
                        break
                if not final_valid:
                    break
                if not final_valid:
                    break
            
            if final_valid:
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
