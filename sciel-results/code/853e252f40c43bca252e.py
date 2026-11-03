"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import random

def construct():
    n, m, p = 2, 3, 10
    num_constraints = n * m * p  # 60
    
    # Try ranks from 59 down to 50
    for target_rank in range(59, 49, -1):
        sol = _find_solution(n, m, p, target_rank)
        if sol is not None:
            return sol
    
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

def _find_solution(n, m, p, rank):
    """Try to find a rank-r solution using randomized approach."""
    for attempt in range(200):
        random.seed(attempt)
        U = [[Fraction(random.randint(-2, 2)) for _ in range(rank)] for _ in range(n * m)]
        V = [[Fraction(random.randint(-2, 2)) for _ in range(rank)] for _ in range(m * p)]
        
        W = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
        
        success = True
        for k in range(p):
            for i in range(n):
                # For this (k,i), we have m equations:
                # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let B_j[c] = U[i*m+j,c] * V[j*p+k,c]
                # Then: sum_c B_j[c] * W[k*n+i,c] = 1 for each j
                # This is a system of m linear equations in r unknowns (W[k*n+i,:])
                
                # Build the system: B_j dot W_row = 1 for j=0..m-1
                # If m < r, we can find a solution if the system is consistent
                # Use a simple approach: set W_row to satisfy the equations
                
                # For simplicity, try to find W_row such that B_j dot W_row = 1 for all j
                # This requires that all B_j are proportional (if m > 1)
                # Or we can use least squares and check if it's exact
                
                # Simple approach: if all B_j are the same, set W_row = B_j / (B_j·B_j)
                # Otherwise, we need a more careful approach
                
                # Let's try: W_row = sum_j alpha_j * B_j, where alpha_j are chosen so that B_i dot W_row = 1
                # This gives a system: sum_j alpha_j * (B_i dot B_j) = 1 for all i
                # This is a Gram matrix system
                
                B = []
                for j in range(m):
                    B_j = [U[i * m + j][c] * V[j * p + k][c] for c in range(rank)]
                    B.append(B_j)
                
                # Build Gram matrix G[i][j] = B_i dot B_j
                G = [[sum(B[i][c] * B[j][c] for c in range(rank)) for j in range(m)] for i in range(m)]
                
                # Solve G * alpha = [1, 1, ..., 1]
                # Then W_row = sum_j alpha_j * B_j
                
                # For small m, solve directly
                try:
                    alpha = _solve_linear_system(G, [1] * m)
                    if alpha is None:
                        success = False
                        break
                    W_row = [sum(alpha[j] * B[j][c] for j in range(m)) for c in range(rank)]
                    W[k * n + i] = W_row
                except:
                    success = False
                    break
        
        if not success:
            continue
        
        if _verify(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _solve_linear_system(A, b):
    """Solve A*x = b using Gaussian elimination with fractions."""
    n = len(A)
    # Augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]
    
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return None
        
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        
        # Eliminate
        for row in range(n):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
    
    x = [aug[i][n] / aug[i][i] for i in range(n)]
    return x

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U[0])):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
