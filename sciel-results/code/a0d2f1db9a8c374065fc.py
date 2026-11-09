import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def verify(U, V, W):
        """Check if the decomposition is exact using exact arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook as fallback
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The key insight: for fixed U and V, W is determined by linear systems
    # For each (i,k), we solve: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j=0..m-1
    # This is a 3xR linear system for W[k*n+i, :]
    
    # Values to try for U and V entries (integer/half-integer)
    values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    # Try ranks from 29 down
    for target_rank in range(best_rank - 1, 14, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_trials = 50000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Randomly initialize U and V
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            
            # Solve for W
            W = [[Fraction(0)] * target_rank for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 3xR system: A @ w = b
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # b = [1, 1, 1]
                    A = [[U[i*m+j][r] * V[j*p+k][r] for r in range(target_rank)] for j in range(m)]
                    b = [Fraction(1) for _ in range(m)]
                    
                    # Solve using exact linear algebra
                    # Use Gaussian elimination with fractions
                    # Augmented matrix [A | b]
                    aug = [row[:] + [b[j]] for j, row in enumerate(A)]
                    
                    # Gaussian elimination
                    rank_A = 0
                    for col in range(target_rank):
                        # Find pivot
                        pivot = -1
                        for row in range(rank_A, m):
                            if aug[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        
                        # Swap
                        aug[rank_A], aug[pivot] = aug[pivot], aug[rank_A]
                        
                        # Normalize
                        pivot_val = aug[rank_A][col]
                        for c in range(col, target_rank + 1):
                            aug[rank_A][c] /= pivot_val
                        
                        # Eliminate
                        for row in range(m):
                            if row != rank_A and aug[row][col] != 0:
                                factor = aug[row][col]
                                for c in range(col, target_rank + 1):
                                    aug[row][c] -= factor * aug[rank_A][c]
                        rank_A += 1
                    
                    # Check consistency
                    for row in range(rank_A, m):
                        if any(aug[row][c] != 0 for c in range(target_rank)):
                            valid = False
                            break
                    if not valid:
                        break
                    
                    # Extract solution (set free variables to 0)
                    w_sol = [Fraction(0)] * target_rank
                    for row in range(rank_A):
                        # Find the column where this row has a 1
                        for col in range(target_rank):
                            if aug[row][col] == 1:
                                w_sol[col] = aug[row][target_rank]
                                break
                    
                    # Check if solution is in half-integers
                    all_half_int = True
                    for val in w_sol:
                        if val.denominator not in [1, 2]:
                            all_half_int = False
                            break
                    if not all_half_int:
                        valid = False
                        break
                    
                    W[k*n+i] = w_sol
            
            if valid:
                # Final verification
                if verify(U, V, W):
                    best_rank = target_rank
                    best = {"U": U, "V": V, "W": W}
                    found = True
                    break
        
        if found:
            break
    
    # Convert fractions to int or float as needed
    def convert_entry(x):
        if isinstance(x, Fraction):
            if x.denominator == 1:
                return int(x.numerator)
            else:
                return float(x)
        return x
    
    U_out = [[convert_entry(x) for x in row] for row in best["U"]]
    V_out = [[convert_entry(x) for x in row] for row in best["V"]]
    W_out = [[convert_entry(x) for x in row] for row in best["W"]]
    
    return {"U": U_out, "V": V_out, "W": W_out}
