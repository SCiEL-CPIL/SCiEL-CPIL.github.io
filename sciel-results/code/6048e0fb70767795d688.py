"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        if rank_ <= 0:
            return False
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    for r in range(rank_):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def make_result(U, V, W):
        """Convert to list-of-lists format with cleaned values."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up near-integer values
        for lst in [U_list, V_list, W_list]:
            for row in lst:
                for idx in range(len(row)):
                    rounded = round(row[idx], 10)
                    if abs(row[idx] - rounded) < 1e-8:
                        row[idx] = rounded
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    # Try structured decompositions with small ranks
    random.seed(42)
    
    # Coefficient sets to try - small integers and half-integers
    coeff_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
        [-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
        [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
    ]
    
    # Try ranks from a reasonable upper bound down to a lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can try lower as well
    
    # Try several target ranks
    target_ranks = [42, 35, 30, 28, 25, 24, 22, 21]
    
    for coeff_set in coeff_sets:
        for target_rank in target_ranks:
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            # Multiple random trials for this rank and coefficient set
            max_trials = 200
            for trial in range(max_trials):
                if time.time() > deadline:
                    break
                
                # Random initialization
                U = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(p * n)]
                
                if verify_decomposition(U, V, W):
                    result = make_result(U, V, W)
                    best_result = result
                    best_rank = target_rank
                    break
            if time.time() > deadline:
                break
    
    # If we found a better rank, try to go even lower
    if best_rank < schoolbook_rank:
        for target_rank in range(best_rank - 1, max(15, best_rank - 10), -1):
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            for coeff_set in coeff_sets[:2]:  # Use smaller coefficient sets for speed
                max_trials = 100
                for trial in range(max_trials):
                    if time.time() > deadline:
                        break
                    
                    U = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(n * m)]
                    V = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(m * p)]
                    W = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(p * n)]
                    
                    if verify_decomposition(U, V, W):
                        result = make_result(U, V, W)
                        best_result = result
                        best_rank = target_rank
                        break
                if time.time() > deadline:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
