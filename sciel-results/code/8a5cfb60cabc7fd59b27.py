"""Matrix multiplication 2x3x5 - search for lower rank decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction
import random

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed() - SAFETY_MARGIN

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor: T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    Should equal 1 for all (i,j,k).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    try:
        U = [[Fraction(x) for x in row] for row in U]
        V = [[Fraction(x) for x in row] for row in V]
        W = [[Fraction(x) for x in row] for row in W]
    except:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if val != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = 2, 3, 5
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 30
    
    # Try to find lower rank decompositions
    # For small dimensions, we can attempt a structured search
    
    # Strategy: Try rank values from 29 down to some minimum
    # The minimal rank for 2x3x5 is at least the tensor rank lower bound.
    # For 2x3x5, known results suggest rank could be less than 30.
    
    # For this specific small case, let's try a direct approach:
    # Try to construct a rank-R decomposition with R < 30
    
    # For rank R, we need U (6xR), V (15xR), W (10xR) such that
    # sum_r U[i*3+j,r] * V[j*5+k,r] * W[k*2+i,r] = 1 for all i,j,k
    
    # This is a system of 30 equations in the entries of U, V, W.
    # Number of variables: 6R + 15R + 10R = 31R
    # So for R >= 1, we have more variables than equations (31R > 30).
    # But the equations are trilinear, making this a hard nonlinear problem.
    
    # For small R, we can try randomized approaches or structured solutions.
    
    # Let's try a specific construction:
    # If we can find rank R where R < 30, we return that.
    
    # Attempt: Use a bilinear algorithm structure
    # For 2x3x5, try to see if rank 29 or lower works.
    
    # Since this is computationally intensive, let's try a few heuristic approaches
    # with a time limit.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try to find a rank-R solution for R from 29 down to 25
    for target_rank in range(29, 24, -1):
        if _time_remaining() < 5.0:
            break
            
        # Try several random initializations
        for attempt in range(50):
            if _time_remaining() < 3.0:
                break
            
            # Random initialization with small values
            U = np.random.randn(6, target_rank) * 0.5
            V = np.random.randn(15, target_rank) * 0.5
            W = np.random.randn(10, target_rank) * 0.5
            
            # Try gradient-free local search (simulated annealing style)
            best_cost = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(2000):
                if _time_remaining() < 3.0:
                    break
                    
                # Compute cost: sum of (tensor_entry - 1)^2
                cost = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] 
                                     for r in range(target_rank))
                            cost += (val - 1.0) ** 2
                
                if cost < best_cost:
                    best_cost = cost
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if cost < 1e-8:
                    break
                
                # Perturbation
                dim = random.choice([0, 1, 2])
                if dim == 0:
                    idx = random.randint(0, 5)
                    col = random.randint(0, target_rank - 1)
                    U[idx, col] += random.gauss(0, 0.1)
                elif dim == 1:
                    idx = random.randint(0, 14)
                    col = random.randint(0, target_rank - 1)
                    V[idx, col] += random.gauss(0, 0.1)
                else:
                    idx = random.randint(0, 9)
                    col = random.randint(0, target_rank - 1)
                    W[idx, col] += random.gauss(0, 0.1)
            
            # Check if we found a valid solution
            if best_cost < 1e-6:
                # Convert to fractions if possible
                U_list = [[float(x) for x in row] for row in best_U]
                V_list = [[float(x) for x in row] for row in best_V]
                W_list = [[float(x) for x in row] for row in best_W]
                
                # Round to half-integers if close
                for row in U_list:
                    for idx, val in enumerate(row):
                        row[idx] = round(val * 2) / 2
                for row in V_list:
                    for idx, val in enumerate(row):
                        row[idx] = round(val * 2) / 2
                for row in W_list:
                    for idx, val in enumerate(row):
                        row[idx] = round(val * 2) / 2
                
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    result = {"U": U_list, "V": V_list, "W": W_list}
                    return result
        
    # Return the schoolbook solution as fallback
    return best

# EVOLVE-BLOCK-END
