import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    # Shapes: U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    # Target tensor: T[i,j,k] = δ_{j,k} for the multiplication
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = δ_{j,k}
    
    # Start with schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        # Build the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    target = 1 if j == k else 0
                    if abs(val - target) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search
    # Use a structured approach: try rank from best_rank-1 down to some minimum
    
    # For each candidate rank, try random initialization and verify
    for target_rank in range(best_rank - 1, 15, -1):  # Try ranks from 31 down to 16
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random initialization with small values
            U = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        # Early stop if we found a good rank
        if best_rank <= 20:
            break
    
    return best
