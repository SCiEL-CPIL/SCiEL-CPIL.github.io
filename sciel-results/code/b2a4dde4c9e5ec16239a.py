"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, n, m, p):
        """Verify that U,V,W give a valid tensor decomposition using exact arithmetic."""
        rank = len(U[0])
        # Build the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        # Check that other entries are zero
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        s = 0
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
        return True
    
    # Try structured approaches: exploit the block structure
    # The tensor is sum over j of (sub-tensor for column j of A and row j of B)
    # For each j, the sub-tensor has rank 14 (2*7)
    # Total rank = 3 * 14 = 42 in schoolbook
    
    # Try to find a lower rank by allowing cross-j interactions
    # Use a random search with structured initialization
    
    def structured_random_search(rank, max_iters=2000, seed=42):
        """Random search with structured initialization."""
        random.seed(seed)
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                return None
            
            # Initialize with schoolbook-like structure but with random perturbations
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Assign columns to (i,j,k) triples with some randomness
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank:
                            break
                        # With some probability, assign to a different (i',j',k')
                        if random.random() < 0.3:
                            i2 = random.randint(0, n-1)
                            j2 = random.randint(0, m-1)
                            k2 = random.randint(0, p-1)
                        else:
                            i2, j2, k2 = i, j, k
                        
                        U[i * m + j][col] = random.choice([1, -1])
                        V[j * p + k][col] = random.choice([1, -1])
                        W[k * n + i][col] = random.choice([1, -1])
                        col += 1
            
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try decreasing rank from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() > deadline - 5.0:
            break
        result = structured_random_search(r, max_iters=1000, seed=r * 1000)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
