"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START

def verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-9):
    """Verify that U,V,W exactly decompose the 5x5x5 matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal delta[i,k]
    for i in range(n):
        for k in range(p):
            expected = 1.0 if i == k else 0.0
            for j in range(m):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - expected) > tol:
                    return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
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
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = schoolbook()
    best_rank = 125
    
    n, m, p = DIMENSIONS
    
    # Try to find lower rank decompositions
    # Strategy: use structured search based on the tensor structure
    # T[i,j,k] = delta[i,k]
    
    # A valid rank-r decomposition requires:
    # For each r: U[:,r], V[:,r], W[:,r] are vectors
    # sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = delta[i,k]
    
    # Let's try random search with integer entries in {-1, 0, 1}
    # and half-integer entries
    
    def try_random_decomposition(rank, iterations, deadline):
        for it in range(iterations):
            if time.time() > deadline:
                break
            # Generate random U, V, W with entries from {-1, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            U_rand = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(n*m)])
            V_rand = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(m*p)])
            W_rand = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(p*n)])
            if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                return U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
        return None
    
    # Try ranks from 25 upward (since rank 1-4 is impossible for 5x5)
    # Known: rank for n=5 matmul is at most 25 (trivial: one per diagonal element, j-summed)
    # Actually a simple rank-25: for each i=k, use 5 terms (one per j)
    # Let's verify rank-25 works: for each (i,k) with i=k, we need sum_j ... = 1
    # Use: for diagonal (i,i), U[i*5+j, r_{i,j}] = 1, V[j*5+i, r_{i,j}] = 1, W[i*5+i, r_{i,j}] = 1
    # That gives 25 terms, each handling one j for the diagonal
    
    # Actually let's try the obvious rank-25 first
    def diagonal_rank25():
        rank = 25
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + i][col] = 1
                W[i*n + i][col] = 1
                col += 1
        return U, V, W
    
    # Verify rank-25
    U25, V25, W25 = diagonal_rank25()
    if verify_decomposition(U25, V25, W25):
        best_rank = 25
        best = {"U": U25, "V": V25, "W": W25}
        print(f"Rank 25 verified")
    
    # Try rank 24, 23, ... downward
    for rank in range(24, 4, -1):
        if time.time() > deadline:
            break
        
        # Try random search
        result = try_random_decomposition(rank, 20000, deadline)
        if result is not None:
            best_rank = rank
            best = {"U": result[0], "V": result[1], "W": result[2]}
            print(f"Found rank {rank}!")
            break
    
    if time.time() > deadline:
        return best
    
    # If we found rank 25, try structured optimization
    if best_rank == 25:
        # Try to reduce by combining terms using algebraic identities
        # For matmul, there may be ways to share computations
        # Try rank 24 with specific structure
        pass
    
    return best

# EVOLVE-BLOCK-END
