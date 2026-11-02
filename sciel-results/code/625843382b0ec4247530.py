"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook decomposition (guaranteed correct, rank = n*m*p)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72
    
    # Verify a decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True
    
    # Try the simple rank-8 construction from the original
    def try_rank_8():
        rank = p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(p):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1
                    V[j * p + r][r] = 1
                    W[r * n + i][r] = 1
        
        if verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W, "rank": rank}
        return None
    
    # Try rank 8
    result = try_rank_8()
    if result:
        best_result = result
        best_rank = result["rank"]
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    
    # Try some structured decompositions with lower rank
    # For 3x3 x 3x8, we can try various patterns
    
    def try_structured(rank, pattern_seed=42):
        if rank >= best_rank:
            return None
        if rank < 1:
            return None
            
        rng = random.Random(pattern_seed)
        
        # Try to find a decomposition with the given rank
        # Use a greedy approach or random search
        # For simplicity, try a few random initializations
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W, "rank": rank}
        
        return None
    
    # Try ranks from best_rank - 1 down to some minimum
    for rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 5:
            break
        result = try_structured(rank)
        if result:
            best_result = result
            best_rank = result["rank"]
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
