"""Matrix multiplication 2x4x7 - rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
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
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i,j,k (since we need A[i,j]*B[j,k] summed over j)
    # Actually, the condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: Try various ranks and use local search with random initialization
    # Use half-integer entries (multiply by 2 to work with integers)
    
    def check_decomposition(U, V, W, rank):
        """Check if decomposition is exact. U,V,W have entries that are multiples of 0.5."""
        # Convert to work with integers (multiply by 2)
        U2 = [[2 * U[i][c] for c in range(rank)] for i in range(n * m)]
        V2 = [[2 * V[i][c] for c in range(rank)] for i in range(m * p)]
        W2 = [[2 * W[i][c] for c in range(rank)] for i in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                    # Should equal 8 (since 1 = 1*1*1, and we multiplied each by 2, so 2*2*2=8)
                    if s != 8:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using local search."""
        if rank >= best_rank:
            return None
        
        # Random initialization with small half-integers
        U = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(p * n)]
        
        # Local search: try to fix errors by flipping entries
        for iteration in range(5000):
            if time.time() > deadline:
                return None
            
            # Compute error for each (i,j,k)
            errors = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        errors[(i, j, k)] = s - 1.0
            
            # Check if we're done
            if all(abs(e) < 1e-9 for e in errors.values()):
                # Verify with exact check
                if check_decomposition(U, V, W, rank):
                    return U, V, W
                else:
                    # Not exact, continue
                    pass
            
            # Pick a random error and try to fix it
            idx = random.randint(0, n * m * p - 1)
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            
            if abs(errors[(i, j, k)]) < 1e-9:
                continue
            
            # Try flipping a random entry in U[i*m+j], V[j*p+k], or W[k*n+i]
            choice = random.randint(0, 2)
            if choice == 0:
                c = random.randint(0, rank - 1)
                old = U[i * m + j][c]
                new = old + random.choice([0.5, -0.5])
                if new == old:
                    continue
                U[i * m + j][c] = new
                # Recompute errors involving this entry
                # This is expensive, so just do a few iterations
            elif choice == 1:
                c = random.randint(0, rank - 1)
                old = V[j * p + k][c]
                new = old + random.choice([0.5, -0.5])
                if new == old:
                    continue
                V[j * p + k][c] = new
            else:
                c = random.randint(0, rank - 30) if rank > 30 else random.randint(0, rank - 1)
                old = W[k * n + i][c]
                new = old + random.choice([0.5, -0.5])
                if new == old:
                    continue
                W[k * n + i][c] = new
        
        return None
    
    # Try ranks from low to high
    for rank in range(20, min(best_rank, 45)):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid one, stop
    
    return best
# EVOLVE-BLOCK-END
