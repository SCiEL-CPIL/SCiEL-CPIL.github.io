"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each (i,j,k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a 3-tensor contraction
        # We can do: for each c, outer product of U[:,c], V[:,c], W[:,c]
        # But that's expensive. Instead, compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_random_rank(target_rank, max_attempts, value_range):
        """Try to find a decomposition of the given rank with random initialization."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Generate random small integer matrices
            U = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(pn)]
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    def try_structured_rank(target_rank):
        """Try structured decompositions based on known patterns."""
        # For 2x3x7, try to find rank-21 or lower decompositions
        # One approach: use the fact that matrix multiplication tensor has structure
        # Try Kronecker-like or block structures
        
        # Try a simple structured approach:
        # U[i*m+j, c] = a[i,j,c], V[j*p+k, c] = b[j,k,c], W[k*n+i, c] = d[k,i,c]
        # We need sum_c a[i,j,c]*b[j,k,c]*d[k,i,c] = 1 for all i,j,k
        
        # For rank r, we have r*(nm+mp+pn) variables and n*m*p=42 equations
        # If r >= 42/(nm+mp+pn) = 42/41, so r >= 1 is theoretically possible
        # But we need integer solutions
        
        # Try a greedy approach: build up the decomposition
        # Start with zero matrices and try to add rank-1 components
        
        # For now, try random with structured values
        # Use values that are more likely to work: {-1, 0, 1}
        result = try_random_rank(target_rank, 200, (-1, 1))
        if result:
            return result
        
        # Try with {-2, -1, 0, 1, 2}
        result = try_random_rank(target_rank, 100, (-2, 2))
        if result:
            return result
        
        return None

    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
    # But tensor rank can be lower than this in some cases
    # For matrix multiplication, the rank is at least max(n*m, m*p, p*n) for some cases
    # Actually, the lower bound is more subtle. Let's try from 21 downward to see if we can go lower
    
    # First, try to find a rank-21 decomposition
    # If we can't find it quickly, try higher ranks
    
    # Strategy: try ranks from 21 up to 41 (schoolbook is 42)
    # For each rank, try multiple random candidates
    
    # Start with rank 21 (the theoretical lower bound for this case)
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try structured approach first
        result = try_structured_rank(target_rank)
        if result:
            best = result
            best_rank = target_rank
            continue
        
        # Try random with different value ranges
        # Use more attempts for lower ranks
        max_attempts = 50 if target_rank < 30 else 20
        result = try_random_rank(target_rank, max_attempts, (-2, 2))
        if result:
            best = result
            best_rank = target_rank
            continue
    
    return best
# EVOLVE-BLOCK-END
