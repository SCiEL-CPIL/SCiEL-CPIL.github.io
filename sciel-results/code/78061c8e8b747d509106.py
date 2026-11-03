"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Schoolbook baseline: rank = n*m*p = 60
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if s != 1:
                        return False
        return True
    
    # Strategy: Try structured decompositions with rank < 60
    # For matrix multiplication, we can use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 3-tensor of order (n*m, m*p, p*n) with all entries = 1
    
    # Known approach: Use rank-1 tensors with coefficients
    # T[i,j,k] = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r] = 1
    
    # Try small ranks with structured search
    # Lower bound is 20, try ranks from 20 upward
    
    # For efficiency, use the observation that for the tensor to have all entries = 1,
    # we need a very specific structure. Let's try a parametric family.
    
    # One known approach: For m=4, we can think of B as having 4 columns,
    # and use the fact that 4 = 2+2, potentially using a 2x2 block structure.
    
    # Let's try a direct search for low-rank decompositions using the fact
    # that the tensor is "all ones" in a specific sense.
    
    # Actually, the key insight: the matrix multiplication tensor T[i,j,k] = 1
    # for all i,j,k means we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k
    
    # This is a system of n*m*p = 60 equations in rank variables per factor.
    
    # Let's try rank 30 first (half of schoolbook) with a structured approach.
    # We can try: U, V, W have entries in {-1, 0, 1} or {-1/2, 0, 1/2, 1}
    
    # For a more systematic approach, let's use the fact that if we set
    # U[i*m+j, r] = a[i,j,r], V[j*p+k, r] = b[j,k,r], W[k*n+i, r] = c[k,i,r]
    # then sum_r a[i,j,r]*b[j,k,r]*c[k,i,r] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the all-ones 3-tensor
    # with the specific index structure.
    
    # Let's try a greedy/iterative approach:
    # 1. Start with rank r
    # 2. Try to find U, V, W with small integer/half-integer entries
    
    # For rank r, we have r columns in each matrix. Let's try r from 20 to 59.
    
    # A smarter approach: use the fact that the problem is linear in each factor
    # given the other two. We can fix U and V, then solve for W linearly.
    
    # But this is still combinatorially hard. Let's try a few specific structured
    # decompositions that are known to work for small matrices.
    
    # Known result: For 2x2x2, Strassen gives rank 7 < 8.
    # For 3x3x3, the best known is 23 < 27.
    # For 3x4x5, we don't have a well-known optimal, but we can try to beat 60.
    
    # Let's try a simple structured decomposition:
    # Use the fact that 3x4x5 = (1+2)x4x(1+4)
    # This doesn't immediately help, but let's try random structured search.
    
    # For each candidate rank r, try to find a solution with entries in {-1, 0, 1}
    # or {-1/2, 0, 1/2, 1}
    
    import random
    random.seed(42)
    
    # Try ranks from 40 down to 20 (skip schoolbook 60, try to find < 60)
    # For each rank, do a limited random search
    
    best_found = None
    
    # We'll try a few ranks and for each, do a limited search
    for target_rank in [50, 45, 40, 35, 30]:
        if time.time() > deadline - 10:
            break
        
        # Try a few random structured decompositions for this rank
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with small random values from {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W, target_rank):
                best_found = (target_rank, U, V, W)
                break
        
        if best_found:
            break
    
    if best_found:
        best_rank, best_U, best_V, best_W = best_found
    
    # If we didn't find anything better, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
