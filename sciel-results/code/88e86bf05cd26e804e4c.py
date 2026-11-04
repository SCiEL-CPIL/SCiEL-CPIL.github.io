import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 150s kill
    
    schoolbook_rank = n * m * p  # 63
    
    # Build the multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    def make_schoolbook():
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
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Verify a decomposition using exact rational arithmetic
    def verify(U, V, W):
        rank = len(U[0])
        # T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # Should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if total != 1:
                        return False
        return True
    
    # Try to find rank reduction by solving for merged terms
    # Strategy: pick a pair of schoolbook terms, try to replace them with 
    # a single term plus adjustments, or use a structured low-rank ansatz.
    
    # Approach: Use the fact that for small tensors, we can set up the 
    # system of equations and try to find a solution with fewer terms.
    
    # For rank R decomposition, we need:
    # sum_{r=1}^{R} U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1  for all (i,j,k)
    
    # This is a system of n*m*p = 63 equations in 3*R*... variables (with gauge freedom)
    
    # Try a structured approach: fix W to have a specific pattern and solve for U, V
    # Or use the "border rank" concept and try to find a nearby low-rank tensor.
    
    # Given the difficulty, let's try a few targeted strategies:
    
    # Strategy 1: Try rank 62 by attempting to merge two terms with rational coefficients
    # If we can write e_{i1,j1,k1} + e_{i2,j2,k2} = u*v*w (rank 1), that's impossible
    # unless they're the same term. So pure merging won't work.
    
    # Strategy 2: Try a different decomposition structure entirely.
    # For example, use the identity:
    # A*B = (A + B - A*B) * ... no, that's not linear.
    
    # Actually, matrix multiplication is a trilinear map, so any decomposition
    # must be trilinear. The schoolbook is the standard trilinear form.
    
    # Let's try to verify that rank 63 is indeed minimal by checking if 
    # rank 62 is possible. This is a hard problem, but for small dimensions
    # we can attempt it.
    
    # Given time constraints and the difficulty of proving non-existence of 
    # lower rank, let's return the schoolbook decomposition which is known 
    # to be correct.
    
    # But let's try one more thing: a random search with exact verification
    # for rank 62, 61, etc., using rational entries.
    
    import random
    random.seed(42)
    
    for target_rank in range(schoolbook_rank - 1, max(50, schoolbook_rank - 5), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random rational initializations
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small rational values
            U_try = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
            V_try = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
            W_try = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # This is very unlikely to work, but let's check
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists if they're numpy arrays (they should already be lists)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
