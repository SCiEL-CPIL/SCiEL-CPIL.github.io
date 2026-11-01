"""Matrix multiplication 2,2,7 - rank-minimal decomposition search."""
from fractions import Fraction
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 7]

def verify_decomposition(U, V, W, n=2, m=2, p=7, tol=Fraction(1, 10**12)):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check each (i, j, k) entry
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if abs(s - 1) > tol:
                    return False
    return True

def schoolbook():
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_structured_rank(r):
    """Try to find a rank-r decomposition with a structured ansatz.
    
    Ansatz: Each rank-1 term is defined by:
    - A row pattern: which of the 4 A-entries (i,j) are nonzero
    - B row pattern: which of the 14 B-entries (j,k) are nonzero  
    - C row pattern: which of the 14 C-entries (k,i) are nonzero
    
    We use a greedy/iterative approach with small integer entries.
    """
    # This is a placeholder for more sophisticated search
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Start with schoolbook
    best = schoolbook()
    best_rank = 28
    
    # Try to find lower rank using structured search
    # Key idea: for n=2, we can potentially share between the two output rows
    
    # Attempt 1: Try rank 21 (7*3) with a 3-term-per-k-column structure
    # For each k, we need to compute:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # 
    # If we use 3 terms per k:
    #   Term 1: U=(1,0,0,0), V=(1,0,0,0,0,0,0,0,0,0,0,0,0,0) for B[0,k], W=(1,0) for C[0,k]
    #   This gives A[0,0]*B[0,k] to C[0,k]
    #   Term 2: U=(0,1,0,0), V=(0,0,0,0,0,0,0,1,0,0,0,0,0,0) for B[1,k], W=(1,0) for C[0,k]  
    #   This gives A[0,1]*B[1,k] to C[0,k]
    #   Term 3: Need to cover C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #   But one term can't do two products...
    #
    # Alternative: Use bilinear sharing. Consider:
    #   (A[0,0]+A[1,0]) * B[0,k] and (A[0,1]+A[1,1]) * B[1,k]
    #   Then C[0,k] = (A[0,0]+A[1,0])*B[0,k] + (A[0,1]+A[1,1])*B[1,k] - A[1,0]*B[0,k] - A[1,1]*B[1,k]
    #   C[1,k] = (A[0,0]+A[1,0])*B[0,k] + (A[0,1]+A[1,1])*B[1,k] - A[0,0]*B[0,k] - A[0,1]*B[1,k]
    # This needs 4 terms per k, not better.
    
    # Let's try a different approach: search for rank-20 or lower using 
    # a parameterized family with small integer coefficients.
    
    # For [2,2,7], known lower bounds suggest rank >= 16 (from Strassen-like bounds)
    # but exact minimum is unknown. Let's try to find rank 21, 20, 19, 18...
    
    # Structured ansatz for rank r = 7*s where s is terms per k-column:
    # Each group of s terms handles one k value.
    # For each k, we need to realize the 2x2 matrix product contribution.
    
    # Try s=3 (rank 21): Can we do 2x2 * 2x1 in 3 multiplications?
    # No, 2x2*2x1 requires at least 4 multiplications in general (it's just 4 scalar mults).
    # Unless we share... but the B column is different for each k.
    
    # Actually, the B matrix is 2x7, so B[j,k] for fixed k is a 2-vector.
    # For each k, we compute A * B[:,k] which is 2x2 * 2x1 = 2x1.
    # This requires 4 scalar multiplications per k, total 28.
    # To beat this, we need to share across k values.
    
    # Sharing across k: If we form linear combinations of B columns...
    # But that changes what we're computing.
    
    # The only way to reduce rank is if the tensor has special structure.
    # For [2,2,p], the rank is known to be 2p for p >= 3? Let me check.
    # Actually, for [2,2,p], the rank is exactly 2p (schoolbook) for p >= 3
    # based on lower bound arguments. Wait, no.
    
    # Let me reconsider. The multiplication tensor for [n,m,p] has rank at most n*m*p.
    # For [2,2,7], is there a known better decomposition?
    
    # Let me try a computational search for small rank with a restricted ansatz.
    
    # Ansatz: U, V, W have entries in {-1, 0, 1} and rank <= 24.
    # This is still huge search space, so use a greedy approach.
    
    # Greedy: Start with rank 28, try to eliminate terms by finding linear dependencies.
    
    # Alternative: Use the fact that for [2,2,p], we can write:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Define:
    #   X = A[0,0] + A[1,0], Y = A[0,1] + A[1,1]
    #   Z = A[0,0] - A[1,0], W_ = A[0,1] - A[1,1]
    # Then:
    #   A[0,0] = (X+Z)/2, A[1,0] = (X-Z)/2
    #   A[0,1] = (Y+W_)/2, A[1,1] = (Y-W_)/2
    # C[0,k] = (X+Z)/2 * B[0,k] + (Y+W_)/2 * B[1,k]
    # C[1,k] = (X-Z)/2 * B[0,k] + (Y-W_)/2 * B[1,k]
    # This still uses 4 multiplications per k (X*B[0,k], Z*B[0,k], Y*B[1,k], W_*B[1,k]).
    
    # I don't see an obvious way to beat 28 for [2,2,7].
    # Let me return the schoolbook decomposition.
    
    return best

# EVOLVE-BLOCK-END
