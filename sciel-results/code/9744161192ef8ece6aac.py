"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Returns True if valid (within numerical tolerance), False otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is an integer)."""
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    """Check all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _sumdiff_decomposition(n, m, p):
    """
    For n=2, use the identity:
    A[0,j]*B[j,k] = 0.5*((A[0,j]+A[1,j]) + (A[0,j]-A[1,j])) * B[j,k]
    A[1,j]*B[j,k] = 0.5*((A[0,j]+A[1,j]) - (A[0,j]-A[1,j])) * B[j,k]
    
    This gives rank = 2*m*p (same as schoolbook for n=2), but we can try
    further optimizations by combining terms across j values.
    
    Actually, let's think more carefully. For each (j,k) pair, we need:
    - Term for A[0,j]*B[j,k]: coefficient 0.5 in both sum and diff
    - Term for A[1,j]*B[j,k]: coefficient 0.5 in sum, -0.5 in diff
    
    So for each (j,k), we use 2 terms:
    - Term 1: U[0,j]=0.5, U[1,j]=0.5, V[j,k]=1, W[k,0]=1, W[k,1]=-1
    - Term 2: U[0,j]=0.5, U[1,j]=-0.5, V[j,k]=1, W[k,0]=1, W[k,1]=1
    
    Wait, that's not quite right. Let me redo this.
    
    We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    
    For i=0: sum_r U[0,j,r] * V[j,k,r] * W[k,0,r] = 1
    For i=1: sum_r U[1,j,r] * V[j,k,r] * W[k,1,r] = 1
    
    Using sum/diff:
    Let s_j = A[0,j] + A[1,j], d_j = A[0,j] - A[1,j]
    Then A[0,j] = (s_j + d_j)/2, A[1,j] = (s_j - d_j)/2
    
    C[0,k] = sum_j A[0,j] B[j,k] = 0.5 * sum_j (s_j + d_j) B[j,k]
    C[1,k] = sum_j A[1,j] B[j,k] = 0.5 * sum_j (s_j - d_j) B[j,k]
    
    So we need bilinear forms for:
    - s_j * B[j,k] for all j,k
    - d_j * B[j,k] for all j,k
    
    That's still 2*m*p terms. No gain.
    
    Let me try a different approach: exploit the fact that for fixed k,
    we're computing a dot product of two length-m vectors.
    
    For fixed k, C[:,k] = A * B[:,k]
    This is a 2x5 times 5x1 multiplication, which has rank 5 (schoolbook).
    But we have 6 such k values, so total rank is 6*5 = 30? No, that's not right either.
    
    Actually, let's just try to find a better construction using random search
    with verification.
    """
    import random
    random.seed(42)
    
    # Try rank from 30 up to 60
    for rank in range(30, 61):
        if time.time() > time.time() + 100:  # placeholder
            break
            
        # Random initialization
        U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(p * n)]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
        pass
    
    # Try sum-diff decomposition
    result = _sumdiff_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
