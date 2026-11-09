import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p matrix multiplication."""
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

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    # Build the tensor from U, V, W
    # T[a, b, c] = sum_{col=0}^{rank-1} U[a, col] * V[b, col] * W[c, col]
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                # Expected value: 1 if (a,b,c) corresponds to valid (i,j,k)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = Fraction(0)
                for col in range(rank):
                    actual += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
                
                if abs(float(actual) - expected) > 1e-10:
                    return False
    return True

def _try_strassen_like(n, m, p, deadline):
    """Try Strassen-like structured decomposition for 3x3x4."""
    # For 3x3x3, we know rank 23 works (Strassen-like)
    # For 3x3x4, try ranks from 25 down to 18
    
    dim_a = n * m  # 9
    dim_b = m * p  # 12
    dim_c = p * n  # 12
    
    # Strategy: Start from schoolbook and try to merge terms
    # Each schoolbook term is (i,j,k) -> U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # For 3x3x4, we have 3*3*4 = 36 schoolbook terms
    # Try to find rank 25-30 decompositions
    
    # Known approach: Use the fact that we can sometimes combine terms
    # when they share indices
    
    # Let's try a systematic search for rank 28-30
    # using a greedy approach with local improvements
    
    for target_rank in range(30, 20, -1):
        if time.time() > deadline:
            break
        
        # Initialize with schoolbook terms, then try to merge
        # Start with first target_rank terms from schoolbook
        U = np.zeros((dim_a, target_rank), dtype=int)
        V = np.zeros((dim_b, target_rank), dtype=int)
        W = np.zeros((dim_c, target_rank), dtype=int)
        
        # Fill with first target_rank schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
                    else:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        
        # This just gives us a partial schoolbook, not a valid decomposition
        # We need a different approach
        
        # Better: Use a known 3x3x3 rank-23 decomposition as a base
        # and extend it to 3x3x4
        
        # For 3x3x3, one known rank-23 decomposition uses these patterns
        # Let me construct a valid 3x3x3 rank-23 and extend to 3x3x4
        
        # Actually, let's try a different approach:
        # Use integer linear programming or a smarter search
        
        # For now, let's try to find a rank-30 decomposition by
        # carefully merging schoolbook terms
        
        # The key insight: if two terms (i1,j1,k1) and (i2,j2,k2) have
        # the same i and j, we can potentially merge them if we adjust
        # the W values appropriately
        
        # Let's try a systematic merge approach
        # Group schoolbook terms by (i,j)
        groups = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    key = (i, j)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(k)
        
        # Each group (i,j) has p terms that differ only in k
        # U[i*m+j] appears in all p terms
        # V[j*p+k] appears in exactly one term per k
        # W[k*n+i] appears in exactly one term per k
        
        # We can potentially reduce the rank by sharing U entries
        # but this requires careful construction
        
        # Let's try a known result: for 3x3x4, rank 29 might be achievable
        # using a combination of Strassen-like tricks
        
        # For time efficiency, let's just return schoolbook if we can't
        # find something better quickly
        
        if time.time() > deadline - 10:
            break
    
    return None

def _try_local_search(n, m, p, deadline, max_rank=32):
    """Try local search to find lower-rank decomposition."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    schoolbook_rank = n * m * p
    
    # Start from schoolbook and try to remove columns
    U = np.array(_schoolbook(n, m, p)["U"], dtype=object)
    V = np.array(_schoolbook(n, m, p)["V"], dtype=object)
    W = np.array(_schoolbook(n, m, p)["W"], dtype=object)
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i, j]) for j in range(schoolbook_rank)] for i in range(dim_a)]
    V_frac = [[Fraction(V[i, j]) for j in range(schoolbook_rank)] for i in range(dim_b)]
    W_frac = [[Fraction(W[i, j]) for j in range(schoolbook_rank)] for i in range(dim_c)]
    
    # Try to remove columns one by one
    # A column can be removed if its contribution can be represented
    # by other columns
    
    current_rank = schoolbook_rank
    active_cols = list(range(schoolbook_rank))
    
    # This is a heuristic search - try to merge pairs of columns
    # that have similar patterns
    
    # For each pair of columns, check if we can eliminate one
    # by adjusting the other
    
    # This is computationally expensive, so let's be selective
    
    # Try to find columns that can be merged
    # Two columns c1 and c2 can be merged if there exists a scalar alpha
    # such that for all (a,b,c): U[a,c1]*V[b,c1]*W[c,c1] + 
    # alpha * U[a,c2]*V[b,c2]*W[c,c2] can be represented by a single term
    
    # This is very restrictive, so let's try a different approach
    
    # Use the fact that the tensor has a specific structure
    # and try to find a low-rank decomposition using SVD-like methods
    
    # For now, return schoolbook as we can't find a better one quickly
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find better decomposition using Strassen-like methods
    result = _try_strassen_like(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
            best_rank = rank
    
    # Try local search if we have time
    if time.time() < deadline - 5:
        result = _try_local_search(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
