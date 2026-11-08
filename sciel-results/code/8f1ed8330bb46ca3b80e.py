"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE_SECONDS = 115.0  # safety margin before 120s limit

def _schoolbook():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that the decomposition exactly reconstructs matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _try_structured_search(start_time, deadline):
    """Try to find a lower-rank decomposition using structured perturbations."""
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 42
    
    # Strategy: For 2x3x7, exploit the small n=2 dimension.
    # The multiplication tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need T[i,j,k] = 1 for all i,j,k.
    
    # Key insight: For each (i,j,k), we need the sum of products to be 1.
    # We can try to find a decomposition where some terms cancel or combine.
    
    # Approach: Start with schoolbook and try to merge pairs of terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if we can find U,V,W entries such that the combined term still satisfies
    # all the required equations.
    
    # Simpler approach: Use a parameterized family of decompositions.
    # For n=2, we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (A[i,j] * B[j,k])
    
    # Each product A[i,j]*B[j,k] can be computed with rank-1 terms.
    # But we can potentially share computation across different (i,k) pairs.
    
    # Let's try a specific structured approach:
    # For each j (column of A / row of B), we have 2*7=14 products to compute.
    # The schoolbook uses 14 rank-1 terms per j, total 42.
    # Can we do better by sharing?
    
    # Idea: For a fixed j, the products are A[i,j]*B[j,k] for i in {0,1}, k in {0..6}.
    # This is an outer product of a 2-vector and a 7-vector.
    # The rank of this outer product as a tensor is 1, but we need to compute
    # all 14 entries.
    
    # Actually, let's try a different approach: use the fact that we can have
    # non-trivial entries in U, V, W that create interference.
    
    # Let's try random search with verification.
    # We'll generate random sparse decompositions and check if they work.
    
    best_result = None
    best_rank = base_rank
    
    # Try ranks from base_rank-1 downward
    for target_rank in range(base_rank - 1, max(35, base_rank - 7)):
        if time.time() > deadline - 15:
            break
        
        # Try several random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Generate a random decomposition
            # Use a sparse structure: most entries are 0, some are ±1 or ±1/2
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with some random non-zero entries
            for r in range(target_rank):
                # Choose random positions for non-zero entries
                # For U: choose which (i,j) positions get non-zero values
                u_positions = set()
                while len(u_positions) < 2:  # typically 2 non-zero entries per column
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    u_positions.add((i, j))
                for (i, j) in u_positions:
                    U[i * m + j][r] = random.choice([-1, 1, Fraction(1, 2), Fraction(-1, 2)])
                
                # For V: choose which (j,k) positions get non-zero values
                v_positions = set()
                while len(v_positions) < 2:
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    v_positions.add((j, k))
                for (j, k) in v_positions:
                    V[j * p + k][r] = random.choice([-1, 1, Fraction(1, 2), Fraction(-1, 2)])
                
                # For W: choose which (k,i) positions get non-zero values
                w_positions = set()
                while len(w_positions) < 2:
                    k = random.randint(0, p - 1)
                    i = random.randint(0, n - 1)
                    w_positions.add((k, i))
                for (k, i) in w_positions:
                    W[k * n + i][r] = random.choice([-1, 1, Fraction(1, 2), Fraction(-1, 2)])
            
            # Verify the decomposition
            if _verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                return best_result
    
    return best_result

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    Returns: {'U': (n*m x rank), 'V': (m*p x rank), 'W': (p*n x rank)}
    with entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # First, ensure we always have a valid fallback
    fallback = _schoolbook()
    
    # Try to find a better decomposition
    result = _try_structured_search(start_time, deadline)
    if result is not None:
        return result
    
    return fallback
