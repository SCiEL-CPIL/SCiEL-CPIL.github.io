"""
Matrix multiplication 3x5x7 - rank-minimal tensor decomposition.

The task is to find factor matrices U, V, W such that for all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
and for all other (a,b,c) triples, the sum is 0.

This is the standard matrix multiplication tensor of size (n*m) x (m*p) x (p*n).
"""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    """
    n, m, p = 3, 5, 7
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    schoolbook_rank = nm * mp * pn  # This is wrong, schoolbook is n*m*p = 105
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Build the support: set of (a, b, c) triples where the tensor is 1
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.add((a, b, c))
    
    # Schoolbook decomposition: rank = n*m*p = 105
    # Each (i,j,k) triple gets its own rank-1 term
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Compute tensor T[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # Use einsum for efficiency
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check support elements
        for (a, b, c) in support:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        
        # Check non-support elements are approximately 0
        # We only need to check that elements not in support are ~0
        # But checking all 15*35*21 = 11025 elements is fine
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in support:
                        if abs(T[a, b, c]) > tol:
                            return False
        return True
    
    # Try to find a better decomposition using randomized search
    # Strategy: random search over small integer/rational entries
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try various ranks from low to high
    # The theoretical minimum rank for 3x5x7 matrix mult is unknown but 
    # certainly less than 105. Let's try to find something better.
    
    # For now, let's try a simple random search with small integer entries
    rng = random.Random(42)
    
    # Try ranks from 80 down to 60, with random initialization
    for target_rank in range(90, 60, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random small integer entries
            U = np.array([[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)], dtype=np.float64)
            V = np.array([[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)], dtype=np.float64)
            W = np.array([[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)], dtype=np.float64)
            
            if verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank < 105:
            break
    
    # If we found something better, return it
    if best_rank < 105:
        # Convert to lists for the required output format
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Otherwise, return schoolbook
    U, V, W, rank = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
