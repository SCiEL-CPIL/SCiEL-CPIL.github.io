"""Matrix multiplication 2x3x7 rank-minimal exact decomposition search."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]
n, m, p = DIMENSIONS
NM = n * m
MP = m * p
PN = p * n

# Precompute the matrix multiplication tensor as a list of (i,j,k) indices
# where the output index is k*n + i (trace convention W[k*n+i, r])
# Input A indices: i*m + j  (i in [0,n), j in [0,m))
# Input B indices: j*p + k  (j in [0,m), k in [0,p))
# Output C indices: k*n + i (k in [0,p), i in [0,n))

# The tensor T[a, b, c] = 1 if there exists (i,j,k) such that
# a = i*m+j, b = j*p+k, c = k*n+i
# We can precompute the set of valid (a,b,c) triples.
_tensor_triples = []
for i in range(n):
    for j in range(m):
        for k in range(p):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            _tensor_triples.append((a, b, c))

def _check_decomposition(U, V, W, rank):
    """
    Check if U (NM x rank), V (MP x rank), W (PN x rank) exactly decompose the tensor.
    The decomposition says: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    For exact integer/half-integer entries, we check each triple.
    """
    for (a, b, c) in _tensor_triples:
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if s != 1:
            return False
    # Also check that non-triples give 0
    # This is expensive, but we can skip for speed during search and verify at end
    return True

def _check_full_decomposition(U, V, W, rank):
    """Full check including zeros for non-triples."""
    # Check all triples give 1
    for (a, b, c) in _tensor_triples:
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if s != 1:
            return False
    # Check all non-triples give 0
    # Build set of valid triples for O(1) lookup
    valid_set = set(_tensor_triples)
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                if (a, b, c) not in valid_set:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 42."""
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_blocks():
    """
    Try to build a decomposition using Strassen-like 2x2x2 blocks.
    For 2x3x7, we can think of it as combining smaller blocks.
    
    Strategy: Use the known optimal rank for 2x2x2 (7) as a building block.
    2x3x7 can be viewed as 2x3 blocks times 3x7, or other factorizations.
    
    Actually, let's try a direct approach: random search with pruning.
    """
    pass

def _random_search(max_rank, iterations, deadline):
    """
    Random search for a decomposition with rank = max_rank.
    We try random integer/half-integer entries and check.
    """
    best = None
    for it in range(iterations):
        if time.time() > deadline:
            break
        # Generate random U, V, W with small integer entries
        # Use entries from {-1, 0, 1} initially
        U = [[random.randint(-1, 1) for _ in range(max_rank)] for _ in range(NM)]
        V = [[random.randint(-1, 1) for _ in range(max_rank)] for _ in range(MP)]
        W = [[random.randint(-1, 1) for _ in range(max_rank)] for _ in range(PN)]
        
        # Quick check: only verify the tensor triples (skip zeros for speed)
        ok = True
        for (a, b, c) in _tensor_triples:
            s = 0
            for r in range(max_rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 1:
                ok = False
                break
        if ok:
            # Full verification
            if _check_full_decomposition(U, V, W, max_rank):
                return {"U": U, "V": V, "W": W}
    return None

def _structured_search(max_rank, deadline):
    """
    Structured search: exploit the block structure.
    For 2x3x7, we can try to find a decomposition where the 2x2 part uses Strassen (rank 7)
    and the rest is handled efficiently.
    
    Let's try: A is 2x3, B is 3x7.
    We can split B into columns: B = [B1, B2, B3, B4, B5, B6, B7] where each Bj is 3x1.
    Or split A into rows: A = [A1; A2] where A1 is 1x3, A2 is 1x3.
    
    Actually, let's try a more systematic approach:
    Assume the decomposition has a block structure based on the index ranges.
    """
    best = None
    for it in range(100000):
        if time.time() > deadline:
            break
        # Generate with entries from {-2,-1,0,1,2}
        U = [[random.randint(-2, 2) for _ in range(max_rank)] for _ in range(NM)]
        V = [[random.randint(-2, 2) for _ in range(max_rank)] for _ in range(MP)]
        W = [[random.randint(-2, 2) for _ in range(max_rank)] for _ in range(PN)]
        
        ok = True
        for (a, b, c) in _tensor_triples:
            s = 0
            for r in range(max_rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 1:
                ok = False
                break
        if ok:
            if _check_full_decomposition(U, V, W, max_rank):
                return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook (rank 42)
    schoolbook = _schoolbook()
    best_result = schoolbook
    best_rank = n * m * p  # 42
    
    # Try decreasing ranks
    # Known lower bound: max(n*m, m*p, p*n, (n*m*p)^{1/2}) = max(6, 21, 14, sqrt(42)~6.5) = 21
    # So rank >= 21. Let's try from 21 upward.
    
    for rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search with this rank
        result = _random_search(rank, 5000, deadline - 5)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try structured search
        if time.time() > deadline - 10:
            break
        result = _structured_search(rank, deadline - 5)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
