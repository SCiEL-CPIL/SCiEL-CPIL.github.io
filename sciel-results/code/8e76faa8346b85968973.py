"""Matrix multiplication 2x3x4 tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_full(U, V, W, n, m, p, rank, tol=1e-9):
    """Full verification: check both on-conditions and off-conditions."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check on-conditions: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-conditions: T[a,b,c] = 0 for invalid combinations
    # For each (a, b), check all c
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            for c in range(p * n):
                k2, i2 = divmod(c, n)
                # Valid if i2==i, j2==j, k2==k
                if i2 == i and j2 == j and k2 == k:
                    continue
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_merge_pairs(n, m, p, deadline):
    """
    Try to merge pairs of schoolbook terms to reduce rank.
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if we can find
    coefficients such that the combined term satisfies all constraints.
    """
    start = time.time()
    n, m, p = 2, 3, 4
    total = n * m * p  # 24
    
    # Generate all schoolbook terms
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try to find a set of rank-1 terms that covers all 24 (i,j,k) triples
    # with some terms covering multiple triples
    
    # Strategy: try to find terms where each term covers 2+ triples
    # A term r has U[i*m+j,r], V[j*p+k,r], W[k*n+i,r]
    # For it to cover (i1,j1,k1) and (i2,j2,k2), we need:
    # U[i1*m+j1,r]*V[j1*p+k1,r]*W[k1*n+i1,r] contributes to (i1,j1,k1)
    # U[i2*m+j2,r]*V[j2*p+k2,r]*W[k2*n+i2,r] contributes to (i2,j2,k2)
    # And it should NOT contribute to other (i,j,k) triples
    
    # The simplest way to cover multiple triples is if they share some indices.
    # E.g., if i1==i2 and j1==j2, then only k differs.
    # In that case, we can set U[i*m+j,r]=1, V[j*p+k1,r]=a, V[j*p+k2,r]=b, W[k1*n+i,r]=1, W[k2*n+i,r]=1
    # But then for other k values, V[j*p+k,r]=0, so no contribution. Good.
    # However, we also need to check that this term doesn't interfere with other (a,b,c) combinations.
    
    # Let's try a structured approach: group by (i,j) and try to cover all k for each (i,j)
    # with fewer than p terms.
    
    # For a fixed (i,j), we need to cover k=0,1,2,3 with some number of terms.
    # Each term that covers (i,j,k) for multiple k must have V[j*p+k,r] nonzero for multiple k.
    # But then for other i',j' with the same j, it might interfere.
    
    # This is complex. Let me try a direct search over small coefficient sets.
    
    coeff_set = [0.0, 0.5, 1.0, -0.5, -1.0]
    
    for target_rank in range(23, 19, -1):
        if time.time() > deadline:
            break
        
        # Try random search with small coefficient set
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with schoolbook-like structure but with some merging
            # Each term covers 1-2 triples
            triples = list(product(range(n), range(m), range(p)))
            random.shuffle(triples)
            
            # Assign triples to terms, allowing some terms to cover 2 triples
            term_triples = {r: [] for r in range(target_rank)}
            idx = 0
            for r in range(target_rank):
                if idx < len(triples):
                    term_triples[r].append(triples[idx])
                    idx += 1
                # Sometimes add a second triple
                if idx < len(triples) and random.random() < 0.3:
                    term_triples[r].append(triples[idx])
                    idx += 1
            
            # Handle remaining triples
            while idx < len(triples):
                r = random.randint(0, target_rank - 1)
                term_triples[r].append(triples[idx])
                idx += 1
            
            # Assign coefficients
            for r in range(target_rank):
                assigned = term_triples[r]
                if not assigned:
                    continue
                
                # For each assigned triple, assign coefficients
                # Use a consistent pattern
                for (i, j, k) in assigned:
                    # Assign 1.0 to each factor for simplicity
                    U[i * m + j, r] += 1.0
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += 1.0
            
            if _verify_full(U, V, W, n, m, p, target_rank):
                return U, V, W, target_rank
    
    return None

def _try_structured(n, m, p, deadline):
    """
    Try a structured decomposition based on grouping.
    """
    start = time.time()
    n, m, p = 2, 3, 4
    
    # Try to use the structure: for each j, we have 2*4=8 (i,k) pairs.
    # Can we cover these with fewer than 4 terms per j?
    
    # Idea: use 2 terms per j, each covering 4 (i,k) pairs.
    # Term A for j: covers (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    # Term B for j: covers (i=0,k=2), (i=0,k=3), (i=1,k=2), (i=1,k=3)
    
    # But we need to check interference with other j values.
    
    # Let me try: for each j, use 2 terms.
    # Total rank = 3 * 2 = 6. That's too optimistic; let's see if it works.
    
    # Actually, let's try rank 12 first (2 per j instead of 4).
    
    for target_rank in [12, 16, 20]:
        if time.time() > deadline:
            break
        
        # Try the structured approach
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # For each j, use target_rank/m terms
        terms_per_j = target_rank // m
        
        for j in range(m):
            for t in range(terms_per_j):
                r = j * terms_per_j + t
                
                # This term covers some (i,k) pairs
                # Split k into groups
                k_start = t * (p // terms_per_j)
                k_end = (t + 1) * (p // terms_per_j)
                
                for i in range(n):
                    for k in range(k_start, k_end):
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
        
        if _verify_full(U, V, W, n, m, p, target_rank):
            return U, V, W, target_rank
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 10  # Short deadline for search, fallback to schoolbook
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try structured approach with limited time
    if time.time() < deadline:
        result = _try_structured(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_result = {
                    "U": [[float(U[i, r]) for r in range(rank)] for i in range(n * m)],
                    "V": [[float(V[i, r]) for r in range(rank)] for i in range(m * p)],
                    "W": [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
                }
                best_rank = rank
    
    # Try merge pairs approach with limited time
    if time.time() < deadline:
        result = _try_merge_pairs(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_result = {
                    "U": [[float(U[i, r]) for r in range(rank)] for i in range(n * m)],
                    "V": [[float(V[i, r]) for r in range(rank)] for i in range(m * p)],
                    "W": [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
                }
    
    return best_result
