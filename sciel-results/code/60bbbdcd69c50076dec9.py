"""Rewritten program for matrix multiplication 2x3x4 tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if the decomposition is valid."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
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

def _try_rank(n, m, p, target_rank, max_attempts=100, deadline=None):
    """
    Try to find a rank-`target_rank` decomposition using structured random search.
    We use the idea that each rank-1 term can cover multiple (i,j,k) triples.
    We initialize with schoolbook and then try to merge/cancel terms.
    """
    start = time.time()
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Strategy: start from a sparse random structure
        # Each term r contributes to a subset of (i,j,k) triples
        # We want sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
        
        # Use a different approach: random initialization with structured sparsity
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # For each term, pick a small number of (i,j,k) triples to cover
        # and assign coefficients
        triples = list(product(range(n), range(m), range(p)))
        num_triples = len(triples)
        
        # Randomly assign triples to terms (with some overlap allowed)
        # Each triple should be covered by at least one term
        # We use a "covering" approach: assign each triple to 1-3 random terms
        
        # First, ensure every triple is covered
        term_assignments = {r: [] for r in range(target_rank)}
        for idx, (i, j, k) in enumerate(triples):
            # Assign to 1 or 2 random terms
            num_assign = random.randint(1, 2)
            terms = random.sample(range(target_rank), min(num_assign, target_rank))
            for r in terms:
                term_assignments[r].append((i, j, k))
        
        # Now assign coefficients
        # For each term r, we need to assign U, V, W values such that
        # the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] sums correctly
        # This is complex, so we use random coefficients and hope for the best
        
        for r in range(target_rank):
            assigned = term_assignments[r]
            if not assigned:
                # Give this term a random small contribution
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                c = random.choice([1.0, -1.0, 0.5, -0.5])
                U[i * m + j, r] = c
                V[j * p + k, r] = c
                W[k * n + i, r] = c
                continue
            
            # For each assigned triple, assign consistent coefficients
            # We want U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] to be nonzero
            # Let's pick a base coefficient for this term
            base = random.choice([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
            for (i, j, k) in assigned:
                # Split the coefficient across U, V, W
                # Use a simple split: all get base^(1/3) or similar
                # For simplicity, assign base to one and 1 to others, or split evenly
                # Let's try: U gets base, V gets 1, W gets 1 (or variations)
                choice = random.randint(0, 2)
                if choice == 0:
                    U[i * m + j, r] += base
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += 1.0
                elif choice == 1:
                    U[i * m + j, r] += 1.0
                    V[j * p + k, r] += base
                    W[k * n + i, r] += 1.0
                else:
                    U[i * m + j, r] += 1.0
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += base
        
        # Check validity
        if _verify(U, V, W, n, m, p, target_rank):
            return U, V, W
    
    return None

def _try_greedy_merge(n, m, p, max_deadline):
    """
    Try to reduce rank by greedily merging schoolbook terms.
    Start with rank 24 and try to find linear dependencies.
    """
    start = time.time()
    n, m, p = 2, 3, 4
    total_triples = n * m * p  # 24
    
    # Build the tensor as a vector of 24 ones
    # Each rank-1 term is a vector in the space of (i,j,k) triples
    # Term r has entries: for each (i,j,k), value = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # We want to find rank-1 terms whose sum is the all-ones vector
    # with as few terms as possible
    
    # Approach: start with all 24 standard basis terms
    # Then try to replace pairs of terms with a single term that covers both
    # This is like finding a sparse representation
    
    # Actually, let's try a different approach: random rank-1 terms
    # and use iterative methods to fit
    
    # For now, let's try a structured approach:
    # Group triples by their (i,j) or (j,k) or (i,k) patterns
    
    # Try to find a decomposition where each term covers multiple triples
    # that share some index pattern
    
    # Let's try: for each i, group all (j,k) pairs
    # Or for each j, group all (i,k) pairs
    
    # Actually, the simplest improvement might be to use the fact that
    # matrix multiplication can sometimes be done with fewer multiplications
    # using Strassen-like tricks, but for 2x3x4 this is not standard.
    
    # Let's try a random search with a better initialization:
    # Start with random rank-1 terms and use gradient descent or similar
    
    # Given time constraints, let's just try a few more random configurations
    # with a focus on terms that cover adjacent triples
    
    for target_rank in range(23, 18, -1):
        if time.time() > max_deadline:
            break
        
        for attempt in range(50):
            if time.time() > max_deadline:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Create terms that cover 2-3 triples each
            triples = list(product(range(n), range(m), range(p)))
            random.shuffle(triples)
            
            triples_per_term = total_triples / target_rank
            term_size = max(1, int(round(triples_per_term)))
            
            for r in range(target_rank):
                # Pick term_size triples for this term
                start_idx = r * term_size
                end_idx = min(start_idx + term_size, total_triples)
                if start_idx >= total_triples:
                    # Wrap around or pick random
                    selected = random.sample(triples, min(term_size, total_triples))
                else:
                    selected = triples[start_idx:end_idx]
                
                if not selected:
                    selected = [random.choice(triples)]
                
                # Assign coefficients to cover these triples
                # For each triple (i,j,k), we want the product to be nonzero
                # Use a consistent coefficient for this term
                c = random.choice([1.0, -1.0])
                for (i, j, k) in selected:
                    # Split c into U, V, W
                    # Use c^(1/3) if c > 0, or handle negative
                    # For simplicity, assign c to U and 1 to V, W
                    U[i * m + j, r] += c
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += 1.0
            
            if _verify(U, V, W, n, m, p, target_rank):
                return U, V, W, target_rank
    
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    The schoolbook rank is 2*3*4 = 24.
    We attempt to find a lower rank decomposition, but if not found quickly,
    return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try to find a lower rank decomposition
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from 23 down to 18
    for target_rank in range(best_rank - 1, 17, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, max_attempts=100, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": [[float(U[i, r]) for r in range(target_rank)] for i in range(n * m)],
                "V": [[float(V[i, r]) for r in range(target_rank)] for i in range(m * p)],
                "W": [[float(W[i, r]) for r in range(target_rank)] for i in range(p * n)]
            }
            best_rank = target_rank
            break
    
    # Also try the greedy merge approach
    if time.time() < deadline:
        result = _try_greedy_merge(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_result = {
                    "U": [[float(U[i, r]) for r in range(rank)] for i in range(n * m)],
                    "V": [[float(V[i, r]) for r in range(rank)] for i in range(m * p)],
                    "W": [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
                }
    
    return best_result
# EVOLVE-BLOCK-END
