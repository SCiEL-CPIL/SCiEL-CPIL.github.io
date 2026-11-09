"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = math.log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / denominator

def exhaustive_search_small(n, max_val=50, deadline=None):
    """Exhaustively search for sets of size n with small values."""
    best_set = None
    best_score = 0.0
    
    # Generate all combinations of n-1 values from 1 to max_val (0 is always included)
    for combo in combinations(range(1, max_val + 1), n - 1):
        if deadline and time.time() > deadline:
            break
        U = [0] + list(combo)
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    return best_set, best_score

def local_search(U, max_steps=1000, deadline=None):
    """Local search to improve a set U."""
    U = sorted(set(U))
    best_score = compute_score(U)
    best_set = U[:]
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try adding a new element
        max_u = max(U)
        for new_val in range(1, max_u + 20):
            if new_val not in U:
                U_new = sorted(U + [new_val])
                score = compute_score(U_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = U_new
                    U = U_new
                    improved = True
                    break
        
        # Try removing an element (if size > 2)
        if len(U) > 2:
            for i in range(1, len(U)):  # Don't remove 0
                U_new = U[:i] + U[i+1:]
                score = compute_score(U_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = U_new
                    U = U_new
                    improved = True
                    break
        
        # Try replacing an element
        for i in range(1, len(U)):
            old_val = U[i]
            for new_val in range(1, max_u + 30):
                if new_val == old_val or new_val in U:
                    continue
                U_new = sorted(U[:i] + [new_val] + U[i+1:])
                score = compute_score(U_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = U_new
                    U = U_new
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_set, best_score

def geometric_construction(k):
    """Construct a geometric-like set {0, 1, r, r^2, ...}."""
    r = 2
    U = [0, 1]
    for i in range(2, k):
        U.append(U[-1] * r)
    return U

def mixed_construction(k, r1=3, r2=2):
    """Construct a mixed geometric set."""
    U = [0, 1]
    for i in range(2, k):
        if i % 2 == 0:
            U.append(U[-1] * r1)
        else:
            U.append(U[-1] * r2)
    return U

def construct():
    """Main construct function."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with known good small sets
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 7],
        [0, 1, 4, 13],
    ]
    
    best_set = None
    best_score = 0.0
    
    for U in candidates:
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # Try exhaustive search for small sizes
    for n in range(3, 7):
        if time.time() > deadline:
            break
        set_found, score = exhaustive_search_small(n, max_val=30, deadline=deadline)
        if set_found and score > best_score:
            best_score = score
            best_set = set_found
    
    # Try local search from various starting points
    if best_set and time.time() < deadline:
        set_found, score = local_search(best_set, max_steps=500, deadline=deadline)
        if score > best_score:
            best_score = score
            best_set = set_found
    
    # Try geometric constructions
    for k in range(3, 10):
        if time.time() > deadline:
            break
        for r in [2, 3, 4, 5]:
            U = [0, 1]
            for i in range(2, k):
                U.append(U[-1] * r)
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
        
        # Local search on geometric
        if time.time() < deadline:
            set_found, score = local_search(best_set, max_steps=200, deadline=deadline)
            if score > best_score:
                best_score = score
                best_set = set_found
    
    # Try mixed constructions
    for k in range(3, 10):
        if time.time() > deadline:
            break
        for r1 in [2, 3, 4]:
            for r2 in [2, 3]:
                if r1 == r2:
                    continue
                U = mixed_construction(k, r1, r2)
                score = compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = U[:]
    
    # Final local search polish
    if best_set and time.time() < deadline:
        set_found, score = local_search(best_set, max_steps=300, deadline=deadline)
        if score > best_score:
            best_score = score
            best_set = set_found
    
    return best_set if best_set else [0, 1, 3]
# EVOLVE-BLOCK-END
