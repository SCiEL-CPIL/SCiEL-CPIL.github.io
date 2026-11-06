import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to 1000 (sufficient for tuples of size ~450)
MAX_PRIME_FOR_PRECOMP = 1000
ALL_PRIMES = _primes(MAX_PRIME_FOR_PRECOMP)

# Precompute residue class indices for fast lookup
prime_indices = {}
for p in ALL_PRIMES:
    indices = [[] for _ in range(p)]
    for x in range(N + 1):
        r = x % p
        indices[r].append(x)
    prime_indices[p] = indices

def _is_admissible(S_active):
    """Check if the active set is admissible."""
    size = int(S_active.sum())
    # For every prime p <= size, some residue class must be missed
    for p in ALL_PRIMES:
        if p > size:
            break
        # Check if all residue classes are present
        all_present = True
        for r in range(p):
            # Check if any active element in this residue class
            for x in prime_indices[p][r]:
                if S_active[x]:
                    break
            else:
                # No active element in this residue class
                all_present = False
                break
        if all_present:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = 0
    best_set = []
    
    # Precompute active masks for each prime and residue
    # active_mask[p][r] = numpy boolean array where True if x is active and x % p == r
    # This is expensive to compute, so we'll compute on the fly using prime_indices
    
    # Strategy: randomized greedy with local search refinement
    
    # First pass: multiple randomized greedy attempts to find good initial solutions
    num_greedy_tries = 20
    greedy_results = []
    
    for trial in range(num_greedy_tries):
        if time.time() > deadline - 60:
            break
            
        S_active = np.ones(N + 1, dtype=bool)
        current_size = N + 1
        
        # Process primes in a random order (or sorted by value)
        # For admissibility, we need to ensure each prime p <= size misses a residue
        # Greedy: process primes in increasing order, remove min-count residue
        
        # Use a priority approach: for each prime, compute counts and remove min
        for p in ALL_PRIMES:
            if p > current_size:
                break
            
            # Calculate counts for each residue
            counts = [0] * p
            for r in range(p):
                cnt = 0
                for x in prime_indices[p][r]:
                    if S_active[x]:
                        cnt += 1
                counts[r] = cnt
            
            min_count = min(counts)
            
            if min_count > 0:
                # All residues are present; remove one
                # Randomly select among min-count residues
                candidates = [r for r in range(p) if counts[r] == min_count]
                r_to_remove = random.choice(candidates)
                
                for x in prime_indices[p][r_to_remove]:
                    if S_active[x]:
                        S_active[x] = False
                        current_size -= 1
        
        greedy_results.append((current_size, S_active.copy()))
        
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
    
    # Sort greedy results by size, best first
    greedy_results.sort(key=lambda x: -x[0])
    
    # Local search refinement on top greedy results
    # Move: remove an element (if admissibility still holds) or add an element (if admissibility restored)
    # We'll try to add elements back to increase size
    
    local_search_budget = 240.0
    local_search_start = time.time()
    
    for size, S_active in greedy_results[:10]:
        if time.time() - local_search_start > local_search_budget:
            break
        if size < best_size:
            continue
        
        S_active = S_active.copy()
        current_size = size
        
        # Try to add elements back
        # For each inactive element, check if adding it preserves admissibility
        inactive_elements = [x for x in range(N + 1) if not S_active[x]]
        random.shuffle(inactive_elements)
        
        # Try adding elements in random order
        added = 0
        for x in inactive_elements:
            if time.time() - local_search_start > local_search_budget:
                break
            
            # Tentatively add x
            S_active[x] = True
            new_size = current_size + 1
            
            # Check admissibility
            if _is_admissible(S_active):
                current_size = new_size
                added += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = [int(x) for x in range(N + 1) if S_active[x]]
            else:
                S_active[x] = False
        
        # Also try removing elements to free up room for additions? 
        # Actually, we want to maximize size, so adding is the goal.
        # But sometimes removing one element allows adding multiple others.
        # This is complex; for now, stick with simple add-only local search.
        
    return sorted(best_set)
