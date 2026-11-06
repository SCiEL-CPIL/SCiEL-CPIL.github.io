import time
import sys

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    deadline = time.time() + 280
    points = list(range(N + 1))
    removed = set()
    
    primes = _primes(N + 1)
    
    # Phase 1: Greedy Removal
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        k = len(points)
        for p in primes:
            if p > k:
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = [v]
                else:
                    classes[r].append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                remove_set = set(smallest)
                points = [v for v in points if v not in remove_set]
                removed.update(remove_set)
                changed = True
                break
    
    # Phase 2: Local Search Expansion
    # We try to add back elements from 'removed'
    # We need to efficiently check admissibility
    # Let's maintain a structure to check if adding x is valid
    
    # For efficiency, we can precompute the residues of all removed elements for all relevant primes
    # But the set of relevant primes changes as k changes.
    
    # A simpler approach: just try to add each removed element and check if the new set is admissible.
    # To check admissibility, we can use the same logic as in Phase 1, but it's O(k * pi(k)).
    # Since k is around 3000-4000, and pi(k) is around 500, this is 1.5M operations per check.
    # With 1000+ removed elements, this is 1.5B operations, which is too slow.
    
    # We need a faster admissibility check.
    # Idea: maintain a count array for each prime p.
    # Let's do it properly.
    
    # Re-initialize points as a set for O(1) lookup
    point_set = set(points)
    
    # We will try to add elements from 'removed'
    # We'll iterate through 'removed' and try to add each one
    # To be efficient, we can sort 'removed' in some order, e.g., by value or by some heuristic
    
    # Let's maintain the residue counts for the current set
    # But the set of primes depends on k.
    # So we'll just recompute the counts when we add an element.
    
    # Actually, a better way: 
    # For each candidate x in removed, check if adding x makes the set non-admissible.
    # This means there exists a prime p <= k+1 such that the new set covers all residues mod p.
    # Since the old set was admissible, the only way adding x breaks admissibility is if:
    # 1. x covers a residue mod p that was previously missing, and now all p residues are covered.
    # 2. This can only happen for primes p where the old set was missing exactly one residue mod p, and x covers that residue.
    
    # So, for the current set S, let's precompute for each prime p <= k+1, the set of missing residues.
    # If a prime p has exactly one missing residue r, and x % p == r, then adding x might break admissibility (if p <= k+1).
    
    # Let's implement this.
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        k = len(point_set)
        
        # Precompute missing residues for each prime p <= k+1
        # missing_residues[p] = set of residues r in [0, p-1] not in {x % p for x in point_set}
        missing_residues = {}
        for p in primes:
            if p > k + 1:
                break
            present = set()
            for x in point_set:
                present.add(x % p)
            missing = set(range(p)) - present
            missing_residues[p] = missing
        
        # Now, try to add each element from 'removed'
        # We'll iterate through a list of removed elements
        removed_list = list(removed)
        # Maybe sort by some heuristic? For now, just iterate.
        for x in removed_list:
            if x in point_set:
                continue
            # Check if adding x is valid
            valid = True
            for p, missing in missing_residues.items():
                if len(missing) == 1:
                    # If x covers the missing residue, and p <= k+1, then adding x would make p fully covered
                    # But we only care if p <= k+1 (new size)
                    if p <= k + 1:
                        if x % p in missing:
                            valid = False
                            break
            if valid:
                point_set.add(x)
                removed.discard(x)
                improved = True
                # Update missing_residues for the new set? 
                # It's easier to just break and recompute in the next iteration of the while loop.
                # Or we can update it incrementally.
                # For simplicity, let's break and recompute.
                break
    
    return list(point_set)
