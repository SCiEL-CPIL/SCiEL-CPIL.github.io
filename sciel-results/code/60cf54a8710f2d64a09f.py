import math
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    limit_time = start_time + 290.0  # 10s margin before 300s limit
    
    # Start with all points in [0, N]
    active_points = list(range(N + 1))
    active_set = set(active_points)
    removed_points = set()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Iterate through each prime in increasing order
    for p in primes:
        if time.time() > limit_time:
            break
        
        # Early termination: if p > len(active_points), no residue class mod p
        # can be fully covered, so we can stop
        if p > len(active_points):
            break
        
        # Count active points in each residue class mod p
        residue_counts = [0] * p
        for x in active_points:
            residue_counts[x % p] += 1
        
        # Check if all classes are non-empty
        if min(residue_counts) > 0:
            # Find the class with the fewest elements
            min_idx = min(range(p), key=lambda i: residue_counts[i])
            
            # Deactivate all points in the smallest class
            # Build set of points to remove for fast lookup
            to_remove = set()
            for x in active_points:
                if x % p == min_idx:
                    to_remove.add(x)
            
            # Remove those points
            active_points = [x for x in active_points if x not in to_remove]
            active_set -= to_remove
            removed_points.update(to_remove)

    # Local search expansion phase: try to add back removed points
    # that are compatible with the final set's admissibility constraints
    if time.time() < limit_time - 5.0:
        # Sort removed points for deterministic ordering
        candidates = sorted(removed_points)
        
        # For each candidate, check if adding it back maintains admissibility
        # We need to check: for every prime p <= len(active_points) + 1,
        # some residue class mod p is still missed after adding the candidate
        for x in candidates:
            if time.time() > limit_time - 2.0:
                break
            
            # Check if adding x back would violate admissibility
            # The new set would be active_set ∪ {x}
            # For each prime p <= len(active_set) + 1, we need some residue class
            # mod p that has no elements in the new set
            
            can_add = True
            new_size = len(active_set) + 1
            
            # Only need to check primes up to new_size
            for p in primes:
                if p > new_size:
                    break
                
                # Count residues in active_set for this prime
                residue_present = [False] * p
                for y in active_set:
                    residue_present[y % p] = True
                
                # Check if x fills in a gap
                if not residue_present[x % p]:
                    # x would fill this residue class, but we need to check if
                    # there's still some other empty residue class
                    # Actually, we need to check: after adding x, is there still
                    # some residue class mod p that's empty?
                    # If x was the only one that could fill this gap, and there
                    # are no other gaps, then adding x would violate admissibility
                    empty_count = sum(1 for r in range(p) if not residue_present[r])
                    if empty_count == 1:
                        # Adding x would fill the last empty class -> violation
                        can_add = False
                        break
                    # Otherwise, there are other empty classes, so it's fine
                # If x % p is already present, adding x doesn't change anything
                # for this prime's coverage
            
            if can_add:
                active_set.add(x)
                active_points.append(x)
                removed_points.discard(x)

    return sorted(active_points)
# EVOLVE-BLOCK-END
