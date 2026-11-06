"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes_upto(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _check_admissible(points, all_primes):
    """Check if points is admissible for all primes p <= len(points)."""
    m = len(points)
    for p in all_primes:
        if p > m:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _greedy_repair(points, all_primes, deadline):
    """
    Greedy repair: repeatedly find a prime p <= len(points) whose residue classes
    are all covered, and remove the least-populated class. Continue until admissible
    or deadline.
    """
    points = list(points)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(points)
        for p in all_primes:
            if p > m:
                break
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) == p:
                # Violation: all p classes covered
                min_count = min(class_counts.values())
                to_remove = set()
                for v in points:
                    if class_counts[v % p] == min_count:
                        to_remove.add(v)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _local_improvement_fast(points, all_primes, deadline):
    """
    Fast local improvement: try to add elements back using incremental residue tracking.
    We maintain residue counts for each relevant prime and only check if adding a candidate
    would violate admissibility.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    # Precompute relevant primes (p <= m+1, since adding one element makes m+1)
    relevant_primes = [p for p in all_primes if p <= m + 1]
    
    # Build residue count structures for current points
    # For each prime p, track count of each residue class
    residue_counts = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        relevant_primes = [p for p in all_primes if p <= m + 1]
        
        # Rebuild residue counts if primes list changed
        for p in relevant_primes:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
        
        # Try candidates in a smart order: start from those that are "safe"
        # Try all candidates, but break early on deadline
        for candidate in range(N + 1):
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                return points
            
            # Check if adding candidate would cause any prime to cover all residues
            admissible = True
            for p in relevant_primes:
                r = candidate % p
                if residue_counts[p][r] == 0:
                    # This residue class was empty, adding it fills it
                    # Check if all other classes are non-empty
                    all_nonempty = True
                    for rr in range(p):
                        if rr != r and residue_counts[p][rr] == 0:
                            all_nonempty = False
                            break
                    if all_nonempty:
                        admissible = False
                        break
                # If residue was already non-empty, adding one more doesn't make it "full"
                # in a way that matters - we only care if ALL p classes become non-empty
            # Actually, the condition is: after adding, do all p residue classes have count >= 1?
            # We need to check: for each p, is it the case that BEFORE adding, some class was 0,
            # and AFTER adding, that class becomes 1, and all others are already >= 1?
            # More precisely: after adding, len({v % p for v in points + [candidate]}) == p?
            # That happens iff before adding, the set of residues was of size p-1, 
            # and candidate fills the missing one.
            
            # Let me redo this more carefully
            admissible = True
            for p in relevant_primes:
                counts = residue_counts[p]
                # Number of non-empty residue classes before
                nonempty_before = sum(1 for c in counts if c > 0)
                r = candidate % p
                if counts[r] == 0:
                    # Adding candidate makes this class non-empty
                    nonempty_after = nonempty_before + 1
                else:
                    nonempty_after = nonempty_before
                
                if nonempty_after == p:
                    # All p classes would be covered - violation!
                    admissible = False
                    break
            
            if admissible:
                # Add the candidate
                points.append(candidate)
                points_set.add(candidate)
                m += 1
                # Update residue counts
                for p in relevant_primes:
                    residue_counts[p][candidate % p] += 1
                # Add new primes if m+1 is a prime
                if m + 1 <= N + 1:
                    for p in all_primes:
                        if p == m + 1:
                            # New prime to track
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            relevant_primes.append(p)
                            break
                improved = True
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    # Precompute primes up to N+1
    all_primes = _primes_upto(N + 1)

    best_points = []
    
    # Strategy 1: Start from a sparse set (1 and primes > sqrt(N)) and repair
    points_sparse = [1] + [p for p in all_primes if p * p > N]
    points_sparse = _greedy_repair(points_sparse, all_primes, deadline)
    if len(points_sparse) > len(best_points):
        best_points = points_sparse
    
    # Strategy 2: Start from all integers and repair
    if time.time() < deadline - 30:
        points_dense = list(range(N + 1))
        points_dense = _greedy_repair(points_dense, all_primes, deadline)
        if len(points_dense) > len(best_points):
            best_points = points_dense
    
    # Strategy 3: Start from evens + 1 (odd numbers)
    if time.time() < deadline - 30:
        points_odd = list(range(1, N + 1, 2))
        points_odd = _greedy_repair(points_odd, all_primes, deadline)
        if len(points_odd) > len(best_points):
            best_points = points_odd

    # Local improvement: try to add back elements using fast method
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(best_points, all_primes):
        best_points = _greedy_repair(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
