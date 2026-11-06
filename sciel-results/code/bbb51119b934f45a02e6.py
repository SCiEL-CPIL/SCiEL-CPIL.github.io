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
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if all(c > 0 for c in class_counts):
                # Violation: all p classes covered
                min_count = min(class_counts)
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
    
    # Track non-empty class count for each prime
    nonempty_counts = {}
    for p in relevant_primes:
        nonempty_counts[p] = sum(1 for c in residue_counts[p] if c > 0)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes to include p <= m+1
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
        relevant_primes = new_relevant
        
        # Try candidates
        for candidate in range(N + 1):
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                return points
            
            # Check if adding candidate would cause any prime to cover all residues
            admissible = True
            for p in relevant_primes:
                counts = residue_counts[p]
                ne = nonempty_counts[p]
                r = candidate % p
                if counts[r] == 0:
                    # Adding candidate makes this class non-empty
                    if ne + 1 == p:
                        # All p classes would be covered - violation!
                        admissible = False
                        break
                # If counts[r] > 0, nonempty count doesn't change, so no new violation
            # Note: if nonempty_before < p-1, then even if we add a new class, nonempty_after < p
            # If nonempty_before == p-1 and counts[r]==0, then nonempty_after == p -> violation
            # If nonempty_before == p-1 and counts[r]>0, nonempty_after == p-1 -> ok
            # If nonempty_before < p-1, nonempty_after <= p-1 -> ok
            
            if admissible:
                # Add the candidate
                points.append(candidate)
                points_set.add(candidate)
                m += 1
                # Update residue counts and nonempty counts
                for p in relevant_primes:
                    r = candidate % p
                    if residue_counts[p][r] == 0:
                        nonempty_counts[p] += 1
                    residue_counts[p][r] += 1
                # Add new primes if m+1 is a prime
                if m + 1 <= N + 1:
                    for p in all_primes:
                        if p == m + 1:
                            # New prime to track
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            nonempty_counts[p] = sum(1 for c in counts if c > 0)
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
    
    # Strategy 1: Start from all integers and repair
    points_dense = list(range(N + 1))
    points_dense = _greedy_repair(points_dense, all_primes, deadline)
    if len(points_dense) > len(best_points):
        best_points = points_dense
    
    # Strategy 2: Start from odd numbers
    if time.time() < deadline - 30:
        points_odd = list(range(1, N + 1, 2))
        points_odd = _greedy_repair(points_odd, all_primes, deadline)
        if len(points_odd) > len(best_points):
            best_points = points_odd
    
    # Strategy 3: Start from multiples of 3
    if time.time() < deadline - 30:
        points_mult3 = list(range(0, N + 1, 3))
        points_mult3 = _greedy_repair(points_mult3, all_primes, deadline)
        if len(points_mult3) > len(best_points):
            best_points = points_mult3

    # Local improvement: try to add back elements using fast method
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(best_points, all_primes):
        best_points = _greedy_repair(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
