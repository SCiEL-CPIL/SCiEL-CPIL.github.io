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


def _greedy_build(points, all_primes, deadline):
    """
    Greedy construction: start with an initial set, repair violations by removing
    least-populated residue classes, then try to add back elements.
    Returns the admissible set.
    """
    points = list(points)
    points_set = set(points)
    
    # Phase 1: Repair - remove elements to fix violations
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(points)
        for p in all_primes:
            if p > m:
                break
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
                points_set = set(points)
                changed = True
                break
    
    # Phase 2: Local improvement - try to add elements back
    points, points_set = _try_add_elements(points, points_set, all_primes, deadline)
    
    return points


def _try_add_elements(points, points_set, all_primes, deadline):
    """
    Try to add elements to the set using incremental residue tracking.
    Adding candidate c is safe iff for every relevant prime p <= len(points)+1,
    the residue class c%p is not the last missing one.
    """
    m = len(points)
    
    # Build residue count structures
    relevant_primes = [p for p in all_primes if p <= m + 1]
    residue_counts = {}
    nonempty_count = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
        nonempty_count[p] = sum(1 for c in counts if c > 0)
    
    # Precompute candidate list sorted by "safety" - prefer candidates that
    # don't fill new residue classes for many primes
    candidates = [c for c in range(N + 1) if c not in points_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes if m changed
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_count[p] = sum(1 for c in counts if c > 0)
        relevant_primes = new_relevant
        
        # Score candidates: count how many primes would be "tightened"
        # Prefer candidates that don't create new non-empty classes
        scored = []
        for candidate in candidates:
            if time.time() >= deadline:
                break
            tighten_count = 0
            admissible = True
            for p in relevant_primes:
                r = candidate % p
                if residue_counts[p][r] == 0:
                    if nonempty_count[p] + 1 == p:
                        admissible = False
                        break
                    tighten_count += 1
            
            if admissible:
                scored.append((tighten_count, candidate))
        
        # Sort by tighten_count (prefer those that don't tighten)
        scored.sort()
        
        for _, candidate in scored:
            if time.time() >= deadline:
                return points, points_set
            
            # Re-verify admissibility (state may have changed)
            admissible = True
            for p in relevant_primes:
                r = candidate % p
                if residue_counts[p][r] == 0:
                    if nonempty_count[p] + 1 == p:
                        admissible = False
                        break
            
            if admissible:
                # Add the candidate
                points.append(candidate)
                points_set.add(candidate)
                m += 1
                # Update residue counts and nonempty counts
                for p in relevant_primes:
                    r = candidate % p
                    if residue_counts[p][r] == 0:
                        nonempty_count[p] += 1
                    residue_counts[p][r] += 1
                
                # Check if a new prime p == m+1 is now relevant
                if m + 1 not in residue_counts:
                    for p in all_primes:
                        if p == m + 1:
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            nonempty_count[p] = sum(1 for c in counts if c > 0)
                            relevant_primes.append(p)
                            break
                
                improved = True
                # Remove from candidates
                candidates = [c for c in candidates if c != candidate]
                break
    
    return points, points_set


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    # Precompute primes up to N+1
    all_primes = _primes_upto(N + 1)
    
    best_points = []
    best_len = 0
    
    # Multiple initial strategies
    strategies = []
    
    # Strategy 1: All numbers, then repair (gives large initial set)
    strategies.append(list(range(N + 1)))
    
    # Strategy 2: Odd numbers
    strategies.append(list(range(1, N + 1, 2)))
    
    # Strategy 3: Numbers not divisible by small primes (2, 3, 5)
    s3 = [x for x in range(N + 1) if x % 2 != 0 and x % 3 != 0 and x % 5 != 0]
    strategies.append(s3)
    
    # Strategy 4: Numbers coprime to 30 (more aggressive filtering)
    s4 = [x for x in range(N + 1) if math.gcd(x, 30) == 1 or x in (0, 1)]
    strategies.append(s4)
    
    for initial in strategies:
        if time.time() >= deadline - 10:
            break
        result, _ = _try_add_elements(list(initial), set(initial), all_primes, deadline)
        # First repair
        result = _greedy_build(result, all_primes, deadline)
        if len(result) > best_len:
            best_points = result
            best_len = len(result)
    
    # Additional rounds of local improvement on the best
    if best_len > 0 and time.time() < deadline:
        best_points, _ = _try_add_elements(best_points, set(best_points), all_primes, deadline)
        if not _check_admissible(best_points, all_primes):
            best_points = _greedy_build(best_points, all_primes, deadline)
    
    # Final verification and repair if needed
    if best_points and not _check_admissible(best_points, all_primes):
        best_points = _greedy_build(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
