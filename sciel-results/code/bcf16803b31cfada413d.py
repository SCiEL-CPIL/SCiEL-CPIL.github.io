"""Rewritten program: fixed-diameter admissible tuple maximization in [0, 5000]."""
import time
import math
from collections import defaultdict

N = 5000
DEADLINE = 300.0  # wall-clock limit with safety margin (evaluator kills at 330s)


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _build_admissible_from_seed(seed_points, primes_to_check, deadline):
    """
    Starting from a set of points, iteratively remove the least-populated residue
    class for any prime p (among primes_to_check) where all p classes are covered.
    Uses efficient counter updates. Returns the final admissible set.
    """
    # Work with a set for O(1) membership
    points = set(seed_points)
    # Precompute for each prime p the residue class of each point
    # We'll maintain: for each p, a dict residue -> count, and residue -> list of points
    # But rebuilding lists is expensive. Instead, maintain counts and a mapping.
    
    # Strategy: for each prime p, maintain count[r] for r in 0..p-1.
    # When we remove a point x, for each p, decrement count[x % p].
    # When count[r] hits 0, that residue is now missed -> p is satisfied.
    # We need to find primes where all residues are still covered (count[r] > 0 for all r).
    
    # Efficient approach:
    # For each p, track how many residues have count 0 (i.e., are missed).
    # Initially compute counts. A prime is "violated" if missed_count == 0.
    # When we remove a point, for each p, if count[x%p] was 1, it becomes 0,
    #   so missed_count[p] increments.
    # We want to remove points to make all primes have missed_count >= 1.
    
    # Let's precompute the primes we need to check: primes <= len(points)
    # But len(points) changes. The task says "for every prime p <= len(tuple)".
    # So we need primes up to the final size. Since we're removing, size decreases.
    # We'll use primes up to initial size (conservative).
    
    max_p = len(seed_points)
    primes = [p for p in primes_to_check if p <= max_p]
    
    if not primes:
        return list(points)
    
    # For each prime, maintain:
    #   counts[p][r] = number of points in residue class r mod p
    #   missed[p] = number of residue classes with count 0
    #   Also need: for each prime, if missed[p] == 0, which residue class to remove?
    #   We want to remove the point from the least-populated covered class.
    
    # Precompute counts
    counts = {}  # p -> list of length p
    missed = {}  # p -> int
    # Also: for each point, its residues mod each prime (for fast update on removal)
    point_residues = {}  # point -> list of (p, r) for each p in primes
    
    for p in primes:
        c = [0] * p
        for x in points:
            c[x % p] += 1
        counts[p] = c
        m = sum(1 for r in range(p) if c[r] == 0)
        missed[p] = m
    
    # For each point, precompute its residue for each prime
    for x in points:
        point_residues[x] = [(p, x % p) for p in primes]
    
    # Find violated primes (missed == 0)
    violated = [p for p in primes if missed[p] == 0]
    
    while violated:
        if time.time() > deadline - 2.0:
            break
        
        # Pick a violated prime to fix. Choose the one with smallest min count
        # to minimize damage.
        best_p = None
        best_min_count = float('inf')
        best_r = None
        for p in violated:
            c = counts[p]
            # Find min count among residues (all are > 0 since missed == 0)
            min_c = min(c)
            if min_c < best_min_count:
                best_min_count = min_c
                best_p = p
                best_r = c.index(min_c)
        
        if best_p is None:
            break
        
        p = best_p
        r = best_r
        c = counts[p]
        
        # Find a point in residue class r mod p to remove
        # We need to find one. Scan points.
        removed = None
        for x in list(points):
            if x % p == r:
                removed = x
                break
        
        if removed is None:
            # Shouldn't happen, but skip
            violated.remove(p)
            continue
        
        # Remove 'removed' from points
        points.discard(removed)
        
        # Update counts and missed for all primes
        for (q, rq) in point_residues[removed]:
            counts[q][rq] -= 1
            if counts[q][rq] == 0:
                missed[q] += 1
            # Check if q was violated and is now satisfied
            if q in violated and missed[q] > 0:
                violated.remove(q)
        
        # Check if any prime became violated (shouldn't happen since we only decrease)
        # Actually, removing can only increase missed counts, so no new violations.
        # But wait: we need to re-check if the prime we just fixed is still violated?
        # No: we removed one point from class r, so count[r] decreased by 1.
        # If count[r] was 1, it's now 0, so missed increased by 1 -> p is now satisfied.
        # If count[r] was > 1, it's still > 0, so missed didn't change -> p still violated.
        # In the latter case, we need to remove more from class r.
        # Our loop continues, so we'll pick p again if still violated.
        
        # Re-check: is p still violated?
        if missed[p] == 0 and p not in violated:
            violated.append(p)
    
    return list(points)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # safety margin
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    best_result = []
    best_size = 0
    
    # Strategy: Try multiple seed sets and pick the best.
    # Seed 1: All numbers in [0, N]
    # Seed 2: Primes > sqrt(N) plus 1 (original approach)
    # Seed 3: Random subsets
    # Seed 4: Numbers with certain properties
    
    # Approach: Start with a large set and greedily remove.
    # Key insight: We want to maximize the final size.
    # The constraint is: for each prime p <= k (where k is final size), some residue mod p is missed.
    
    # Let's try a greedy removal approach:
    # 1. Start with all integers in [0, N].
    # 2. For each prime p (in some order), if all p residues are covered,
    #    remove the least-populated residue class.
    # 3. Repeat until admissible.
    
    # But the order matters. Let's try different orders.
    
    # First, try the straightforward approach: start with all points.
    all_points = list(range(N + 1))
    result = _build_admissible_from_seed(all_points, all_primes, deadline)
    best_size = len(result)
    best_result = result
    
    if time.time() > deadline - 10.0:
        return best_result
    
    # Try starting with a subset: only numbers that are "hard" to remove.
    # Actually, let's try a different strategy:
    # Greedy construction: build the set incrementally, ensuring admissibility.
    # Or: try removing primes in different orders.
    
    # Let's try: for each permutation of small primes, do the removal.
    # But that's too many. Instead, try a few heuristic orders.
    
    # Order 1: Smallest primes first
    # Order 2: Largest primes first
    # Order 3: Random orders
    
    # The _build_admissible_from_seed already handles the logic.
    # Let's try with different initial sets.
    
    # Seed: only even numbers (will miss odd residues for p=2)
    even_points = list(range(0, N + 1, 2))
    if time.time() < deadline - 5.0:
        result2 = _build_admissible_from_seed(even_points, all_primes, deadline)
        if len(result2) > best_size:
            best_size = len(result2)
            best_result = result2
    
    # Seed: numbers coprime to 2*3*5 (i.e., not divisible by 2, 3, or 5)
    if time.time() < deadline - 5.0:
        seed3 = [x for x in range(N + 1) if x % 2 != 0 and x % 3 != 0 and x % 5 != 0]
        result3 = _build_admissible_from_seed(seed3, all_primes, deadline)
        if len(result3) > best_size:
            best_size = len(result3)
            best_result = result3
    
    # Try a randomized local search: start from best_result, try adding points back
    # if they don't violate admissibility.
    if time.time() < deadline - 10.0:
        current = set(best_result)
        # Try to add points from [0, N] that aren't in current
        candidates = [x for x in range(N + 1) if x not in current]
        import random
        random.seed(42)
        random.shuffle(candidates)
        
        # For each candidate, check if adding it preserves admissibility
        # This is expensive. Instead, try a batch approach.
        
        # Alternative: try swapping - remove one point, add two others.
        # This is complex. Let's just try adding one at a time.
        
        # Precompute primes up to len(current) + 10
        max_possible = len(current) + 100
        check_primes = [p for p in all_primes if p <= max_possible]
        
        added_any = True
        while added_any and time.time() < deadline - 5.0:
            added_any = False
            for x in candidates[:]:
                # Check if adding x keeps the set admissible
                # New size would be len(current) + 1
                new_size = len(current) + 1
                # Need to check primes <= new_size
                relevant_primes = [p for p in check_primes if p <= new_size]
                
                ok = True
                for p in relevant_primes:
                    # Count residues of (current + {x}) mod p
                    # If all p residues are covered, we need to check if this was already
                    # the case before adding x. If it was already covered, adding x doesn't
                    # change that. If it was NOT covered (some residue missed), adding x might
                    # cover it, making it violated.
                    # So: for each p, if there was a missed residue, check if x fills it.
                    
                    # Compute residues of current mod p
                    res_counts = [0] * p
                    for y in current:
                        res_counts[y % p] += 1
                    missed_res = [r for r in range(p) if res_counts[r] == 0]
                    if missed_res:
                        # There's at least one missed residue
                        # Adding x covers residue x % p
                        if x % p in missed_res:
                            # This fills one missed residue, but there might be others
                            # We need ALL residues to be covered for violation
                            # So if there are still other missed residues after adding x,
                            # p is still satisfied.
                            # Only if x fills the LAST missed residue does p become violated.
                            if len(missed_res) == 1 and x % p == missed_res[0]:
                                ok = False
                                break
                    # If no missed residues, p was already violated before adding x.
                    # Adding x doesn't fix that. So p is still violated -> not admissible.
                    # Wait, that means the current set is NOT admissible?
                    # No: if all residues are covered, p is violated.
                    # But our current set should be admissible.
                    # So for all p <= len(current), there should be at least one missed residue.
                    # So missed_res should be non-empty for all relevant p.
                    # But we're checking p <= new_size, which includes p = new_size
                    # which might not have been checked before.
                    
                if ok:
                    current.add(x)
                    candidates.remove(x)
                    added_any = True
                    # Update best
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
        
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
    
    return best_result
