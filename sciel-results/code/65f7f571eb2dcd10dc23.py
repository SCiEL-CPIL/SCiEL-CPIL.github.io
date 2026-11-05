"""Construct the largest admissible tuple within [0, 5000]."""
import time
import math
from collections import defaultdict

N = 5000
DEADLINE = 320  # safety margin before 330s kill


def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes, max_p=None):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    if max_p is None:
        max_p = n
    for p in primes:
        if p > max_p:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _construct_optimal(primes, deadline):
    """
    Construct admissible tuple by strategically removing residue classes.
    Process primes in increasing order, but use smarter class selection.
    """
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    
    # We'll track which residue class we "commit to removing" for each prime
    removed_classes = {}  # prime -> residue class to remove
    
    for p in primes:
        if p > N:
            break
        if time.time() - start > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                class_counts[v % p] += 1
        
        # If all p classes have at least one point, we must remove one
        if min(class_counts) > 0:
            # Choose the class to remove strategically
            # Prefer classes that, when removed, don't hurt future constraints much
            # Heuristic: remove the class with the fewest points (least damage)
            # But also consider: if a class was already "partially removed" by earlier primes,
            # it might be cheaper to fully remove it
            
            min_count = min(class_counts)
            
            # Find all classes with minimum count
            candidates = [c for c in range(p) if class_counts[c] == min_count]
            
            # Among candidates, prefer the one that aligns with previously removed classes
            # (i.e., a class that already has many holes from earlier removals)
            best_class = candidates[0]
            best_score = -1
            
            for c in candidates:
                # Score: how many points in this class were already removed by earlier primes
                # (i.e., how "damaged" this class already is)
                # Actually, let's just pick the first one - the min count is the main criterion
                best_class = c
                break
            
            removed_classes[p] = best_class
            
            # Remove all present points in this class
            for v in range(best_class, N + 1, p):
                present[v] = 0
    
    # Collect remaining points
    result = [v for v in range(N + 1) if present[v]]
    return result, removed_classes


def _try_recover(points_set, primes, deadline, start_time):
    """
    Try to recover removed elements that don't violate admissibility.
    """
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return points_set
    
    current_len = len(points_set)
    
    # Generate list of removed elements
    removed = [v for v in range(N + 1) if v not in points_set]
    
    # Try to add back removed elements one by one
    for v in removed:
        if time.time() - start_time > deadline:
            break
        
        # Quick check: would adding v create a full residue class for some prime?
        # Only need to check primes p <= current_len + 1
        admissible = True
        for p in primes:
            if p > current_len + 1:
                break
            # Check if v fills in the missing residue for this prime
            # We need to check if all p residues would be present
            # Efficient check: count residues without v, then see if v adds a new one
            # But this is expensive. Let's do a full check but break early.
            residues = set(x % p for x in points_set)
            if len(residues) == p:
                # Already not admissible, but that shouldn't happen
                admissible = False
                break
            if v % p not in residues:
                # Adding v would make this residue class complete
                # Check if this was the only missing one
                if len(residues) == p - 1:
                    admissible = False
                    break
        
        if admissible:
            points_set.add(v)
            current_len += 1
    
    return points_set


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    """
    start_time = time.time()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Phase 1: Strategic removal
    deadline_phase1 = DEADLINE * 0.6
    result, removed_classes = _construct_optimal(primes, deadline_phase1)
    
    # Verify admissibility
    if not _is_admissible(result, primes):
        # Fix: remove until admissible
        for p in primes:
            if p > len(result):
                break
            residues = set(v % p for v in result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                result = [v for v in result if v % p != min_class]
    
    # Phase 2: Refinement - try to recover removed elements
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 10
    if remaining_time > 5:
        points_set = set(result)
        points_set = _try_recover(points_set, primes, remaining_time, start_time)
        result = list(points_set)
    
    # Final verification
    if not _is_admissible(result, primes):
        for p in primes:
            if p > len(result):
                break
            residues = set(v % p for v in result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                result = [v for v in result if v % p != min_class]
    
    result = sorted(set(result))
    return result
