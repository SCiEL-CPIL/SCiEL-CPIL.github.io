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


def _construct_greedy(primes, deadline, tie_break='min_count'):
    """
    Construct admissible tuple using greedy removal with different tie-breaking strategies.
    
    tie_break: 'min_count' - remove class with fewest points
               'max_damage' - remove class most aligned with previous removals
               'random' - random tie-breaking
    """
    import random
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    removed_classes = {}
    
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
            min_count = min(class_counts)
            candidates = [c for c in range(p) if class_counts[c] == min_count]
            
            if tie_break == 'min_count':
                best_class = candidates[0]
            elif tie_break == 'max_damage':
                # Prefer classes that already have holes from earlier primes
                best_class = candidates[0]
                best_damage = -1
                for c in candidates:
                    damage = 0
                    for prev_p, prev_c in removed_classes.items():
                        # Count how many numbers in class c mod p were already removed by prev_p
                        # A number v is in class c mod p and class prev_c mod prev_p if
                        # v ≡ c (mod p) and v ≡ prev_c (mod prev_p)
                        # By CRT, this happens with density 1/(p*prev_p) if gcd(p,prev_p)=1
                        # For primes, gcd is always 1, so density is 1/(p*prev_p)
                        # Expected damage contribution: (N+1)/(p*prev_p)
                        damage += (N + 1) / (p * prev_p)
                    if damage > best_damage:
                        best_damage = damage
                        best_class = c
            elif tie_break == 'random':
                best_class = random.choice(candidates)
            else:
                best_class = candidates[0]
            
            removed_classes[p] = best_class
            
            # Remove all present points in this class
            for v in range(best_class, N + 1, p):
                present[v] = 0
    
    result = [v for v in range(N + 1) if present[v]]
    return result


def _refine(points, primes, deadline, start_time):
    """
    Try to recover removed elements that don't violate admissibility.
    """
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return points
    
    points_set = set(points)
    current_len = len(points_set)
    
    # Generate list of removed elements
    removed = [v for v in range(N + 1) if v not in points_set]
    
    # Try to add back removed elements one by one
    for v in removed:
        if time.time() - start_time > deadline:
            break
        
        # Quick check: would adding v create a full residue class for some prime?
        admissible = True
        for p in primes:
            if p > current_len + 1:
                break
            residues = set(x % p for x in points_set)
            if len(residues) == p - 1 and v % p not in residues:
                admissible = False
                break
        
        if admissible:
            points_set.add(v)
            current_len += 1
    
    return list(points_set)


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    """
    start_time = time.time()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Try multiple greedy strategies with different tie-breaking
    strategies = ['min_count', 'max_damage', 'random']
    best_result = []
    
    # Allocate time: 60% for construction, 40% for refinement
    construct_deadline = DEADLINE * 0.6
    per_strategy_time = construct_deadline / len(strategies)
    
    for strategy in strategies:
        elapsed = time.time() - start_time
        if elapsed > construct_deadline:
            break
        
        strategy_deadline = min(per_strategy_time, construct_deadline - elapsed)
        if strategy_deadline < 1:
            break
        
        result = _construct_greedy(primes, strategy_deadline, tie_break=strategy)
        
        # Verify admissibility
        if _is_admissible(result, primes):
            if len(result) > len(best_result):
                best_result = result
        else:
            # Fix admissibility
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
            if len(result) > len(best_result):
                best_result = result
    
    # Refinement phase
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 10
    if remaining_time > 5 and best_result:
        best_result = _refine(best_result, primes, remaining_time, start_time)
    
    # Final verification
    if best_result and not _is_admissible(best_result, primes):
        for p in primes:
            if p > len(best_result):
                break
            residues = set(v % p for v in best_result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in best_result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                best_result = [v for v in best_result if v % p != min_class]
    
    return sorted(set(best_result))
