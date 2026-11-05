"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random
import numpy as np
from collections import defaultdict

N = 5000
DEADLINE = 310  # safety margin before 330s kill


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


def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(primes, order="increasing", deadline=300):
    """
    Greedy removal: start with all points [0, N]. For each prime p,
    if all p residue classes are still present, remove the least-populated class.
    """
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    
    if order == "increasing":
        prime_list = primes
    elif order == "decreasing":
        prime_list = list(reversed(primes))
    else:
        prime_list = list(primes)
        random.shuffle(prime_list)
    
    for p in prime_list:
        if p > N:
            break
        if time.time() - start > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                class_counts[v % p] += 1
        
        # If all p classes have at least one point, remove the least populated class
        if min(class_counts) > 0:
            min_count = min(class_counts)
            target_class = class_counts.index(min_count)
            for v in range(target_class, N + 1, p):
                present[v] = 0
    
    result = [v for v in range(N + 1) if present[v]]
    return result


def _local_search_refine(points, primes, deadline=300, num_iterations=50):
    """
    Local search: try to add back removed elements using smarter criteria.
    For each candidate element v, compute the "cost" of adding it (number of primes
    for which adding v would create a conflict). Add if cost is 0.
    Then do swap operations: remove one element and add another if net gain > 0.
    """
    start = time.time()
    if not points:
        return points
    
    points_set = set(points)
    current_len = len(points)
    
    # Precompute for each prime p <= N, the residue counts
    # For fast admissibility checking
    
    # Try multiple rounds of adding elements
    for iteration in range(num_iterations):
        if time.time() - start > deadline:
            break
        
        removed = [v for v in range(N + 1) if v not in points_set]
        random.shuffle(removed)
        
        improved = False
        for v in removed:
            if time.time() - start > deadline:
                break
            
            new_len = current_len + 1
            
            # Check admissibility for all primes p <= new_len
            admissible = True
            for p in primes:
                if p > new_len:
                    break
                # Quick check: does v fill a missing residue class?
                # If so, we might create a conflict
                residues_present = set(x % p for x in points_set)
                if v % p not in residues_present:
                    # Adding v creates a new residue, which might fill the last missing one
                    if len(residues_present) == p - 1:
                        admissible = False
                        break
                # If v's residue is already present, no new residue created, so safe
                # for this prime
            else:
                # All primes checked, admissible
                points_set.add(v)
                current_len += 1
                improved = True
        
        if not improved:
            # Try swap operations
            for _ in range(100):
                if time.time() - start > deadline:
                    break
                
                removed = [v for v in range(N + 1) if v not in points_set]
                if not removed:
                    break
                
                # Try to find a swap: remove one element from points_set,
                # add one from removed, if result is admissible and same size
                # (but this doesn't help size). Instead, try to add multiple at once.
                # For now, just break if no improvement.
                break
        
        if not improved:
            break
    
    return list(points_set)


def _simulated_annealing(points, primes, deadline=300):
    """
    Simulated annealing: start with a valid admissible tuple, then try to add
    elements. If adding violates admissibility, temporarily remove one element
    to make room, then try to add the new one. Accept based on a cooling schedule.
    """
    start = time.time()
    if not points:
        return points
    
    points_set = set(points)
    current_len = len(points)
    best_set = set(points_set)
    best_len = current_len
    
    temperature = 1.0
    cooling = 0.95
    
    # Precompute primes list
    relevant_primes = [p for p in primes if p <= N]
    
    removed = [v for v in range(N + 1) if v not in points_set]
    
    while time.time() - start < deadline:
        if temperature < 0.01:
            break
        
        # Pick a random element to try adding
        if not removed:
            break
        v = random.choice(removed)
        
        new_len = current_len + 1
        
        # Check if adding v keeps admissibility
        admissible = True
        for p in relevant_primes:
            if p > new_len:
                break
            residues = set(x % p for x in points_set)
            if v % p not in residues:
                if len(residues) == p - 1:
                    admissible = False
                    break
        else:
            # Admissible, add it
            points_set.add(v)
            removed.remove(v)
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_set = set(points_set)
        
        # Also try removing a random element from points_set to allow more additions
        if points_set:
            # With some probability, remove a random element
            if random.random() < 0.3:
                to_remove = random.choice(list(points_set))
                points_set.remove(to_remove)
                removed.append(to_remove)
                current_len -= 1
        
        temperature *= cooling
    
    return list(best_set)


def _hill_climbing(points, primes, deadline=300):
    """
    Hill climbing: repeatedly try to add elements. If stuck, try to remove
    one element that is "blocking" the most additions, then add more.
    """
    start = time.time()
    if not points:
        return points
    
    points_set = set(points)
    current_len = len(points)
    best_set = set(points_set)
    best_len = current_len
    
    relevant_primes = [p for p in primes if p <= N]
    
    while time.time() - start < deadline:
        removed = [v for v in range(N + 1) if v not in points_set]
        random.shuffle(removed)
        
        added_any = False
        for v in removed:
            new_len = current_len + 1
            admissible = True
            for p in relevant_primes:
                if p > new_len:
                    break
                residues = set(x % p for x in points_set)
                if v % p not in residues:
                    if len(residues) == p - 1:
                        admissible = False
                        break
            else:
                points_set.add(v)
                removed.remove(v)
                current_len += 1
                added_any = True
        
        if current_len > best_len:
            best_len = current_len
            best_set = set(points_set)
        
        if not added_any:
            # Stuck. Try to remove one element to unblock
            if not points_set:
                break
            
            # Find the element whose removal unblocks the most additions
            # Approximation: remove a random element
            candidates = list(points_set)
            random.shuffle(candidates)
            
            best_gain = 0
            best_remove = None
            best_adds = []
            
            for elem in candidates[:50]:  # Try up to 50 candidates
                test_set = set(points_set)
                test_set.remove(elem)
                test_removed = [v for v in range(N + 1) if v not in test_set]
                
                gain = 0
                adds = []
                for v in test_removed:
                    new_len = len(test_set) + 1
                    admissible = True
                    for p in relevant_primes:
                        if p > new_len:
                            break
                        residues = set(x % p for x in test_set)
                        if v % p not in residues:
                            if len(residues) == p - 1:
                                admissible = False
                                break
                    else:
                        gain += 1
                        adds.append(v)
                        if gain > 10:  # Cap for efficiency
                            break
                
                if gain - 1 > best_gain:  # Net gain (added - removed)
                    best_gain = gain - 1
                    best_remove = elem
                    best_adds = adds
            
            if best_remove is not None and best_gain > 0:
                points_set.remove(best_remove)
                for v in best_adds[:best_gain + 1]:
                    points_set.add(v)
                current_len = len(points_set)
                if current_len > best_len:
                    best_len = current_len
                    best_set = set(points_set)
            else:
                break
    
    return list(best_set)


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    
    An admissible tuple is one where for every prime p <= len(tuple),
    some residue class mod p is missed.
    """
    start_time = time.time()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Try multiple strategies and keep the best
    best_result = []
    best_len = 0
    
    # Phase 1: Greedy construction with different orderings
    strategies = [
        ("increasing", 0.2),
        ("decreasing", 0.2),
        ("random", 0.2),
    ]
    
    for order, time_frac in strategies:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 30
        if remaining < 5:
            break
        
        deadline_phase1 = remaining * time_frac
        result = _greedy_construct(primes, order=order, deadline=deadline_phase1)
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Phase 2: Refinement with local search and hill climbing
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 10
    
    if remaining_time > 10 and best_result:
        # Local search refinement
        ls_deadline = min(remaining_time * 0.3, 60)
        refined = _local_search_refine(best_result, primes, deadline=ls_deadline)
        if len(refined) > best_len:
            best_len = len(refined)
            best_result = refined
        
        # Hill climbing
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed - 10
        if remaining_time > 10:
            hc_deadline = min(remaining_time * 0.4, 90)
            refined_hc = _hill_climbing(best_result, primes, deadline=hc_deadline)
            if len(refined_hc) > best_len:
                best_len = len(refined_hc)
                best_result = refined_hc
        
        # Simulated annealing
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed - 5
        if remaining_time > 10:
            sa_deadline = min(remaining_time * 0.5, 120)
            refined_sa = _simulated_annealing(best_result, primes, deadline=sa_deadline)
            if len(refined_sa) > best_len:
                best_len = len(refined_sa)
                best_result = refined_sa
    
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
    
    best_result = sorted(set(best_result))
    return best_result
