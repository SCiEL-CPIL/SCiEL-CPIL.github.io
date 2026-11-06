"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

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
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if all(c > 0 for c in class_counts):
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
    Optimized with precomputed safety masks for each prime.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    relevant_primes = [p for p in all_primes if p <= m + 1]
    
    residue_counts = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    nonempty_counts = {}
    for p in relevant_primes:
        nonempty_counts[p] = sum(1 for c in residue_counts[p] if c > 0)
    
    # Precompute which residue classes are "critical" (adding would complete the set)
    critical_mask = {}
    for p in relevant_primes:
        ne = nonempty_counts[p]
        if ne + 1 == p:
            # All but one class are filled; the empty class is critical
            empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
            critical_mask[p] = set(empty_classes)
        else:
            critical_mask[p] = set()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
                ne = nonempty_counts[p]
                if ne + 1 == p:
                    empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                    critical_mask[p] = set(empty_classes)
                else:
                    critical_mask[p] = set()
        relevant_primes = new_relevant
        
        # Try candidates in a smarter order: those with fewer conflicts first
        candidates = [c for c in range(N + 1) if c not in points_set]
        
        # Score candidates by number of critical conflicts
        def conflict_score(c):
            score = 0
            for p in relevant_primes:
                if p in critical_mask and c % p in critical_mask[p]:
                    score += 1
            return score
        
        candidates.sort(key=conflict_score)
        
        for candidate in candidates:
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                return points
            
            admissible = True
            for p in relevant_primes:
                counts = residue_counts[p]
                ne = nonempty_counts[p]
                r = candidate % p
                if counts[r] == 0:
                    if ne + 1 == p:
                        admissible = False
                        break
            
            if admissible:
                points.append(candidate)
                points_set.add(candidate)
                m += 1
                for p in relevant_primes:
                    r = candidate % p
                    was_empty = residue_counts[p][r] == 0
                    residue_counts[p][r] += 1
                    if was_empty:
                        nonempty_counts[p] += 1
                        # Update critical mask
                        if nonempty_counts[p] + 1 == p:
                            empty_classes = [rr for rr in range(p) if residue_counts[p][rr] == 0]
                            critical_mask[p] = set(empty_classes)
                        else:
                            critical_mask[p] = set()
                if m + 1 <= N + 1:
                    for p in all_primes:
                        if p == m + 1:
                            counts = [0] * p
                            for v in points:
                                counts[v % p] += 1
                            residue_counts[p] = counts
                            nonempty_counts[p] = sum(1 for c in counts if c > 0)
                            ne = nonempty_counts[p]
                            if ne + 1 == p:
                                empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                                critical_mask[p] = set(empty_classes)
                            else:
                                critical_mask[p] = set()
                            relevant_primes.append(p)
                            break
                improved = True
    
    return points


def _swap_improvement(points, all_primes, deadline):
    """
    Swap improvement: try to replace one element with two others to increase tuple size.
    Optimized: limit search depth and use smarter candidate ordering.
    """
    points_set = set(points)
    points = list(points)
    m = len(points)
    
    if m == 0:
        return points
    
    # Precompute residues for all relevant primes
    relevant_primes = [p for p in all_primes if p <= m + 2]
    
    # Build residue counts
    residue_counts = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    nonempty_counts = {}
    for p in relevant_primes:
        nonempty_counts[p] = sum(1 for c in residue_counts[p] if c > 0)
    
    # Precompute critical masks
    critical_mask = {}
    for p in relevant_primes:
        ne = nonempty_counts[p]
        if ne + 1 == p:
            empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
            critical_mask[p] = set(empty_classes)
        else:
            critical_mask[p] = set()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes
        new_relevant = [p for p in all_primes if p <= m + 2]
        for p in new_relevant:
            if p not in residue_counts:
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                residue_counts[p] = counts
                nonempty_counts[p] = sum(1 for c in counts if c > 0)
                ne = nonempty_counts[p]
                if ne + 1 == p:
                    empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                    critical_mask[p] = set(empty_classes)
                else:
                    critical_mask[p] = set()
        relevant_primes = new_relevant
        
        # Try swap: for each x in points, try to remove x and add two candidates
        # Limit to first 50 elements to keep runtime bounded
        max_swaps_to_try = min(len(points), 50)
        for x_idx in range(max_swaps_to_try):
            if time.time() >= deadline:
                return points
            x = points[x_idx]
            
            # Temporarily remove x
            temp_points = points[:x_idx] + points[x_idx+1:]
            temp_set = points_set - {x}
            temp_m = m - 1
            
            # Update residue counts for temp state
            temp_residue_counts = {}
            for p in relevant_primes:
                counts = list(residue_counts[p])
                r = x % p
                counts[r] -= 1
                if counts[r] < 0:
                    counts[r] = 0
                temp_residue_counts[p] = counts
            
            temp_nonempty = {}
            for p in relevant_primes:
                temp_nonempty[p] = sum(1 for c in temp_residue_counts[p] if c > 0)
            
            # Update critical masks for temp state
            temp_critical = {}
            for p in relevant_primes:
                ne = temp_nonempty[p]
                if ne + 1 == p:
                    empty_classes = [r for r in range(p) if temp_residue_counts[p][r] == 0]
                    temp_critical[p] = set(empty_classes)
                else:
                    temp_critical[p] = set()
            
            # Now try to add two candidates
            added = []
            
            # Get candidate list sorted by conflict score
            candidates = [c for c in range(N + 1) if c not in temp_set]
            
            def conflict_score(c):
                score = 0
                for p in relevant_primes:
                    if p in temp_critical and c % p in temp_critical[p]:
                        score += 1
                return score
            
            candidates.sort(key=conflict_score)
            
            # Limit candidates to first 200 for efficiency
            candidates = candidates[:200]
            
            for a in candidates:
                if a in temp_set:
                    continue
                if time.time() >= deadline:
                    break
                
                # Check if adding a is admissible
                admissible_a = True
                for p in relevant_primes:
                    counts = temp_residue_counts[p]
                    ne = temp_nonempty[p]
                    r = a % p
                    if counts[r] == 0:
                        if ne + 1 == p:
                            admissible_a = False
                            break
                
                if not admissible_a:
                    continue
                
                # Tentatively add a
                for p in relevant_primes:
                    r = a % p
                    was_empty = temp_residue_counts[p][r] == 0
                    temp_residue_counts[p][r] += 1
                    if was_empty:
                        temp_nonempty[p] += 1
                        # Update temp critical
                        if temp_nonempty[p] + 1 == p:
                            empty_classes = [rr for rr in range(p) if temp_residue_counts[p][rr] == 0]
                            temp_critical[p] = set(empty_classes)
                        else:
                            temp_critical[p] = set()
                temp_set.add(a)
                temp_m += 1
                added.append(a)
                
                # Now try to add a second candidate b
                # Limit to first 100 candidates for b
                candidates_b = [c for c in candidates if c not in temp_set][:100]
                for b in candidates_b:
                    if b in temp_set:
                        continue
                    if time.time() >= deadline:
                        break
                    
                    # Check if adding b is admissible
                    admissible_b = True
                    for p in relevant_primes:
                        counts = temp_residue_counts[p]
                        ne = temp_nonempty[p]
                        r = b % p
                        if counts[r] == 0:
                            if ne + 1 == p:
                                admissible_b = False
                                break
                    
                    if not admissible_b:
                        continue
                    
                    # Found a valid swap! Apply it
                    for p in relevant_primes:
                        r = b % p
                        was_empty = temp_residue_counts[p][r] == 0
                        temp_residue_counts[p][r] += 1
                        if was_empty:
                            temp_nonempty[p] += 1
                            if temp_nonempty[p] + 1 == p:
                                empty_classes = [rr for rr in range(p) if temp_residue_counts[p][rr] == 0]
                                temp_critical[p] = set(empty_classes)
                            else:
                                temp_critical[p] = set()
                    temp_set.add(b)
                    temp_m += 1
                    added.append(b)
                    
                    # Update global state
                    points = temp_points + added
                    points_set = temp_set
                    m = temp_m
                    residue_counts = temp_residue_counts
                    nonempty_counts = temp_nonempty
                    # Rebuild critical_mask from new state
                    for p in relevant_primes:
                        ne = nonempty_counts[p]
                        if ne + 1 == p:
                            empty_classes = [r for r in range(p) if residue_counts[p][r] == 0]
                            critical_mask[p] = set(empty_classes)
                        else:
                            critical_mask[p] = set()
                    improved = True
                    break
                
                # Undo adding a if we didn't find b
                if len(added) == 1:
                    for p in relevant_primes:
                        r = a % p
                        temp_residue_counts[p][r] -= 1
                        if temp_residue_counts[p][r] == 0:
                            temp_nonempty[p] -= 1
                            if temp_nonempty[p] + 1 == p:
                                empty_classes = [rr for rr in range(p) if temp_residue_counts[p][rr] == 0]
                                temp_critical[p] = set(empty_classes)
                            else:
                                temp_critical[p] = set()
                    temp_set.remove(a)
                    temp_m -= 1
                    added.pop()
                
                if len(added) == 2:
                    break
                
                if time.time() >= deadline:
                    break
            
            if improved:
                break
    
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
    
    # Strategy 4: Start from numbers coprime to 6
    if time.time() < deadline - 20:
        points_coprime6 = [x for x in range(N + 1) if x % 2 != 0 and x % 3 != 0]
        points_coprime6 = _greedy_repair(points_coprime6, all_primes, deadline)
        if len(points_coprime6) > len(best_points):
            best_points = points_coprime6

    # Local improvement: try to add back elements using fast method
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Swap improvement: try 1-out-2-in exchanges
    if time.time() < deadline and len(best_points) > 0:
        best_points = _swap_improvement(best_points, all_primes, deadline)

    # Final local improvement pass
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Second swap pass if time remains
    if time.time() < deadline - 60 and len(best_points) > 0:
        best_points = _swap_improvement(best_points, all_primes, deadline)

    # Final local improvement
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improvement_fast(best_points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(best_points, all_primes):
        best_points = _greedy_repair(best_points, all_primes, deadline)

    return best_points
# EVOLVE-BLOCK-END
