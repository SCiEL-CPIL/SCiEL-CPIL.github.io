import numpy as np
import time
import random

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _is_admissible(points, primes):
    """Check if a set of points is admissible."""
    n = len(points)
    if n == 0:
        return True
    point_set = set(points)
    for p in primes:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        residues = set(x % p for x in point_set)
        if len(residues) == p:
            return False
    return True

def _greedy_residue_removal(x_indices, active, PRIMES, lookahead=1):
    """
    Greedy residue removal with lookahead tie-breaking.
    """
    current_size = int(np.sum(active))
    
    for i, p in enumerate(PRIMES):
        if p > current_size:
            break
        
        # Count active elements in each residue class mod p
        residues = x_indices[active] % p
        if len(residues) == 0:
            break
        
        # Count occurrences of each residue
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            # Already admissible for this prime
            continue
        
        # Find all classes with min count
        candidate_classes = np.where(counts == min_count)[0]
        
        if len(candidate_classes) == 1:
            r = candidate_classes[0]
        else:
            # Lookahead: pick the candidate that minimizes the min_count of the next prime
            best_r = candidate_classes[0]
            best_score = float('inf')
            
            # Find the next prime to check
            next_p = None
            for q in PRIMES[i+1:]:
                if q <= current_size - min_count: # Approx check
                    next_p = q
                    break
            
            if next_p is not None:
                for r in candidate_classes:
                    # Simulate removal
                    mask = (x_indices % p) != r
                    new_active = active & mask
                    new_residues = x_indices[new_active] % next_p
                    new_counts = np.bincount(new_residues, minlength=next_p)
                    new_min = np.min(new_counts)
                    
                    # We want to minimize the "difficulty" for the next step.
                    # A smaller min_count means we have to remove fewer elements later?
                    # Actually, if min_count is 0, it's already satisfied.
                    # If min_count is large, it's "harder" to satisfy?
                    # Let's say we prefer candidates that leave the next prime with min_count == 0 if possible.
                    # Or generally, prefer lower min_count?
                    # If min_count is 0, we skip removal. That's good.
                    # So score = new_min. Lower is better.
                    
                    if new_min < best_score:
                        best_score = new_min
                        best_r = r
            else:
                # No lookahead, pick random or smallest
                best_r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
            
            r = best_r
            
        # Deactivate elements in this class
        mask = (x_indices % p) == r
        current_size -= min_count
        active = active & (x_indices % p != r)
        
        if current_size == 0:
            break
    
    return active

def _polish(points, deadline):
    """
    Try to add back removed elements to maximize size.
    This is a heuristic polish.
    We know the current set is admissible.
    We try to add elements from [0, N] that are not in points.
    """
    point_set = set(points)
    current_size = len(points)
    
    # Get all possible candidates to add
    candidates = [x for x in range(RANGE) if x not in point_set]
    random.shuffle(candidates)
    
    # To make this fast, we can check in batches or just iterate.
    # Since N=5000, iterating 5000 times is fine if the check is fast.
    # But _is_admissible is O(N * pi(N)) which is ~5000 * 600 = 3M ops.
    # 5000 * 3M = 15B ops. Too slow.
    
    # We need a faster check.
    # We can maintain the residue counts for the current set.
    # For each prime p <= current_size, we have counts of residues.
    # If we add x, for each p, counts[x % p] increases by 1.
    # We need to ensure that no prime p has all classes covered (i.e., min_count > 0 for all classes).
    # Wait, admissible means SOME class is missed. i.e., min_count == 0.
    # So we need to ensure that for every p <= new_size, min_count(p) == 0.
    
    # Let's maintain a list of "critical" primes where min_count == 0.
    # Actually, we just need to check if adding x causes any prime to become "full".
    # A prime p is "full" if min_count(p) > 0.
    # If we add x, min_count(p) can only increase or stay same (if x was in a class with count > 0).
    # Wait, if min_count(p) was 0, and we add x to a class that had count 0, it becomes 1.
    # If all other classes had count > 0, then min_count becomes > 0 (if the 0 was the unique min).
    # Actually, if min_count was 0, there is at least one class with 0.
    # If we add x to a class with count > 0, min_count stays 0.
    # If we add x to a class with count 0, that class becomes 1.
    # If it was the ONLY class with 0, min_count becomes 1 (if all others > 0).
    # So we need to know for each p:
    # 1. min_count
    # 2. number of classes with min_count
    
    # This state is complex to maintain dynamically for all primes as size changes.
    # However, the set of primes we care about is fixed: p <= new_size.
    # As we add elements, new_size increases, so we might need to check new primes.
    
    # Let's try a simpler approach:
    # Randomly try to add candidates.
    # For each candidate, check admissibility of the NEW set.
    # To speed up, we can precompute residues.
    
    # Given time constraints, maybe a limited polish is better.
    # Or just skip polish if it's too slow.
    # The greedy removal is already quite good.
    # Let's try a very fast polish:
    # Try to add candidates in random order.
    # Keep a set of active primes and their counts?
    
    # Let's implement a fast check:
    # For the current set, compute counts for all p <= current_size.
    # Store min_count and num_min_classes for each p.
    
    active = np.zeros(RANGE, dtype=bool)
    for x in points:
        active[x] = True
    
    x_indices = np.arange(RANGE)
    
    # Precompute counts for current set
    # This is O(P * N) where P is number of primes <= N.
    # P ~ 600. N = 5000. 3M ops. Fast enough.
    
    counts_cache = {}
    min_counts = {}
    num_mins = {}
    
    def update_cache(active_mask):
        for p in PRIMES:
            if p > int(np.sum(active_mask)):
                break
            residues = x_indices[active_mask] % p
            if len(residues) == 0:
                counts_cache[p] = np.zeros(p, dtype=int)
                min_counts[p] = 0
                num_mins[p] = p
                continue
            c = np.bincount(residues, minlength=p)
            counts_cache[p] = c
            m = np.min(c)
            min_counts[p] = m
            num_mins[p] = np.sum(c == m)
    
    update_cache(active)
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Check if adding x keeps it admissible
        # We need to check primes p <= current_size + 1
        
        is_valid = True
        for p in PRIMES:
            if p > current_size + 1:
                break
                
            if p not in counts_cache:
                # New prime, need to compute
                residues = x_indices[active] % p
                c = np.bincount(residues, minlength=p)
                counts_cache[p] = c
                min_counts[p] = np.min(c)
                num_mins[p] = np.sum(c == min_counts[p])
                
            c = counts_cache[p]
            m = min_counts[p]
            nm = num_mins[p]
            
            r = x % p
            current_count = c[r]
            
            # If we add x, c[r] becomes current_count + 1
            # New min:
            if current_count == m:
                # This class was one of the mins
                # If it was the only min, and the next smallest is m+1, then new min is m+1?
                # No, if current_count == m, and we increment it, it becomes m+1.
                # If nm == 1, then all other classes are > m.
                # The next smallest is at least m+1.
                # So new min is min(m+1, next_smallest).
                # Since next_smallest >= m+1, new min is m+1.
                # If nm > 1, there is another class with count m. So new min is m.
                
                if nm == 1:
                    # Need to find next smallest
                    # This is expensive to do dynamically.
                    # Approximation: if nm == 1, adding x to the unique min class
                    # will increase the min count.
                    # We need to know if the new min count is still 0.
                    # If m == 0, and nm == 1, then there is exactly one class with 0.
                    # Adding x to it makes it 1.
                    # If all other classes are >= 1, then new min is 1.
                    # This would make the prime "full" (min > 0), which is bad.
                    # So if m == 0 and nm == 1 and current_count == 0, adding x is INVALID.
                    
                    if m == 0:
                        is_valid = False
                        break
                else:
                    # nm > 1, so another class has count m. New min is still m.
                    pass
            else:
                # current_count > m.
                # Adding x increases a non-min class.
                # Min stays m.
                pass
        
        if is_valid:
            # Add x
            active[x] = True
            point_set.add(x)
            current_size += 1
            
            # Update cache for affected primes
            # This is lazy, we can just recompute if needed or update incrementally.
            # For simplicity, let's just mark cache as dirty or recompute?
            # Recomputing is O(P*N). Too slow per step.
            # Incremental update:
            # For each p, c[r] += 1.
            # Update m and nm.
            
            for p in PRIMES:
                if p > current_size:
                    break
                if p not in counts_cache:
                    continue
                c = counts_cache[p]
                m = min_counts[p]
                nm = num_mins[p]
                r = x % p
                old_c = c[r]
                c[r] = old_c + 1
                
                if old_c == m:
                    # We incremented a min class
                    if nm == 1:
                        # It was the unique min.
                        # New min is min(old_c+1, second_min).
                        # We don't have second_min stored.
                        # This makes incremental update hard.
                        # Let's just recompute for this prime?
                        # Or accept that the check might be slightly off?
                        # If we are conservative, we might reject valid additions.
                        # That's okay for a heuristic.
                        pass
                    else:
                        nm -= 1
                        # m stays same
                else:
                    # We incremented a non-min class.
                    # m and nm stay same.
                    pass
                
                min_counts[p] = m
                num_mins[p] = nm

    return np.where(active)[0].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Precompute x indices
    x_indices = np.arange(RANGE)
    
    # Strategy 1: Greedy with lookahead
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        active = np.ones(RANGE, dtype=bool)
        
        # Use lookahead=1
        active = _greedy_residue_removal(x_indices, active, PRIMES, lookahead=1)
        
        points = np.where(active)[0].tolist()
        
        # Try to polish
        if time.time() < deadline - 2:
            polished_points = _polish(points, deadline - 2)
            if len(polished_points) > len(points):
                points = polished_points
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
    return best_points
