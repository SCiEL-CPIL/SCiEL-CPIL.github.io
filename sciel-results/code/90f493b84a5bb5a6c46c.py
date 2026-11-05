import time
import random
import itertools

def is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def count_collision_pairs(A):
    """
    Count the number of duplicate unordered pair sums (i <= j) in A.
    More collisions = harder to find large Sidon subsets.
    """
    n = len(A)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(A[i] + A[j])
    # Number of duplicates = total pairs - unique sums
    return len(sums) - len(set(sums))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_A = None
    best_val = 12  # Max possible is 12 (the whole set)
    best_collisions = -1
    
    # Strategy: Use collision-maximization via local search
    # We want to maximize the number of duplicate pair sums to minimize
    # the largest Sidon subset.
    
    # Good initial patterns:
    # 1. Dense arithmetic progressions (many sum collisions)
    # 2. Sets with structure that creates many collisions
    
    initial_patterns = []
    
    # Dense APs with various moduli
    for start in range(0, 20):
        for step in range(1, 12):
            pattern = []
            for i in range(12):
                val = start + i * step
                if val <= 128:
                    pattern.append(val)
            if len(pattern) == 12:
                initial_patterns.append(pattern)
    
    # Symmetric patterns around center
    for center in range(60, 100):
        for spread in range(1, 30):
            pattern = []
            for i in range(12):
                val = center + (i - 5.5) * spread
                val = int(round(val))
                if 0 <= val <= 128:
                    pattern.append(val)
            if len(set(pattern)) == 12 and len(pattern) == 12:
                initial_patterns.append(pattern)
    
    # Random patterns
    for _ in range(50):
        pattern = random.sample(range(129), 12)
        initial_patterns.append(pattern)
    
    # Evaluate all initial patterns
    for pattern in initial_patterns:
        if time.time() > deadline:
            break
        
        if len(set(pattern)) != 12:
            continue
        if any(x < 0 or x > 128 for x in pattern):
            continue
        
        val = max_sidon_size(pattern)
        col = count_collision_pairs(pattern)
        
        if val < best_val or (val == best_val and col > best_collisions):
            best_A = list(pattern)
            best_val = val
            best_collisions = col
    
    if best_A is None:
        best_A = list(range(12))
        best_val = max_sidon_size(best_A)
        best_collisions = count_collision_pairs(best_A)
    
    # Local search to improve
    current_A = list(best_A)
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from best
            current_A = list(best_A)
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.5)
            continue
        
        # Pick a random index to modify
        idx = random.randint(0, 11)
        old_val = current_A[idx]
        
        # Generate candidate values that create collisions
        other_indices = [i for i in range(12) if i != idx]
        existing_sums = set()
        for i in range(12):
            for j in range(i, 12):
                if i != idx and j != idx:
                    existing_sums.add(current_A[i] + current_A[j])
        
        candidates = set()
        
        # Values that would create a sum collision
        for s in list(existing_sums):
            for i in other_indices:
                v = s - current_A[i]
                if 0 <= v <= 128:
                    candidates.add(v)
        
        # Values where 2*v creates a collision
        for s in list(existing_sums):
            if s % 2 == 0:
                v = s // 2
                if 0 <= v <= 128:
                    candidates.add(v)
        
        # Random candidates
        for _ in range(30):
            candidates.add(random.randint(0, 128))
        
        # Values near old_val
        for delta in range(-10, 11):
            v = old_val + delta
            if 0 <= v <= 128:
                candidates.add(v)
        
        # Filter out invalid candidates
        valid_candidates = []
        for v in candidates:
            if v == old_val:
                continue
            if v in current_A:
                continue
            valid_candidates.append(v)
        
        best_candidate = None
        best_candidate_score = -1
        best_candidate_val = 12
        best_candidate_col = 0
        
        for v in valid_candidates:
            new_A = list(current_A)
            new_A[idx] = v
            
            new_val = max_sidon_size(new_A)
            new_col = count_collision_pairs(new_A)
            
            # Score: strongly prefer lower max_sidon_size, then higher collisions
            score = -new_val * 10000 + new_col
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = v
                best_candidate_val = new_val
                best_candidate_col = new_col
        
        if best_candidate is not None:
            current_A[idx] = best_candidate
            
            if best_candidate_val < best_val or (best_candidate_val == best_val and best_candidate_col > best_collisions):
                best_A = list(current_A)
                best_val = best_candidate_val
                best_collisions = best_candidate_col
                no_improve = 0
            else:
                no_improve += 1
    
    return best_A
