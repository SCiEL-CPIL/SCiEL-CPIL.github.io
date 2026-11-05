"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys

def _check_candidate(x, S, existing_sums):
    """Check if x can be added to S without triple-sum collisions. Returns (valid, new_sums)."""
    n = len(S)
    new_sums = set()
    new_sums.add(3 * x)
    for y in S:
        new_sums.add(2 * x + y)
        new_sums.add(x + 2 * y)
    for i in range(n):
        for j in range(i + 1, n):
            new_sums.add(x + S[i] + S[j])
    
    expected = 1 + n + n + n*(n-1)//2
    if len(new_sums) < expected:
        return False, None
    if new_sums & existing_sums:
        return False, None
    return True, new_sums

def _greedy_increasing(deadline):
    """Simple greedy in increasing order."""
    S = []
    existing_sums = set()
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    return S

def _greedy_power_sequence(deadline, base=3):
    """Greedy using power-based spacing to reduce collisions."""
    S = []
    existing_sums = set()
    
    # Generate candidates with increasing gaps
    candidates = []
    x = 1
    gap = 1
    while x <= 4096:
        candidates.append(x)
        gap += 1
        x += gap
    
    # Also add some intermediate values for flexibility
    extra = []
    for i in range(1, 4097):
        if i not in candidates and i % 2 == 0:
            extra.append(i)
    
    candidates = sorted(candidates + extra)
    
    for x in candidates:
        if time.time() > deadline:
            break
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    return S

def _greedy_randomized(deadline, seed=0):
    """Greedy with multiple random orderings, biased toward small numbers."""
    rng = random.Random(seed)
    best = []
    
    num_attempts = 10
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Create ordering: small numbers first with some shuffling
        order = list(range(1, 4097))
        small = order[:100]
        medium = order[100:500]
        large = order[500:]
        
        rng.shuffle(medium)
        rng.shuffle(large)
        order = small + medium + large
        
        S = []
        existing_sums = set()
        
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = _check_candidate(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums |= new_sums
        
        if len(S) > len(best):
            best = S[:]
    
    return best

def _greedy_hill_climbing(deadline, seed=0):
    """Start with a base set, then try to improve by swapping elements."""
    rng = random.Random(seed)
    
    # Start with greedy increasing
    S = _greedy_increasing(deadline)
    existing_sums = set()
    
    # Rebuild existing_sums from S
    for i, x in enumerate(S):
        for j in range(i, len(S)):
            for k in range(j, len(S)):
                existing_sums.add(S[i] + S[j] + S[k])
    
    best = S[:]
    best_size = len(S)
    
    # Try local search: remove one element, try to add a better one
    for _ in range(20):
        if time.time() > deadline:
            break
        if not S:
            break
        
        # Pick a random element to try removing
        idx = rng.randint(0, len(S) - 1)
        removed = S[idx]
        S.pop(idx)
        
        # Recompute existing_sums without the removed element
        existing_sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    existing_sums.add(S[i] + S[j] + S[k])
        
        # Try to find a replacement
        found = False
        for x in range(1, 4097):
            if x == removed:
                continue
            if x in S:
                continue
            valid, new_sums = _check_candidate(x, S, existing_sums)
            if valid:
                S.append(x)
                S.sort()
                existing_sums |= new_sums
                found = True
                break
        
        if found and len(S) >= best_size:
            best = S[:]
            best_size = len(S)
        else:
            # Revert
            S.append(removed)
            S.sort()
            # Recompute existing_sums
            existing_sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    for k in range(j, len(S)):
                        existing_sums.add(S[i] + S[j] + S[k])
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Strategy 1: Simple greedy in increasing order
    S1 = _greedy_increasing(deadline)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Power sequence greedy
    if time.time() < deadline:
        S2 = _greedy_power_sequence(deadline, base=3)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Randomized greedy
    if time.time() < deadline:
        S3 = _greedy_randomized(deadline, seed=42)
        if len(S3) > len(best):
            best = S3[:]
    
    # Strategy 4: Hill climbing from best so far
    if time.time() < deadline:
        S4 = _greedy_hill_climbing(deadline, seed=123)
        if len(S4) > len(best):
            best = S4[:]
    
    return best
