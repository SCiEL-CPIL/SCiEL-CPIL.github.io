"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random

def _new_sums_involving(x, S):
    """Compute all triple sums that include x, given existing sorted set S."""
    n = len(S)
    sums = [3 * x]
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _check_collision(new_sums, existing_sums):
    """Check if any new sum collides with existing sums."""
    for s in new_sums:
        if s in existing_sums:
            return True
    return False

def _greedy_linear(start=1, end=4096, step=1):
    """Greedy construction scanning linearly from start to end."""
    selected = []
    existing_sums = set()
    for x in range(start, end + 1, step):
        new_sums = _new_sums_involving(x, selected)
        if not _check_collision(new_sums, existing_sums):
            existing_sums.update(new_sums)
            selected.append(x)
    return selected

def _greedy_exponential(base_start=1, ratio=2.0, max_val=4096):
    """Greedy construction using exponential spacing to spread elements."""
    selected = []
    existing_sums = set()
    x = base_start
    while x <= max_val:
        new_sums = _new_sums_involving(x, selected)
        if not _check_collision(new_sums, existing_sums):
            existing_sums.update(new_sums)
            selected.append(x)
        x = int(x * ratio) + 1
        if x <= base_start:
            x = base_start + 1
    # Fill in gaps with linear scan
    last = selected[-1] if selected else 0
    for x in range(1, max_val + 1):
        if x in selected:
            continue
        new_sums = _new_sums_involving(x, selected)
        if not _check_collision(new_sums, existing_sums):
            existing_sums.update(new_sums)
            selected.append(x)
    selected.sort()
    return selected

def _local_search_improve(selected, max_val=4096, time_limit=10):
    """Try to improve the set by removing one element and adding others."""
    start_time = time.time()
    current = list(selected)
    best = list(current)
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        current_sums = set()
        n = len(current)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    current_sums.add(current[i] + current[j] + current[k])
        
        # Try removing each element and see if we can add more
        for rem_idx in range(n):
            if time.time() - start_time > time_limit:
                break
            trial = [x for i, x in enumerate(current) if i != rem_idx]
            trial_sums = set()
            tn = len(trial)
            for i in range(tn):
                for j in range(i, tn):
                    for k in range(j, tn):
                        trial_sums.add(trial[i] + trial[j] + trial[k])
            
            added = 0
            for x in range(1, max_val + 1):
                if x in trial:
                    continue
                new_sums = _new_sums_involving(x, trial)
                if not _check_collision(new_sums, trial_sums):
                    trial_sums.update(new_sums)
                    trial.append(x)
                    added += 1
            
            if len(trial) > len(best):
                best = list(trial)
                current = list(trial)
    
    return best

def construct():
    """Find largest B_3[1] set in [1, 4096]."""
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    best = []
    
    # Strategy 1: Linear greedy from 1
    candidate = _greedy_linear(1, 4096)
    if len(candidate) > len(best):
        best = candidate
    
    # Strategy 2: Exponential-based greedy with different ratios
    for ratio in [1.5, 1.7, 2.0, 2.2, 2.5]:
        if time.time() > deadline:
            break
        candidate = _greedy_exponential(1, ratio, 4096)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 3: Randomized greedy with varying starts and steps
    rng = random.Random(12345)
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        start = rng.randint(1, 20)
        step = rng.randint(1, 5)
        candidate = _greedy_linear(start, 4096, step)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 4: Local search improvement on best
    if time.time() < deadline - 15 and len(best) < 25:
        improved = _local_search_improve(best, 4096, time_limit=20)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 5: Final greedy fill on best
    if time.time() < deadline - 5:
        existing = list(best)
        existing_sums = set()
        n = len(existing)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(existing[i] + existing[j] + existing[k])
        
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            new_sums = _new_sums_involving(x, existing)
            if not _check_collision(new_sums, existing_sums):
                existing_sums.update(new_sums)
                existing.append(x)
        
        if len(existing) > len(best):
            best = existing
    
    return best
