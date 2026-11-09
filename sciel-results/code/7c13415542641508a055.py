import itertools
import time

def is_sidon_4(a, b, c, d):
    """Check if 4 sorted elements form a Sidon set (all pair sums including diagonals unique)."""
    # Pair sums: a+a, a+b, a+c, a+d, b+b, b+c, b+d, c+c, c+d, d+d
    sums = (2*a, a+b, a+c, a+d, 2*b, b+c, b+d, 2*c, c+d, 2*d)
    return len(set(sums)) == 10

def max_sidon_size(arr):
    """Return the size of the largest Sidon subset of arr (12 elements)."""
    # Check for 5-element Sidon subset
    for combo in itertools.combinations(arr, 5):
        # Check if this 5-set is Sidon
        sums = set()
        ok = True
        for i in range(5):
            for j in range(i, 5):
                s = combo[i] + combo[j]
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if not ok:
                break
        if ok:
            return 5
    # Check for 4-element Sidon subset
    for combo in itertools.combinations(arr, 4):
        if is_sidon_4(combo[0], combo[1], combo[2], combo[3]):
            return 4
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    best_set = None
    best_val = 12
    
    # Strategy: search for a set with max Sidon size = 3
    # This requires that NO 4-element subset is Sidon.
    
    # Try structured sets first
    # Pattern: blocks of consecutive integers
    structured_candidates = []
    
    # Try various block patterns
    for block_sizes in [(3,3,3,3), (4,4,4), (2,2,2,2,2,2), (3,3,3,2,1), (4,3,3,2), (5,4,3)]:
        if sum(block_sizes) != 12:
            continue
        # Generate sets with these block sizes
        for start in range(0, 129 - 20):
            s = set()
            pos = start
            for bs in block_sizes:
                for k in range(bs):
                    if pos + k < 129:
                        s.add(pos + k)
                pos += bs + 1  # gap of 1 between blocks
            if len(s) == 12 and max(s) < 129:
                structured_candidates.append(list(s))
    
    # Also try arithmetic progressions with small differences
    for d in range(1, 8):
        for start in range(0, 129 - 11*d):
            s = list(range(start, start + 12*d, d))
            if len(s) == 12 and max(s) < 129:
                structured_candidates.append(s)
    
    # Try mixed patterns
    # Sets with many 3-APs
    for a in range(0, 20):
        for b in range(a+1, 40):
            for c in range(b+1, 60):
                s = {a, b, c, a+b-c, a+c-b, b+c-a, 2*a-b, 2*b-a, 2*c-b}
                # Add more to get 12
                s = set(x for x in s if 0 <= x <= 128)
                if len(s) >= 12:
                    structured_candidates.append(sorted(s)[:12])
    
    # Evaluate structured candidates
    for cand in structured_candidates:
        if time.time() > deadline:
            break
        val = max_sidon_size(cand)
        if val < best_val:
            best_val = val
            best_set = cand
            if best_val == 3:
                return best_set
    
    # Random search
    import random
    random.seed(42)
    for _ in range(50000):
        if time.time() > deadline:
            break
        s = random.sample(range(129), 12)
        val = max_sidon_size(s)
        if val < best_val:
            best_val = val
            best_set = s
            if best_val == 3:
                return best_set
    
    # Local search: try to improve best_set by replacing elements
    if best_set is not None and best_val >= 4:
        current = best_set[:]
        current_val = best_val
        for iteration in range(200000):
            if time.time() > deadline:
                break
            if current_val == 3:
                return current
            
            # Pick a random index and try replacing with a random value
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            trial = current[:]
            trial[idx] = new_val
            trial_val = max_sidon_size(trial)
            
            if trial_val < current_val:
                current = trial
                current_val = trial_val
            elif trial_val == current_val and random.random() < 0.1:
                # Occasionally accept equal to escape local minima
                current = trial
    
    if best_set is not None:
        return best_set
    return list(range(12))
