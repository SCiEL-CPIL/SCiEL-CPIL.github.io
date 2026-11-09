import random
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _validate(a):
    a = sorted(set(a))
    if 0 not in a:
        return False
    if len(a) > 32:
        return False
    diffs = set()
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _greedy_construct(seed=0, max_elem=10**6):
    random.seed(seed)
    a = [0]
    diffs = set()
    
    # We need to cover 1..30.
    # Strategy: try to add elements that cover many missing differences at once.
    missing = set(range(1, 31))
    
    while missing:
        candidates = []
        # Generate candidate values
        # 1. Values that create a difference with an existing element
        # 2. Random values in a reasonable range
        existing = set(a)
        
        # Prioritize small values to keep diameter small
        # Let's try to find a value x such that adding x covers many missing diffs.
        best_x = None
        best_gain = 0
        best_new_diffs = set()
        
        # Candidate generation
        # Try values in [1, 200] first, then expand if needed
        # We can be smarter: for each missing d, and each a_i, x = a_i + d or a_i - d.
        candidate_set = set()
        for d in missing:
            for x in a:
                if x + d > 0 and x + d <= max_elem:
                    candidate_set.add(x + d)
                if x - d > 0:
                    candidate_set.add(x - d)
        
        # Add some random noise to escape local optima
        for _ in range(50):
            candidate_set.add(random.randint(1, 200))
        
        # Evaluate candidates
        for x in candidate_set:
            if x in existing:
                continue
            if x > max_elem:
                continue
            
            # Check Sidon property with x
            valid = True
            new_diffs = set()
            for y in a:
                d = abs(x - y)
                if d == 0:
                    continue
                if d in diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Count how many missing diffs this covers
            gain = len(new_diffs & missing)
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_new_diffs = new_diffs
        
        if best_x is None:
            # If no candidate works, we might be stuck.
            # This can happen if we need a larger value.
            # Try searching for a valid x in a larger range.
            found = False
            # We need to add x such that all diffs (x-a_i) are new.
            # This is a constraint satisfaction problem.
            # Let's try random search in a larger range.
            for _ in range(1000):
                x = random.randint(1, max_elem // 2)
                valid = True
                new_diffs = set()
                for y in a:
                    d = abs(x - y)
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    gain = len(new_diffs & missing)
                    if gain > 0:
                        best_x = x
                        best_new_diffs = new_diffs
                        best_gain = gain
                        found = True
                        break
            if not found:
                # Fail, return None
                return None
        
        # Add best_x
        a.append(best_x)
        diffs.update(best_new_diffs)
        missing -= best_new_diffs
        
        # Optional: local search to reduce diameter?
        # For now, just continue.
        
    return a

def _local_search(a, max_steps=1000):
    # Try to swap elements to reduce the max value
    # This is complex, so we'll stick to a simple hill-climbing on the set.
    # Actually, just returning the greedy result is safer.
    # Let's try a simple random perturbation: remove an element, re-add it?
    # No, let's keep it simple. The greedy with good candidate selection should be decent.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_a = None
    best_score = float('inf')
    
    # First, try a deterministic greedy with a specific seed
    # Then, randomized restarts
    
    seeds = list(range(100))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        a = _greedy_construct(seed)
        if a is None:
            continue
        
        # Validate
        if not _validate(a):
            continue
        
        score = max(a)
        if score < best_score:
            best_score = score
            best_a = a
            
            # If we find a very good one, we can stop early?
            # Theoretical minimum for Sidon set with 1..30 diffs:
            # n elements -> n(n-1)/2 diffs.
            # We need at least 30 diffs, so n(n-1)/2 >= 30 => n=9 (36 diffs).
            # So min size is 9.
            # Diameter for n=9 Sidon set?
            # Known small Sidon sets:
            # {0, 1, 4, 10, 12, 17, 21, 23, 28} ? No, check diffs.
            # Let's just search.
            
            if best_score < 30:
                break
            if best_score < 20:
                break
        
        # Try to improve current best with local search?
        # Maybe just keep searching with different seeds.
        
    if best_a is None:
        # Fallback to original greedy if all fail?
        # The original program always returned something.
        # Let's implement a very robust fallback.
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
        
    return best_a

# EVOLVE-BLOCK-END
