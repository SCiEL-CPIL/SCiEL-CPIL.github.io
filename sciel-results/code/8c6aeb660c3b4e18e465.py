"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n_max = 4096
    best = []
    best_sums = set()
    best_len = 0
    
    def try_add(selected, sums_set, x):
        """Try adding x to selected. Returns True if successful, updates sums_set."""
        if x in selected:
            return False
        # Check all triples involving x with elements in selected
        # Triples: (x,x,x), (x,x,a), (x,a,b) where a,b in selected, a<=b
        new_sums = set()
        # (x,x,x)
        s = 3 * x
        if s in sums_set:
            return False
        new_sums.add(s)
        
        # (x,x,a) for each a in selected
        for a in selected:
            s = 2 * x + a
            if s in sums_set or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x,a,b) for each a<=b in selected
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in sums_set or s in new_sums:
                    return False
                new_sums.add(s)
        
        # Also need to check triples among selected that we haven't checked before
        # But those are already in sums_set
        sums_set.update(new_sums)
        return True
    
    # Strategy 1: Standard greedy from 1
    selected = []
    sums_set = set()
    for x in range(1, n_max + 1):
        if time.time() > deadline:
            break
        if try_add(selected, sums_set, x):
            selected.append(x)
    
    if len(selected) > best_len:
        best = selected
        best_sums = sums_set
        best_len = len(best)
    
    # Strategy 2: Try greedy from various offsets with random restarts
    random.seed(42)
    for restart in range(100):
        if time.time() > deadline:
            break
        offset = random.randint(1, 200)
        selected = []
        sums_set = set()
        for x in range(offset, n_max + 1):
            if time.time() > deadline:
                break
            if try_add(selected, sums_set, x):
                selected.append(x)
        
        if len(selected) > best_len:
            best = selected
            best_sums = sums_set
            best_len = len(best)
    
    # Strategy 3: Try with larger gaps (skip some numbers)
    for restart in range(50):
        if time.time() > deadline:
            break
        selected = []
        sums_set = set()
        current = 1
        while current <= n_max:
            if time.time() > deadline:
                break
            if try_add(selected, sums_set, current):
                selected.append(current)
            # Randomly skip some numbers
            skip = random.randint(1, 10)
            current += skip
    
    if len(selected) > best_len:
        best = selected
        best_sums = sums_set
        best_len = len(best)
    
    # Strategy 4: Try exponential spacing
    for restart in range(30):
        if time.time() > deadline:
            break
        base = random.uniform(1.5, 4.0)
        selected = []
        sums_set = set()
        current = 1
        while current <= n_max:
            if time.time() > deadline:
                break
            val = int(current)
            if try_add(selected, sums_set, val):
                selected.append(val)
            current *= base
    
    if len(selected) > best_len:
        best = selected
        best_sums = sums_set
        best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
