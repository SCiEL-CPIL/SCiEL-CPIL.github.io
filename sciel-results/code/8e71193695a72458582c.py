"""Program for maximizing 3-AP differences in a 32-element set within 1..512."""

import random
import time
import sys

def count_aps(s):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in the set s.
    """
    s_set = set(s)
    diffs = set()
    for i in range(len(s)):
        a = s[i]
        for j in range(len(s)):
            if i == j: continue
            b = s[j]
            if b <= a: continue
            d = b - a
            c = a + 2 * d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def count_aps_fast(s):
    """
    Faster count using sorted list and set lookups.
    """
    s_set = set(s)
    diffs = set()
    for a in s:
        for b in s:
            if b <= a:
                continue
            d = b - a
            if a + 2*d in s_set:
                diffs.add(d)
    return len(diffs)

def construct():
    """
    Use simulated annealing with improved moves to find a 32-element set
    in [1, 512] that maximizes the number of distinct 3-AP differences.
    """
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize with a structured set: use multiples to create many APs
    # Try several initial configurations and pick the best
    best_init = None
    best_init_score = 0
    
    # Strategy 1: Random
    for _ in range(5):
        s = set(random.sample(range(1, MAX_VAL + 1), N))
        sc = count_aps(list(s))
        if sc > best_init_score:
            best_init_score = sc
            best_init = s
    
    # Strategy 2: Arithmetic progression based
    for base in [1, 2, 3, 5, 7, 10, 13, 16, 20]:
        s = set()
        # Mix of APs with different differences
        # Take every k-th element from 1..MAX_VAL
        step = base
        count = 0
        for val in range(1, MAX_VAL + 1, step):
            if count < N:
                s.add(val)
                count += 1
        if len(s) < N:
            # Fill remaining randomly
            remaining = [x for x in range(1, MAX_VAL + 1) if x not in s]
            s.update(random.sample(remaining, N - len(s)))
        sc = count_aps(list(s))
        if sc > best_init_score:
            best_init_score = sc
            best_init = s
    
    # Strategy 3: Two interleaved APs
    for d1 in [2, 3, 4, 5, 6, 8]:
        for d2 in [1, 2, 3, 4, 5]:
            s = set()
            # First AP: 1, 1+d1, 1+2*d1, ...
            for i in range(N // 2):
                val = 1 + i * d1
                if val <= MAX_VAL:
                    s.add(val)
            # Second AP: 2, 2+d2, 2+2*d2, ...
            for i in range(N - N // 2):
                val = 2 + i * d2
                if val <= MAX_VAL and val not in s:
                    s.add(val)
            if len(s) < N:
                remaining = [x for x in range(1, MAX_VAL + 1) if x not in s]
                s.update(random.sample(remaining, N - len(s)))
            sc = count_aps(list(s))
            if sc > best_init_score:
                best_init_score = sc
                best_init = s
    
    # Strategy 4: Dense clusters with gaps
    for cluster_size in [4, 5, 6, 8]:
        s = set()
        num_clusters = N // cluster_size
        for c in range(num_clusters):
            start = c * (cluster_size + 2) + 1
            for i in range(cluster_size):
                val = start + i
                if val <= MAX_VAL:
                    s.add(val)
        if len(s) < N:
            remaining = [x for x in range(1, MAX_VAL + 1) if x not in s]
            s.update(random.sample(remaining, N - len(s)))
        sc = count_aps(list(s))
        if sc > best_init_score:
            best_init_score = sc
            best_init = s
    
    current_set = best_init
    current_score = best_init_score
    best_set = list(current_set)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.005
    
    # Helper for efficient set operations
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to change
        elements = list(current_set)
        idx = random.randint(0, N - 1)
        old_val = elements[idx]
        
        # Generate a new value - multiple strategies
        strategy = random.random()
        if strategy < 0.3:
            # Random value
            new_val = random.randint(1, MAX_VAL)
        elif strategy < 0.5:
            # Value near an existing element
            ref_val = random.choice(elements)
            offset = random.randint(-30, 30)
            new_val = ref_val + offset
            if new_val < 1:
                new_val = 1
            elif new_val > MAX_VAL:
                new_val = MAX_VAL
        elif strategy < 0.7:
            # Value that creates an AP with two existing elements
            # Pick two elements and try to form a 3-AP
            if len(elements) >= 2:
                i, j = random.sample(range(N), 2)
                a, b = elements[i], elements[j]
                if a < b:
                    # Try c = b + (b-a)
                    new_val = b + (b - a)
                    if new_val > MAX_VAL:
                        # Try c = a - (b-a)
                        new_val = a - (b - a)
                else:
                    # Try c = a + (a-b)
                    new_val = a + (a - b)
                    if new_val > MAX_VAL:
                        new_val = b - (a - b)
                if new_val < 1:
                    new_val = random.randint(1, MAX_VAL)
            else:
                new_val = random.randint(1, MAX_VAL)
        elif strategy < 0.85:
            # Swap with another element
            jdx = random.randint(0, N - 1)
            if jdx == idx:
                jdx = (jdx + 1) % N
            new_val = elements[jdx]
            # This is a swap, handle specially
            current_set.remove(old_val)
            current_set.add(new_val)
            new_score = count_aps(list(current_set))
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_set = list(current_set)
            else:
                current_set.remove(new_val)
                current_set.add(old_val)
            T *= cooling_rate
            continue
        else:
            # Move to a value that's at a "gap" position
            # Find common difference patterns
            new_val = random.randint(1, MAX_VAL)
        
        if new_val in current_set and new_val != old_val:
            continue
        
        # Apply change
        current_set.remove(old_val)
        current_set.add(new_val)
        
        new_score = count_aps(list(current_set))
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(current_set)
        else:
            # Reject: revert change
            current_set.remove(new_val)
            current_set.add(old_val)
            
        T *= cooling_rate
    
    return best_set
