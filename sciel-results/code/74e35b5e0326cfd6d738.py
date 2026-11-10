import time
import random
import numpy as np
from array import array

def _try_greedy_fast(order, deadline):
    """Fast greedy using array for counts."""
    chosen = []
    counts = array('b', [0] * 2001)
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Check validity
        if counts[2 * x] >= 2:
            continue
            
        valid = True
        for y in chosen:
            if counts[x + y] >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            counts[2 * x] += 1
            for y in chosen[:-1]:
                counts[x + y] += 1
                
    return chosen

def _local_search(initial_set, deadline, max_iter=1000):
    """Try to improve a set by swapping elements or adding/removing."""
    chosen = sorted(initial_set)
    in_set = bytearray(1001)
    for x in chosen:
        in_set[x] = 1
        
    counts = array('b', [0] * 2001)
    for i, a in enumerate(chosen):
        for b in chosen[i:]:
            counts[a + b] += 1
            
    best_size = len(chosen)
    best_set = list(chosen)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a candidate
        for x in range(1, 1001):
            if in_set[x]:
                continue
                
            if counts[2 * x] >= 2:
                continue
                
            valid = True
            for y in chosen:
                if counts[x + y] >= 2:
                    valid = False
                    break
                    
            if valid:
                # Add x
                chosen.append(x)
                in_set[x] = 1
                counts[2 * x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
                    
                best_size = len(chosen)
                best_set = list(chosen)
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping: remove one, add another
        if chosen:
            idx = random.randint(0, len(chosen) - 1)
            x = chosen[idx]
            
            # Remove x
            chosen.pop(idx)
            in_set[x] = 0
            for y in chosen:
                counts[x + y] -= 1
            counts[2 * x] -= 1
            
            # Try to add a random candidate
            improved_swap = False
            for _ in range(10):
                y = random.randint(1, 1000)
                if in_set[y]:
                    continue
                    
                if counts[2 * y] >= 2:
                    continue
                    
                valid = True
                for z in chosen:
                    if counts[y + z] >= 2:
                        valid = False
                        break
                        
                if valid:
                    # Add y
                    chosen.append(y)
                    in_set[y] = 1
                    counts[2 * y] += 1
                    for z in chosen[:-1]:
                        counts[y + z] += 1
                        
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_set = list(chosen)
                    improved_swap = True
                    break
                    
            if not improved_swap:
                # Restore x
                chosen.insert(idx, x)
                in_set[x] = 1
                for y in chosen:
                    if y != x:
                        counts[x + y] += 1
                counts[2 * x] += 1

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Ascending)
    s1 = _try_greedy_fast(range(1, 1001), deadline)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # 2. Deterministic Greedy (Descending)
    s2 = _try_greedy_fast(range(1000, 0, -1), deadline)
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # 3. Randomized Greedy + Local Search
    rng = random.Random(42)
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = _try_greedy_fast(order, deadline)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # Local search on the current best (or current s if it's better)
        if time.time() < deadline - 2.0:
            improved = _local_search(s, deadline - 2.0, max_iter=200)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
                
    return best_set
