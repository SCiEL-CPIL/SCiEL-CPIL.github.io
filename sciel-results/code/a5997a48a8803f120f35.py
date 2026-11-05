"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys
from collections import defaultdict

def _triple_sums_involving_x(x, S):
    """Return the set of triple sums that involve x, given x is NOT in S."""
    sums = set()
    n = len(S)
    sums.add(3 * x)
    for y in S:
        sums.add(2 * x + y)
        sums.add(x + 2 * y)
    for i in range(n):
        for j in range(i + 1, n):
            sums.add(x + S[i] + S[j])
    return sums

def _compute_all_triple_sums(S):
    """Compute all triple sums for set S. Returns a dict mapping sum -> list of triples."""
    sums_dict = defaultdict(list)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                sums_dict[s].append((i, j, k))
    return sums_dict

def _greedy_build(deadline, seed=0):
    """Optimized greedy build. Returns the largest valid set found."""
    S = []
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        n = len(S)
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(n):
            for j in range(i + 1, n):
                new_sums.add(x + S[i] + S[j])
        
        expected = 1 + n + n + n * (n - 1) // 2
        if len(new_sums) < expected:
            continue
        if new_sums & existing_sums:
            continue
        
        S.append(x)
        existing_sums |= new_sums
    
    return S

def _local_search(S, deadline, seed=0):
    """Hill-climbing: try to add elements or swap to increase cardinality."""
    rng = random.Random(seed)
    best = S[:]
    best_size = len(S)
    
    # Build current triple sums set
    current_sums = set()
    for x in best:
        current_sums |= _triple_sums_involving_x(x, best)
    
    # Try adding elements
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if x in set(best):
            continue
        
        new_sums = _triple_sums_involving_x(x, best)
        expected = 1 + len(best) + len(best) + len(best) * (len(best) - 1) // 2
        if len(new_sums) < expected:
            continue
        if new_sums & current_sums:
            continue
        
        # Try adding x
        new_best = best + [x]
        new_sums_set = current_sums | new_sums
        # Verify validity
        all_sums = set()
        for y in new_best:
            all_sums |= _triple_sums_involving_x(y, [z for z in new_best if z != y] or new_best)
        
        # Simpler: just check no collisions
        if len(new_sums_set) == len(current_sums) + len(new_sums - current_sums):
            best = new_best
            current_sums = new_sums_set
            best_size = len(best)
    
    # Try swaps: remove one element, add another
    for _ in range(200):
        if time.time() > deadline:
            break
        
        if not best:
            break
        
        # Pick a random element to remove
        idx = rng.randint(0, len(best) - 1)
        removed = best[idx]
        S_without = best[:idx] + best[idx+1:]
        
        # Compute sums without removed element
        sums_without = set()
        for y in S_without:
            sums_without |= _triple_sums_involving_x(y, S_without)
        
        # Try to add a new element
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            if x in set(S_without):
                continue
            
            new_sums = _triple_sums_involving_x(x, S_without)
            expected = 1 + len(S_without) + len(S_without) + len(S_without) * (len(S_without) - 1) // 2
            if len(new_sums) < expected:
                continue
            if new_sums & sums_without:
                continue
            
            new_best = S_without + [x]
            if len(new_best) > best_size:
                best = new_best
                best_size = len(best)
                break
            elif len(new_best) == best_size and rng.random() < 0.1:
                # Accept with small probability for diversity
                best = new_best
                break
    
    return best

def _greedy_with_randomization(deadline, seed=0):
    """Greedy with randomized ordering."""
    rng = random.Random(seed)
    best = []
    
    for attempt in range(8):
        if time.time() > deadline:
            break
        
        order = list(range(1, 4097))
        small = order[:100]
        large = order[100:]
        rng.shuffle(large)
        order = small + large
        
        S = []
        existing_sums = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            n = len(S)
            new_sums = set()
            new_sums.add(3 * x)
            for y in S:
                new_sums.add(2 * x + y)
                new_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i + 1, n):
                    new_sums.add(x + S[i] + S[j])
            
            expected = 1 + n + n + n * (n - 1) // 2
            if len(new_sums) < expected:
                continue
            if new_sums & existing_sums:
                continue
            
            S.append(x)
            existing_sums |= new_sums
        
        if len(S) > len(best):
            best = S[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Strategy 1: Greedy in increasing order
    S1 = _greedy_build(deadline, seed=42)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Local search to improve
    if time.time() < deadline:
        S2 = _local_search(best, deadline, seed=123)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Randomized greedy
    if time.time() < deadline:
        S3 = _greedy_with_randomization(deadline, seed=456)
        if len(S3) > len(best):
            best = S3[:]
    
    # Final local search on best
    if time.time() < deadline:
        S4 = _local_search(best, deadline, seed=789)
        if len(S4) > len(best):
            best = S4[:]
    
    return best
