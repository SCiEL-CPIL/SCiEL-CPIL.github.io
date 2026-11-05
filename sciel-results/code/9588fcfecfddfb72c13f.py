"""Rewritten program: find a 32-element set in 1..512 maximizing distinct 3-AP differences."""
import time
import random
from itertools import combinations

def _score(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in set s."""
    sset = set(s)
    diffs = set()
    for a in s:
        for b in s:
            if b == a:
                continue
            d = b - a
            if d <= 0:
                continue
            if a + 2 * d in sset:
                diffs.add(d)
    return len(diffs)

def _score_fast(s):
    """Fast score using bitmask over 1..512. Returns count of distinct d with a 3-AP."""
    # Build a bitmask: bit i set if (i+1) is in s
    mask = 0
    for x in s:
        mask |= (1 << x)
    # For each d from 1 to 255 (max d: need a, a+d, a+2d <= 512, so 2d <= 511, d <= 255)
    count = 0
    for d in range(1, 256):
        # Check if there exists a such that bits a, a+d, a+2d are all set
        # a ranges from 1 to 512 - 2d
        # We can check: (mask & (mask >> d) & (mask >> (2*d))) != 0
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _local_search(s, deadline, max_iter=5000):
    """Greedy local search with 1-swap and 2-swap moves."""
    best = list(s)
    best_score = _score_fast(best)
    n = len(best)
    best_set = set(best)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try 1-element swap: remove one, add another
        for i in range(n):
            removed = best[i]
            best_set.remove(removed)
            
            # Try random candidates
            candidates = set()
            for _ in range(30):
                c = random.randint(1, 512)
                candidates.add(c)
            # Also try neighbors
            for delta in range(-15, 16):
                c = removed + delta
                if 1 <= c <= 512:
                    candidates.add(c)
            
            for c in candidates:
                if c in best_set:
                    continue
                best_set.add(c)
                new_s = list(best)
                new_s[i] = c
                ns = _score_fast(new_s)
                if ns > best_score:
                    best_score = ns
                    best = new_s
                    improved = True
                    break
                best_set.remove(c)
            
            if improved:
                break
            best_set.add(removed)
        
        # Try 2-element swap: remove two, add two
        if not improved:
            for _ in range(50):
                if time.time() > deadline:
                    break
                i, j = random.sample(range(n), 2)
                r1, r2 = best[i], best[j]
                best_set.discard(r1)
                best_set.discard(r2)
                
                c1 = random.randint(1, 512)
                c2 = random.randint(1, 512)
                if c1 in best_set or c2 in best_set or c1 == c2:
                    best_set.add(r1)
                    best_set.add(r2)
                    continue
                
                best_set.add(c1)
                best_set.add(c2)
                new_s = list(best)
                new_s[i] = c1
                new_s[j] = c2
                ns = _score_fast(new_s)
                if ns > best_score:
                    best_score = ns
                    best = new_s
                    improved = True
                    break
                best_set.remove(c1)
                best_set.remove(c2)
                best_set.add(r1)
                best_set.add(r2)
            
            if not improved:
                break
    
    return best, best_score

def _simulated_annealing(s, deadline, max_iter=50000):
    """Simulated annealing with 1-swap and occasional 2-swap moves."""
    best = list(s)
    best_score = _score_fast(best)
    current = list(best)
    current_score = best_score
    n = len(current)
    current_set = set(current)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        if T < T_min:
            break
        
        # Occasionally do a 2-swap
        if random.random() < 0.15:
            i, j = random.sample(range(n), 2)
            r1, r2 = current[i], current[j]
            current_set.discard(r1)
            current_set.discard(r2)
            
            c1 = random.randint(1, 512)
            c2 = random.randint(1, 512)
            attempts = 0
            while (c1 in current_set or c2 in current_set or c1 == c2) and attempts < 50:
                c1 = random.randint(1, 512)
                c2 = random.randint(1, 512)
                attempts += 1
            
            if c1 in current_set or c2 in current_set or c1 == c2:
                current_set.add(r1)
                current_set.add(r2)
                continue
            
            current_set.add(c1)
            current_set.add(c2)
            new_s = list(current)
            new_s[i] = c1
            new_s[j] = c2
            new_score = _score_fast(new_s)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current = new_s
                current_score = new_score
                current_set = set(current)
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current_set.remove(c1)
                current_set.remove(c2)
                current_set.add(r1)
                current_set.add(r2)
        else:
            # 1-swap
            i = random.randint(0, n - 1)
            removed = current[i]
            current_set.remove(removed)
            
            new_elem = random.randint(1, 512)
            attempts = 0
            while new_elem in current_set and attempts < 50:
                new_elem = random.randint(1, 512)
                attempts += 1
            
            if new_elem in current_set:
                current_set.add(removed)
                continue
            
            current_set.add(new_elem)
            new_s = list(current)
            new_s[i] = new_elem
            new_score = _score_fast(new_s)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current = new_s
                current_score = new_score
                current_set = set(current)
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current_set.remove(new_elem)
                current_set.add(removed)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2 second safety margin
    
    random.seed(42)
    
    # Diverse initializations
    initial_sets = []
    
    # 1. Contiguous dense
    initial_sets.append(list(range(1, 33)))
    
    # 2. 5-smooth numbers
    smooth5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90]
    if len(smooth5) >= 32:
        initial_sets.append(list(smooth5[:32]))
    
    # 3. Geometric-ish spread
    geo = []
    x = 1
    while len(geo) < 32 and x <= 512:
        geo.append(x)
        x = int(x * 1.15) + 1
    while len(geo) < 32:
        geo.append(geo[-1] + 1)
    initial_sets.append(geo[:32])
    
    # 4. Random
    initial_sets.append(random.sample(range(1, 513), 32))
    
    # 5. Interleaved (evens then odds)
    interleaved = list(range(1, 513, 2))[:16] + list(range(2, 513, 2))[:16]
    initial_sets.append(interleaved)
    
    # 6. Another random
    initial_sets.append(random.sample(range(1, 513), 32))
    
    # 7. Clustered
    cluster = []
    for base in [1, 60, 120, 180, 240, 300, 360, 420]:
        for k in range(4):
            v = base + k * 3
            if 1 <= v <= 512:
                cluster.append(v)
    initial_sets.append(cluster[:32])
    
    # Evaluate all initial sets
    best = None
    best_score = -1
    for s in initial_sets:
        if len(s) != 32 or len(set(s)) != 32:
            continue
        sc = _score_fast(s)
        if sc > best_score:
            best_score = sc
            best = list(s)
    
    # Run local search from best
    if time.time() < deadline - 30:
        s, sc = _local_search(best, deadline - 30, max_iter=20000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Run simulated annealing from best
    if time.time() < deadline - 10:
        s, sc = _simulated_annealing(best, deadline - 10, max_iter=100000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Try more random restarts if time permits
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        
        random_set = random.sample(range(1, 513), 32)
        rs = _score_fast(random_set)
        
        if rs > best_score - 3:
            deadline_ls = deadline - 2
            s, sc = _local_search(random_set, deadline_ls, max_iter=5000)
            if sc > best_score:
                best_score = sc
                best = s
    
    # Final validation
    assert len(best) == 32
    assert len(set(best)) == 32
    assert all(1 <= x <= 512 for x in best)
    
    return best

# EVOLVE-BLOCK-START
# The actual construct() is defined above; keep the block marker for compatibility
# EVOLVE-BLOCK-END
