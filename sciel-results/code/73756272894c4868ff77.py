"""Combined program: maximize distinct 3-AP differences for 32-element set in 1..512."""
import time
import random
import math

def _score_fast(s):
    """Fast score using bitmask over 1..512. Returns count of distinct d with a 3-AP."""
    mask = 0
    for x in s:
        mask |= (1 << x)
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _local_search(s, deadline, max_iter=20000):
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
            
            candidates = set()
            for _ in range(25):
                c = random.randint(1, 512)
                candidates.add(c)
            for delta in range(-12, 13):
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
        
        # Try 2-element swap
        if not improved:
            for _ in range(40):
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

def _sa_optimize(s, deadline, temp_start=8.0, cooling=0.9997, min_temp=0.05):
    """Simulated annealing with diverse moves: replace, shift, swap, 2-replace."""
    current = list(s)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    n = len(current)
    current_set = set(current)
    
    T = temp_start
    
    while T > min_temp and time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.45:
            # Replace one element
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            new_val = random.randint(1, 512)
            if new_val == old_val or new_val in current_set:
                continue
            
            current[idx] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            
            new_score = _score_fast(current)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[idx] = old_val
                current_set.remove(new_val)
                current_set.add(old_val)
                
        elif move_type < 0.70:
            # Shift one element by small amount
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            shift = random.randint(-8, 8)
            if shift == 0:
                continue
            new_val = old_val + shift
            if new_val < 1 or new_val > 512 or new_val in current_set:
                continue
            
            current[idx] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            
            new_score = _score_fast(current)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[idx] = old_val
                current_set.remove(new_val)
                current_set.add(old_val)
                
        elif move_type < 0.85:
            # Swap values between two positions
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            if idx1 == idx2:
                continue
            
            v1 = current[idx1]
            v2 = current[idx2]
            
            current[idx1] = v2
            current[idx2] = v1
            
            new_score = _score_fast(current)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[idx1] = v1
                current[idx2] = v2
        else:
            # 2-replace: remove two, add two
            i, j = random.sample(range(n), 2)
            r1, r2 = current[i], current[j]
            current_set.discard(r1)
            current_set.discard(r2)
            
            c1 = random.randint(1, 512)
            c2 = random.randint(1, 512)
            attempts = 0
            while (c1 in current_set or c2 in current_set or c1 == c2) and attempts < 30:
                c1 = random.randint(1, 512)
                c2 = random.randint(1, 512)
                attempts += 1
            
            if c1 in current_set or c2 in current_set or c1 == c2:
                current_set.add(r1)
                current_set.add(r2)
                continue
            
            current_set.add(c1)
            current_set.add(c2)
            current[i] = c1
            current[j] = c2
            
            new_score = _score_fast(current)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[i] = r1
                current[j] = r2
                current_set.remove(c1)
                current_set.remove(c2)
                current_set.add(r1)
                current_set.add(r2)
        
        T *= cooling
    
    return best, best_score

def _polish(s, deadline):
    """Final high-precision polish: tight neighborhood local search."""
    best = list(s)
    best_score = _score_fast(best)
    n = len(best)
    best_set = set(best)
    
    # Multiple passes with shrinking neighborhood
    for radius in [5, 3, 2, 1]:
        if time.time() > deadline:
            break
        for _ in range(3000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                removed = best[i]
                best_set.remove(removed)
                for delta in range(-radius, radius + 1):
                    if delta == 0:
                        continue
                    c = removed + delta
                    if c < 1 or c > 512 or c in best_set:
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
            if not improved:
                break
    
    return best, best_score

def _init_dense_small_d():
    """Initialize to cover small differences densely via greedy placement."""
    # Start with a base set that covers d=1..k well
    # Use a structured set: start with 1..16, then add spread elements
    s = list(range(1, 17))
    # Add elements to cover more differences
    random.shuffle(s)
    s.sort()
    
    # Add 16 more elements spread across the range
    used = set(s)
    candidates = list(range(17, 513))
    random.shuffle(candidates)
    
    for c in candidates:
        if len(s) >= 32:
            break
        if c not in used:
            s.append(c)
            used.add(c)
    
    # Fill if needed
    if len(s) < 32:
        for c in range(1, 513):
            if len(s) >= 32:
                break
            if c not in used:
                s.append(c)
                used.add(c)
    
    return s[:32]

def _init_spread_jittered():
    """Jittered grid initialization."""
    points = []
    step = 512 / 32.0
    for i in range(32):
        p = int(i * step) + random.randint(0, max(1, int(step)))
        p = max(1, min(512, p))
        points.append(p)
    while len(set(points)) < 32:
        idx = random.randint(0, 31)
        points[idx] = random.randint(1, 512)
    return list(set(points))[:32]

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # 3s safety margin before 150s kill
    
    random.seed(42)
    
    # Generate diverse initializations
    initial_sets = []
    
    # 1. Contiguous
    initial_sets.append(list(range(1, 33)))
    
    # 2. Dense small-d greedy
    initial_sets.append(_init_dense_small_d())
    
    # 3. Jittered spread
    initial_sets.append(_init_spread_jittered())
    
    # 4. Random
    initial_sets.append(random.sample(range(1, 513), 32))
    
    # 5. 5-smooth
    smooth5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90]
    if len(smooth5) >= 32:
        initial_sets.append(list(smooth5[:32]))
    
    # 6. Geometric
    geo = []
    x = 1
    while len(geo) < 32 and x <= 512:
        geo.append(x)
        x = int(x * 1.12) + 1
    while len(geo) < 32:
        geo.append(min(512, geo[-1] + 1))
    initial_sets.append(geo[:32])
    
    # 7. Another random
    initial_sets.append(random.sample(range(1, 513), 32))
    
    # 8. Clustered
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
        if not all(1 <= x <= 512 for x in s):
            continue
        sc = _score_fast(s)
        if sc > best_score:
            best_score = sc
            best = list(s)
    
    # Phase 1: Local search from best
    if time.time() < deadline - 40:
        s, sc = _local_search(best, deadline - 40, max_iter=15000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Phase 2: Multiple SA runs from diverse starts
    sa_starts = [best]
    # Add a few other good initializations
    for s in initial_sets:
        if len(s) == 32 and len(set(s)) == 32 and s != best:
            sc = _score_fast(s)
            if sc > best_score - 5:
                sa_starts.append(s)
    
    # Run SA from top starts
    for idx, start_s in enumerate(sa_starts[:5]):
        if time.time() > deadline - 20:
            break
        
        # Allocate time: more time for first run, less for subsequent
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        alloc = min(remaining / max(1, 5 - idx), 30.0)
        
        s, sc = _sa_optimize(start_s, time.time() + alloc, temp_start=8.0, cooling=0.9997, min_temp=0.05)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Phase 3: Polish
    if time.time() < deadline - 5:
        s, sc = _polish(best, deadline - 2)
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
