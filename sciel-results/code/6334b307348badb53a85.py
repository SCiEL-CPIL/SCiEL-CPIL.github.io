import time
import random
import math

def _score_fast(s):
    """Fast score using bitmask."""
    mask = 0
    for x in s:
        mask |= (1 << x)
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _sa_optimize(s, deadline, temp_start=10.0, cooling=0.9995):
    """Simulated Annealing optimization."""
    current = list(s)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    current_set = set(current)
    n = len(current)
    
    T = temp_start
    while time.time() < deadline and T > 0.01:
        # Choose move
        r = random.random()
        if r < 0.5:
            # 1-swap
            i = random.randint(0, n-1)
            old = current[i]
            new = random.randint(1, 512)
            if new == old or new in current_set:
                continue
            current[i] = new
            current_set.remove(old)
            current_set.add(new)
            new_score = _score_fast(current)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[i] = old
                current_set.remove(new)
                current_set.add(old)
        elif r < 0.8:
            # 2-swap
            i, j = random.sample(range(n), 2)
            old1, old2 = current[i], current[j]
            new1 = random.randint(1, 512)
            new2 = random.randint(1, 512)
            if new1 in current_set or new2 in current_set or new1 == new2:
                continue
            current[i] = new1
            current[j] = new2
            current_set.remove(old1)
            current_set.remove(old2)
            current_set.add(new1)
            current_set.add(new2)
            new_score = _score_fast(current)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[i] = old1
                current[j] = old2
                current_set.remove(new1)
                current_set.remove(new2)
                current_set.add(old1)
                current_set.add(old2)
        else:
            # Shift
            i = random.randint(0, n-1)
            old = current[i]
            shift = random.randint(-20, 20)
            if shift == 0: continue
            new = old + shift
            if new < 1 or new > 512 or new in current_set:
                continue
            current[i] = new
            current_set.remove(old)
            current_set.add(new)
            new_score = _score_fast(current)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[i] = old
                current_set.remove(new)
                current_set.add(old)
                
        T *= cooling
    return best, best_score

def _greedy_complete(s, deadline):
    """Try to fill in missing small d's."""
    current = list(s)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    current_set = set(current)
    
    while time.time() < deadline:
        # Find missing d's
        mask = 0
        for x in current:
            mask |= (1 << x)
        
        missing = []
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) == 0:
                missing.append(d)
        
        if not missing:
            break
            
        # Pick a random missing d
        d = random.choice(missing)
        
        # Try to complete a 3-AP for this d
        # Look for a, a+d in S, try to add a+2d
        found_improvement = False
        for a in current:
            if a + d in current_set:
                target = a + 2 * d
                if target > 512:
                    continue
                if target in current_set:
                    continue # Already there? Then d shouldn't be missing.
                
                # Replace some random element with target
                # Heuristic: Replace an element that is "least useful"? 
                # For now, random replacement.
                i = random.randint(0, 31)
                old = current[i]
                if old == target:
                    continue
                
                current[i] = target
                current_set.remove(old)
                current_set.add(target)
                
                new_score = _score_fast(current)
                if new_score > current_score:
                    current_score = new_score
                    found_improvement = True
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                else:
                    # Revert
                    current[i] = old
                    current_set.remove(target)
                    current_set.add(old)
        
        if not found_improvement:
             # If no improvement for this d, try another d or break
             # To avoid infinite loop on same d, we rely on random choice
             pass

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    random.seed(42)
    
    best_score = 0
    best_set = []
    
    # Initializations
    inits = []
    
    # 1. Jittered Grid
    pts = []
    for i in range(32):
        p = int(i * 512 / 32) + random.randint(-10, 10)
        if p < 1: p = 1
        if p > 512: p = 512
        pts.append(p)
    while len(set(pts)) < 32:
        idx = random.randint(0, 31)
        pts[idx] = random.randint(1, 512)
    inits.append(pts)
    
    # 2. Dense Block
    inits.append(list(range(1, 33)))
    
    # 3. Spread (Powers)
    pts = []
    x = 1
    while len(pts) < 32 and x <= 512:
        pts.append(x)
        x = int(x * 1.15) + 1
    while len(pts) < 32:
        pts.append(pts[-1] + 1)
    inits.append(pts[:32])
    
    # 4. Random
    inits.append(random.sample(range(1, 513), 32))
    
    # 5. Clustered
    cluster = []
    for base in [1, 60, 120, 180, 240, 300, 360, 420]:
        for k in range(4):
            v = base + k * 3
            if 1 <= v <= 512:
                cluster.append(v)
    inits.append(cluster[:32])
    
    # Run optimization on each init
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # SA Phase
        sa_deadline = time.time() + min(20, deadline - time.time() - 5)
        s, sc = _sa_optimize(init, sa_deadline)
        
        # Greedy Phase
        g_deadline = time.time() + min(10, deadline - time.time() - 2)
        s, sc = _greedy_complete(s, g_deadline)
        
        if sc > best_score:
            best_score = sc
            best_set = s
            
        # If we have time, restart SA from this best
        # (Optional, might be redundant)
        
    # Final check
    if not best_set:
        return list(range(1, 33))
        
    return best_set
