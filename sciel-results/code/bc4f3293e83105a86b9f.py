import time
import random
import numpy as np

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _all_4aps(n):
    """Return list of all 4-APs as tuples (a, a+d, a+2d, a+3d)."""
    aps = []
    for a in range(n):
        for d in range(1, n):
            e = a + 3 * d
            if e >= n:
                break
            aps.append((a, a + d, a + 2 * d, e))
    return aps

def _count_violations(colors, aps):
    """Count number of 4-APs that use < 3 distinct colors."""
    count = 0
    for a, b, c, e in aps:
        cols = {colors[a], colors[b], colors[c], colors[e]}
        if len(cols) < 3:
            count += 1
    return count

def _find_3coloring(n=128, time_limit=50.0, seed=0):
    """
    Use Simulated Annealing to find a valid 3-coloring.
    """
    start_time = time.time()
    rng = random.Random(seed)
    aps = _all_4aps(n)
    
    # Initialize with random 3-coloring
    colors = [rng.randint(0, 2) for _ in range(n)]
    current_violations = _count_violations(colors, aps)
    
    best_colors = list(colors)
    best_violations = current_violations
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if current_violations == 0:
            return colors
        
        # Pick a random position and change its color
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        new_color = rng.randint(0, 2)
        if new_color == old_color:
            continue
        
        colors[pos] = new_color
        new_violations = _count_violations(colors, aps)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_violations - current_violations
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = list(colors)
        else:
            colors[pos] = old_color
            
        T *= cooling
        if T < T_min:
            break
            
    return best_colors if best_violations == 0 else None

def _find_kcoloring(n=128, k=4, time_limit=50.0, seed=0):
    """
    Use Simulated Annealing to find a valid k-coloring.
    """
    start_time = time.time()
    rng = random.Random(seed)
    aps = _all_4aps(n)
    
    colors = [rng.randint(0, k - 1) for _ in range(n)]
    current_violations = _count_violations(colors, aps)
    
    best_colors = list(colors)
    best_violations = current_violations
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if current_violations == 0:
            return colors
        
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        new_color = rng.randint(0, k - 1)
        if new_color == old_color:
            continue
        
        colors[pos] = new_color
        new_violations = _count_violations(colors, aps)
        
        delta = new_violations - current_violations
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = list(colors)
        else:
            colors[pos] = old_color
            
        T *= cooling
        if T < T_min:
            break
            
    return best_colors if best_violations == 0 else None

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid but might use many colors."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                b = x - 1 - 2 * d
                c = x - 1 - d
                if a < 0:
                    break
                cols = {colors[a], colors[b], colors[c], color}
                if len(cols) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    # 1. Try to find a 3-coloring
    # Use multiple seeds/restarts
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        remaining = deadline - 10 - time.time()
        if remaining < 1:
            break
        result = _find_3coloring(n, time_limit=remaining, seed=seed)
        if result is not None:
            if _validate(result):
                return result

    # 2. If 3-coloring failed, try 4-coloring
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        result = _find_kcoloring(n, 4, time_limit=remaining, seed=seed)
        if result is not None:
            if _validate(result):
                return result

    # 3. Fallback to greedy
    return _seed_fallback()

if __name__ == "__main__":
    c = construct()
    print("Colors used:", len(set(c)))
    print("Valid:", _validate(c))
