import time
import random
import numpy as np

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _precompute_aps(n=128):
    """Precompute all 4-APs as (a, b, c, e) tuples and index by position."""
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                ap = (a, b, c, e)
                all_aps.append(ap)
                for p in ap:
                    pos_to_aps[p].append(ap)
    return all_aps, pos_to_aps

def _greedy_bitmask(n=128, seed=0):
    """
    Greedy assignment using bitmasks for fast conflict checking.
    For each position, determine which colors are valid given previously assigned positions.
    """
    rng = random.Random(seed)
    all_aps, pos_to_aps = _precompute_aps(n)
    colors = [-1] * n
    
    # For each position, precompute which APs it participates in
    # and for each AP, which other positions are involved
    
    for i in range(n):
        # valid_mask: bit k set means color k is still possible
        valid_mask = 0b111  # colors 0, 1, 2 all possible initially
        
        for ap in pos_to_aps[i]:
            a, b, c, e = ap
            # Get the other three positions
            others = [p for p in (a, b, c, e) if p != i]
            # Check if all others are assigned
            if all(colors[p] != -1 for p in others):
                # Count distinct colors among others
                other_colors = set(colors[p] for p in others)
                # For each candidate color, check if adding it gives >= 3 distinct
                for col in range(3):
                    if len(other_colors | {col}) < 3:
                        valid_mask &= ~(1 << col)
        
        if valid_mask == 0:
            return None
        
        # Choose a color from valid ones, preferring already-used colors
        used_colors = set(colors[:i]) if i > 0 else set()
        valid_list = [col for col in range(3) if valid_mask & (1 << col)]
        
        # Prefer colors already in use
        preferred = [col for col in valid_list if col in used_colors]
        if preferred:
            colors[i] = rng.choice(preferred)
        else:
            colors[i] = rng.choice(valid_list)
    
    return colors

def _local_search(colors, n=128, max_iters=20000, seed=0, time_limit=5.0):
    """
    Local search to minimize number of colors used.
    Tries to recolor positions to reduce the total distinct color count.
    """
    rng = random.Random(seed)
    colors = list(colors)
    start_time = time.time()
    
    all_aps, pos_to_aps = _precompute_aps(n)
    
    def get_color_count(cols):
        return len(set(cols))
    
    current_count = get_color_count(colors)
    if current_count <= 3:
        return colors
    
    # Map colors to 0,1,2 for internal use
    color_map = {}
    next_id = 0
    for c in sorted(set(colors)):
        color_map[c] = next_id
        next_id += 1
    
    internal = [color_map[c] for c in colors]
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random position to try recoloring
        i = rng.randrange(n)
        old_color = internal[i]
        
        # Try each possible color
        best_new_color = old_color
        best_count = current_count
        
        for new_color in range(3):
            if new_color == old_color:
                continue
            
            internal[i] = new_color
            
            # Quick validation: check only APs containing position i
            valid = True
            for ap in pos_to_aps[i]:
                a, b, c, e = ap
                cols = {internal[a], internal[b], internal[c], internal[e]}
                if len(cols) < 3:
                    valid = False
                    break
            
            if valid:
                # Count colors
                new_count = len(set(internal))
                if new_count < best_count:
                    best_count = new_count
                    best_new_color = new_color
            else:
                internal[i] = old_color
        
        if best_new_color != old_color:
            internal[i] = best_new_color
            current_count = best_count
            if current_count <= 3:
                break
        else:
            # No improvement from this position; try swapping two positions
            if rng.random() < 0.3:
                j = rng.randrange(n)
                if j == i:
                    continue
                c1, c2 = internal[i], internal[j]
                if c1 == c2:
                    continue
                internal[i], internal[j] = c2, c1
                
                # Validate APs containing i or j
                valid = True
                for ap in pos_to_aps[i]:
                    a, b, c, e = ap
                    cols = {internal[a], internal[b], internal[c], internal[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if valid:
                    for ap in pos_to_aps[j]:
                        a, b, c, e = ap
                        cols = {internal[a], internal[b], internal[c], internal[e]}
                        if len(cols) < 3:
                            valid = False
                            break
                
                if not valid:
                    internal[i], internal[j] = c1, c2
                else:
                    # Count colors after swap
                    new_count = len(set(internal))
                    if new_count < current_count:
                        current_count = new_count
                        if current_count <= 3:
                            break
    
    # Convert back to original color labels
    inv_map = {v: k for k, v in color_map.items()}
    return [inv_map[c] for c in internal]

def _try_pattern(n=128):
    """Try known pattern-based constructions."""
    # 2-adic valuation mod 3
    colors = []
    for x in range(1, n + 1):
        v2 = 0
        tmp = x
        while tmp % 2 == 0:
            tmp //= 2
            v2 += 1
        colors.append(v2 % 3)
    if _validate(colors):
        return colors
    
    # Sum of bits mod 3
    colors = []
    for x in range(1, n + 1):
        s = bin(x - 1).count('1')
        colors.append(s % 3)
    if _validate(colors):
        return colors
    
    # (x-1) mod 3
    colors = [(x - 1) % 3 for x in range(1, n + 1)]
    if _validate(colors):
        return colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Try pattern-based construction first (fast)
    result = _try_pattern(n)
    if result is not None:
        if len(set(result)) == 3:
            return result
    
    # Greedy with bitmask for multiple seeds
    best = None
    best_count = 128
    
    for seed in range(500):
        if time.time() > deadline - 10:
            break
        result = _greedy_bitmask(n, seed=seed)
        if result is not None and _validate(result):
            count = len(set(result))
            if count < best_count:
                best = result
                best_count = count
                if best_count == 3:
                    return best
    
    if best is None:
        # Fallback: simple greedy
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 1 - 3 * d
                    b = x - 1 - 2 * d
                    c = x - 1 - d
                    if a >= 0:
                        if len({colors[a], colors[b], colors[c], color}) < 3:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    break
        return colors
    
    # Local search to reduce colors further
    remaining = deadline - time.time()
    if remaining > 5 and best_count > 3:
        for seed in range(20):
            if time.time() > deadline - 2:
                break
            result = _local_search(best, n, max_iters=10000, seed=seed, 
                                  time_limit=min(3.0, remaining - 3))
            count = len(set(result))
            if count < best_count:
                best = result
                best_count = count
                if best_count == 3:
                    return best
    
    return best
