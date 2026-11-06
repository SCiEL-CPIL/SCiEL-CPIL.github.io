import time
import random
import numpy as np

def _precompute_aps(n=128):
    """Precompute all 4-APs and index by last position for efficient checking."""
    # For each position i, store APs where i is the LAST element: (a, b, c, i) where i = a+3d
    last_aps = [[] for _ in range(n)]
    all_aps = []
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                ap = (a, b, c, e)
                all_aps.append(ap)
                last_aps[e].append((a, b, c))
    return all_aps, last_aps

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

def _greedy_construct(n=128, seed=0, num_colors=3):
    """Greedy construction: assign positions 0..n-1, checking constraints."""
    rng = random.Random(seed)
    colors = [-1] * n
    last_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            e = a + 3 * d
            if e < n:
                last_aps[e].append((a, a + d, a + 2 * d))
    
    for i in range(n):
        valid = list(range(num_colors))
        for (a, b, c) in last_aps[i]:
            c1, c2, c3 = colors[a], colors[b], colors[c]
            # Need {c1, c2, c3, color} to have >= 3 distinct
            # So color cannot make the set have < 3 distinct
            # If {c1,c2,c3} already has >= 3, any color works
            # If {c1,c2,c3} has exactly 2, color must be different from both
            # If {c1,c2,c3} has 1, color must be different from that one
            existing = {c1, c2, c3}
            if len(existing) == 1:
                # color must differ from the single value
                valid = [c for c in valid if c != c1]
            elif len(existing) == 2:
                # color must differ from both values
                valid = [c for c in valid if c not in existing]
            # if len(existing) == 3, no constraint
        if not valid:
            return None
        # Prefer already-used colors to minimize count
        used = set(colors[:i])
        preferred = [c for c in valid if c in used]
        if preferred:
            colors[i] = rng.choice(preferred)
        else:
            colors[i] = rng.choice(valid)
    return colors

def _sa_reduce_colors(colors, n=128, max_time=15.0, seed=0):
    """Simulated annealing to minimize distinct colors in a valid coloring."""
    rng = random.Random(seed)
    current = list(colors)
    current_colors = set(current)
    current_count = len(current_colors)
    best = list(current)
    best_count = current_count
    
    # Precompute last_aps for constraint checking
    last_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            e = a + 3 * d
            if e < n:
                last_aps[e].append((a, a + d, a + 2 * d))
    
    # Track color counts for efficiency
    color_counts = {}
    for c in current:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    start_time = time.time()
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        pos = rng.randrange(n)
        old_color = current[pos]
        
        # Choose new color: bias toward existing colors
        if rng.random() < 0.85:
            new_color = rng.choice(list(current_colors))
        else:
            new_color = rng.randrange(128)
        
        if new_color == old_color:
            continue
        
        # Check validity: only APs where pos is the last element need checking
        valid = True
        for (a, b, c) in last_aps[pos]:
            cols = {colors_val for colors_val in [current[a], current[b], current[c], new_color]}
            if len(cols) < 3:
                valid = False
                break
        if not valid:
            continue
        
        # Apply change
        current[pos] = new_color
        color_counts[old_color] -= 1
        if color_counts[old_color] == 0:
            del color_counts[old_color]
            current_colors.discard(old_color)
        color_counts[new_color] = color_counts.get(new_color, 0) + 1
        current_colors.add(new_color)
        
        new_count = len(current_colors)
        delta = new_count - current_count
        current_count = new_count
        
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            if new_count < best_count:
                best = list(current)
                best_count = new_count
                if best_count <= 3:
                    break
        else:
            # Revert
            current[pos] = old_color
            color_counts[new_color] -= 1
            if color_counts[new_color] == 0:
                del color_counts[new_color]
                current_colors.discard(new_color)
            color_counts[old_color] = color_counts.get(old_color, 0) + 1
            current_colors.add(old_color)
            current_count = len(current_colors)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best

def _pattern_2adic(n=128):
    """Color by 2-adic valuation mod 3."""
    colors = []
    for x in range(1, n + 1):
        v2 = 0
        tmp = x
        while tmp % 2 == 0:
            tmp //= 2
            v2 += 1
        colors.append(v2 % 3)
    return colors

def _pattern_bitcount(n=128):
    """Color by bit count mod 3."""
    colors = []
    for x in range(1, n + 1):
        colors.append(bin(x - 1).count('1') % 3)
    return colors

def _pattern_mod(n=128):
    """Color by (x-1) mod 3."""
    return [(x - 1) % 3 for x in range(1, n + 1)]

def _pattern_random(n=128, seed=0):
    """Random 3-coloring."""
    rng = random.Random(seed)
    return [rng.randrange(3) for _ in range(n)]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    best = None
    best_count = 128
    
    # Phase 1: Try pattern-based initializations
    patterns = [
        _pattern_2adic(n),
        _pattern_bitcount(n),
        _pattern_mod(n),
    ]
    for pat in patterns:
        if time.time() > deadline - 30:
            break
        if _validate(pat):
            count = len(set(pat))
            if count < best_count:
                best = pat
                best_count = count
                if best_count == 3:
                    return best
    
    # Phase 2: Greedy constructions with many seeds
    if time.time() < deadline - 25:
        for seed in range(500):
            if time.time() > deadline - 25:
                break
            result = _greedy_construct(n, seed=seed, num_colors=3)
            if result is not None and _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
            # Also try with 2 colors (will likely fail but check)
            result2 = _greedy_construct(n, seed=seed + 1000, num_colors=2)
            if result2 is not None and _validate(result2):
                count = len(set(result2))
                if count < best_count:
                    best = result2
                    best_count = count
                    if best_count == 2:
                        return best
    
    # Phase 3: If we have a valid coloring with >3 colors, use SA to reduce
    if best is None:
        # Fallback: use greedy with more colors to guarantee validity
        for num_c in [4, 5, 6]:
            result = _greedy_construct(n, seed=42, num_colors=num_c)
            if result is not None and _validate(result):
                best = result
                best_count = len(set(result))
                break
        if best is None:
            # Last resort: brute force sequential
            colors = []
            for x in range(1, n + 1):
                for color in range(128):
                    ok = True
                    for d in range(1, (x - 1) // 3 + 1):
                        a = x - 3 * d
                        b = x - 2 * d
                        c = x - d
                        if a < 1:
                            break
                        cols = {colors[a - 1], colors[b - 1], colors[c - 1], color}
                        if len(cols) < 3:
                            ok = False
                            break
                    if ok:
                        colors.append(color)
                        break
            best = colors
            best_count = len(set(best))
    
    # Phase 4: SA to reduce colors further
    if best_count > 3 and time.time() < deadline - 5:
        sa_time = min(deadline - time.time() - 3, 40.0)
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            t_per = sa_time / (20 - seed) if seed < 19 else sa_time
            result = _sa_reduce_colors(best, n, max_time=min(t_per, 5.0), seed=seed)
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count <= 3:
                        break
    
    # Phase 5: Final polish - try more SA iterations if still > 3
    if best_count > 3 and time.time() < deadline - 2:
        remaining = deadline - time.time() - 2
        result = _sa_reduce_colors(best, n, max_time=remaining, seed=999)
        if _validate(result):
            count = len(set(result))
            if count < best_count:
                best = result
                best_count = count
    
    return best
