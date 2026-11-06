import time
import random
import numpy as np

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

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

def _is_valid_assignment(colors, pos, color, pos_to_aps):
    """Check if assigning `color` to `pos` keeps all APs containing pos valid."""
    for ap in pos_to_aps[pos]:
        a, b, c, e = ap
        cols = [colors[a], colors[b], colors[c], colors[e]]
        for idx, p in enumerate((a, b, c, e)):
            if p == pos:
                cols[idx] = color
                break
        if len(set(cols)) < 3:
            return False
    return True

def _backtrack_3color(n=128, deadline=None, seed=0):
    """Backtracking search for a valid 3-coloring of [0..n-1].
    
    Structural approach: When assigning position i, only 4-APs where i is the
    LAST element (i = a+3d) are fully constrained by prior assignments.
    Uses DFS with forward checking and multiple randomized restarts.
    """
    rng = random.Random(seed)
    colors = [-1] * n
    
    # Precompute: for each position i, the list of (a, b, c) where (a,b,c,i) is a 4-AP
    last_aps = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            if a >= 0:
                last_aps[i].append((a, b, c))
    
    nodes_visited = [0]
    
    def backtrack(i):
        if deadline is not None and time.time() > deadline:
            return False
        if i == n:
            return True
        
        nodes_visited[0] += 1
        if nodes_visited[0] % 50000 == 0 and deadline is not None:
            if time.time() > deadline:
                return False
        
        # Determine valid colors for position i
        valid = []
        for color in range(3):
            ok = True
            for (a, b, c) in last_aps[i]:
                ca, cb, cc = colors[a], colors[b], colors[c]
                # Need |{ca, cb, cc, color}| >= 3
                distinct_prior = len({ca, cb, cc})
                if distinct_prior == 3:
                    pass  # Any color works
                elif distinct_prior == 2:
                    # color must differ from both existing distinct colors
                    if color == ca or color == cb or color == cc:
                        ok = False
                        break
                else:  # distinct_prior == 1
                    # All three prior same -> adding any color gives at most 2 distinct
                    ok = False
                    break
            if ok:
                valid.append(color)
        
        if not valid:
            return False
        
        # Color ordering: prefer already-used colors
        used = set(colors[:i])
        preferred = [c for c in valid if c in used]
        new_colors = [c for c in valid if c not in used]
        rng.shuffle(new_colors)
        order = preferred + new_colors
        
        for color in order:
            colors[i] = color
            if backtrack(i + 1):
                return True
            colors[i] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _construct_with_backtrack(n=128, deadline=None):
    """Try to construct a valid 3-coloring using backtracking with multiple seeds."""
    for seed in range(20):
        if deadline is not None and time.time() > deadline - 3:
            break
        result = _backtrack_3color(n, deadline=deadline, seed=seed)
        if result is not None:
            return result
    return None

def _greedy_3color_fast(n=128, seed=0):
    """Fast randomized greedy trying to use only 3 colors, with incremental constraint check."""
    rng = random.Random(seed)
    colors = [-1] * n
    for i in range(n):
        valid_colors = [0, 1, 2]
        for d in range(1, (i) // 3 + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            if a < 0:
                break
            c1, c2, c3 = colors[a], colors[b], colors[c]
            for col in list(valid_colors):
                if len({c1, c2, c3, col}) < 3:
                    valid_colors.remove(col)
        if not valid_colors:
            return None
        used = set(colors[:i])
        preferred = [c for c in valid_colors if c in used]
        if preferred:
            colors[i] = rng.choice(preferred)
        else:
            colors[i] = rng.choice(valid_colors)
    return colors

def _local_search_reduce_colors(colors, n=128, max_iters=5000, seed=0):
    """Try to reduce color count by merging colors."""
    rng = random.Random(seed)
    colors = list(colors)
    current_count = len(set(colors))
    
    if current_count <= 3:
        return colors
    
    color_list = sorted(set(colors))
    
    for _ in range(max_iters):
        if len(color_list) <= 3:
            break
        c1, c2 = rng.sample(color_list, 2)
        new_colors = list(colors)
        for i in range(n):
            if new_colors[i] == c2:
                new_colors[i] = c1
        if _validate(new_colors):
            colors = new_colors
            color_list = sorted(set(colors))
            current_count = len(color_list)
            if current_count <= 3:
                break
    return colors

def _construct_3color_direct(n=128, seed=0, time_limit=3.0):
    """Direct construction of 3-coloring using pattern-based approach."""
    start_time = time.time()
    rng = random.Random(seed)
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        if attempt == 0:
            colors = []
            for x in range(1, n + 1):
                v2 = 0
                tmp = x
                while tmp % 2 == 0:
                    tmp //= 2
                    v2 += 1
                colors.append(v2 % 3)
        elif attempt == 1:
            colors = []
            for x in range(1, n + 1):
                v3 = 0
                tmp = x
                while tmp % 3 == 0:
                    tmp //= 3
                    v3 += 1
                colors.append(v3 % 3)
        elif attempt == 2:
            colors = []
            for x in range(1, n + 1):
                s = bin(x - 1).count('1')
                colors.append(s % 3)
        elif attempt == 3:
            colors = []
            for x in range(1, n + 1):
                base = (x - 1) % 3
                if rng.random() < 0.1:
                    base = (base + 1) % 3
                colors.append(base)
        elif attempt < 10:
            colors = [-1] * n
            valid = True
            for i in range(n):
                valid_colors = {0, 1, 2}
                for d in range(1, (i) // 3 + 1):
                    a = i - 3 * d
                    b = i - 2 * d
                    c = i - d
                    if a >= 0:
                        c1, c2, c3 = colors[a], colors[b], colors[c]
                        for col in list(valid_colors):
                            if len({c1, c2, c3, col}) < 3:
                                valid_colors.discard(col)
                if not valid_colors:
                    valid = False
                    break
                colors[i] = rng.choice(list(valid_colors))
            if not valid:
                continue
        else:
            colors = [-1] * n
            valid = True
            for i in range(n):
                valid_colors = {0, 1, 2}
                for d in range(1, (i) // 3 + 1):
                    a = i - 3 * d
                    b = i - 2 * d
                    c = i - d
                    if a >= 0:
                        c1, c2, c3 = colors[a], colors[b], colors[c]
                        for col in list(valid_colors):
                            if len({c1, c2, c3, col}) < 3:
                                valid_colors.discard(col)
                if not valid_colors:
                    valid = False
                    break
                used = set(colors[:i])
                preferred = valid_colors & used
                if preferred:
                    colors[i] = rng.choice(list(preferred))
                else:
                    colors[i] = rng.choice(list(valid_colors))
            if not valid:
                continue
        
        if _validate(colors):
            if len(set(colors)) == 3:
                return colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    all_aps, pos_to_aps = _precompute_aps(n)
    
    # First, get the seed fallback (guaranteed valid)
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # PRIMARY: Backtracking search for 3-coloring (structural change)
    remaining = deadline - time.time()
    if remaining > 10:
        result = _construct_with_backtrack(n, deadline=deadline)
        if result is not None:
            best = result
            best_count = 3
            return best
    
    # SECONDARY: Fast 3-color greedy with many seeds
    remaining = deadline - time.time()
    if remaining > 5:
        for seed in range(200):
            if time.time() > deadline - 15:
                break
            result = _greedy_3color_fast(n, seed=seed)
            if result is not None and _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # TERTIARY: Direct 3-coloring construction (patterns)
    remaining = deadline - time.time()
    if remaining > 10:
        result = _construct_3color_direct(n, seed=0, time_limit=min(4.0, remaining - 5))
        if result is not None:
            best = result
            best_count = 3
            return best
    
    # QUATERNARY: Local search merge to reduce colors
    remaining = deadline - time.time()
    if remaining > 2 and best_count > 3:
        for seed in range(50):
            if time.time() > deadline - 1:
                break
            result = _local_search_reduce_colors(best, n, max_iters=1000, seed=seed)
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    return best
