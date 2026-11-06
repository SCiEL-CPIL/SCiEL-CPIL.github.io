import time
import random
import itertools

_DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _k4_violations(colors, n=16):
    """Count K4s with < 5 distinct colors. Returns list of (k4_tuple, num_distinct)."""
    violations = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        cols = set()
        cols.add(colors[a*n + b])
        cols.add(colors[a*n + c])
        cols.add(colors[a*n + d])
        cols.add(colors[b*n + c])
        cols.add(colors[b*n + d])
        cols.add(colors[c*n + d])
        nd = len(cols)
        if nd < 5:
            violations.append((quad, nd))
    return violations

def _count_distinct_colors(colors, n=16):
    return len(set(colors[i*n+j] for i in range(n) for j in range(i+1, n)))

def _get_cost(colors, n=16):
    """Cost = number of violating K4s (each weighted by (5 - distinct))."""
    total = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        cols = set()
        cols.add(colors[a*n + b])
        cols.add(colors[a*n + c])
        cols.add(colors[a*n + d])
        cols.add(colors[b*n + c])
        cols.add(colors[b*n + d])
        cols.add(colors[c*n + d])
        nd = len(cols)
        if nd < 5:
            total += (5 - nd)
    return total

def construct():
    n = 16
    random.seed(42)
    
    # Initialize with (i+j) mod 17
    colors = [0] * (n * n)
    for i in range(n):
        for j in range(i+1, n):
            colors[i*n + j] = (i + j) % 17
            colors[j*n + i] = (i + j) % 17
    
    best_colors = colors[:]
    best_cost = _get_cost(best_colors, n)
    best_num_colors = _count_distinct_colors(best_colors, n)
    
    if _remaining() < 5.0:
        return [best_colors[i*n+j] for i in range(n) for j in range(i+1, n)]
    
    # Local search: try recoloring individual edges
    max_iter = 500000
    iter_count = 0
    no_improve = 0
    
    while _remaining() > 3.0 and iter_count < max_iter:
        iter_count += 1
        if iter_count % 10000 == 0 and no_improve > 50000:
            # Restart with perturbation
            break
        
        # Pick a random edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        old_color = colors[i*n + j]
        
        # Try changing to a new color
        # Candidate colors: existing colors + one new
        existing = set(colors[i*n+j] for i2 in range(n) for j2 in range(i2+1, n))
        max_existing = max(existing) if existing else 0
        candidates = list(existing)
        if random.random() < 0.1:
            candidates.append(max_existing + 1)
        
        best_new_color = old_color
        best_new_cost = best_cost
        
        for new_color in candidates:
            if new_color == old_color:
                continue
            colors[i*n + j] = new_color
            colors[j*n + i] = new_color
            new_cost = _get_cost(colors, n)
            if new_cost < best_new_cost:
                best_new_cost = new_cost
                best_new_color = new_color
            colors[i*n + j] = old_color
            colors[j*n + i] = old_color
        
        if best_new_color != old_color:
            colors[i*n + j] = best_new_color
            colors[j*n + i] = best_new_color
            best_cost = best_new_cost
            no_improve = 0
            if best_cost < _get_cost(best_colors, n):
                best_colors = colors[:]
                best_num_colors = _count_distinct_colors(best_colors, n)
        else:
            no_improve += 1
    
    # Try to minimize number of distinct colors used (remap colors)
    if _remaining() > 2.0:
        # Remap colors to minimize distinct count while preserving validity
        # First, find which colors are actually used
        used_colors = set()
        for i in range(n):
            for j in range(i+1, n):
                used_colors.add(best_colors[i*n+j])
        
        # Try to merge colors: for each pair of used colors, check if merging is valid
        color_list = sorted(used_colors)
        changed = True
        while changed and _remaining() > 2.0:
            changed = False
            for a_idx in range(len(color_list)):
                for b_idx in range(a_idx+1, len(color_list)):
                    ca, cb = color_list[a_idx], color_list[b_idx]
                    if ca not in used_colors or cb not in used_colors:
                        continue
                    # Try merging cb into ca
                    test_colors = best_colors[:]
                    for i in range(n):
                        for j in range(i+1, n):
                            if test_colors[i*n+j] == cb:
                                test_colors[i*n+j] = ca
                                test_colors[j*n+i] = ca
                    if _get_cost(test_colors, n) == 0:
                        best_colors = test_colors
                        used_colors.discard(cb)
                        color_list.remove(cb)
                        changed = True
                        break
                if changed:
                    break
    
    # Final: remap to 0..k-1
    used = sorted(set(best_colors[i*n+j] for i in range(n) for j in range(i+1, n)))
    mapping = {c: idx for idx, c in enumerate(used)}
    final_colors = [mapping[best_colors[i*n+j]] for i in range(n) for j in range(i+1, n)]
    
    return final_colors
