import time
import random

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            if b == c or c == e or b == e:
                continue  # constant AP
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_3color(n=128, seed=0):
    """Try to construct a valid 3-coloring greedily."""
    rng = random.Random(seed)
    colors = [None] * n
    
    for i in range(n):
        # Try colors 0, 1, 2 in random order
        for color in rng.sample([0, 1, 2], 3):
            colors[i] = color
            # Check all 4-APs ending at position i
            valid = True
            for d in range(1, (i) // 3 + 1):
                a = i - 3*d
                b = i - 2*d
                c = i - d
                if a < 0:
                    break
                cols = {colors[a], colors[b], colors[c], colors[i]}
                if len(cols) < 3:
                    valid = False
                    break
            if valid:
                break
        if colors[i] is None:
            return None
    return colors

def _local_search(colors, n=128, max_iters=50000, seed=0):
    """Improve a valid coloring by trying to reduce color count."""
    rng = random.Random(seed)
    current = list(colors)
    current_colors = set(current)
    
    for _ in range(max_iters):
        # Try to remove a color by recoloring positions with that color
        target_color = rng.choice(list(current_colors))
        positions = [i for i in range(n) if current[i] == target_color]
        
        # Try to recolor all positions with target_color to other colors
        new_current = list(current)
        success = True
        for pos in positions:
            # Try each other color
            other_colors = [c for c in current_colors if c != target_color]
            found = False
            for new_color in rng.sample(other_colors, len(other_colors)):
                new_current[pos] = new_color
                # Check validity for 4-APs involving this position
                valid = True
                for a in range(max(0, pos - 3*20), min(n, pos + 1)):
                    for d in range(1, 20):
                        b = a + d
                        c = a + 2*d
                        e = a + 3*d
                        if e >= n or e < 0:
                            break
                        if b == c or c == e or b == e:
                            continue
                        cols = {new_current[a], new_current[b], new_current[c], new_current[e]}
                        if len(cols) < 3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found = True
                    break
            if not found:
                success = False
                break
        
        if success:
            current = new_current
            current_colors = set(current)
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    
    # First try to find a valid 3-coloring with various seeds
    best = None
    for seed in range(100):
        if time.time() > deadline:
            break
        result = _greedy_3color(n, seed)
        if result is not None:
            if _validate(result):
                best = result
                break
    
    if best is not None:
        # Try to improve with local search
        if time.time() < deadline - 5:
            improved = _local_search(best, n, max_iters=20000, seed=42)
            if _validate(improved) and _count_colors(improved) <= _count_colors(best):
                best = improved
    
    if best is not None:
        return best
    
    # Fallback: use the original greedy approach with more colors
    colors = []
    for x in range(1, n + 1):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    
    return colors
