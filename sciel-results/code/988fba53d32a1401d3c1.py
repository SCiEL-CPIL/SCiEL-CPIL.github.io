import time
import random
import numpy as np

def count_violations(colors):
    n = len(colors)
    violations = 0
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            c1 = colors[a-1]
            c2 = colors[a+d-1]
            c3 = colors[a+2*d-1]
            c4 = colors[a+3*d-1]
            if len({c1, c2, c3, c4}) < 3:
                violations += 1
    return violations

def try_greedy(n, k, seed=0):
    random.seed(seed)
    colors = []
    for x in range(1, n + 1):
        valid_color = None
        # Try colors in random order to help escape local traps
        color_order = list(range(k))
        random.shuffle(color_order)
        for c in color_order:
            is_valid = True
            # Check all APs ending at x
            for d in range(1, (x-1)//3 + 1):
                if x - 3*d < 1: break
                c1 = colors[x-1-3*d]
                c2 = colors[x-1-2*d]
                c3 = colors[x-1-d]
                if len({c, c1, c2, c3}) < 3:
                    is_valid = False
                    break
            if is_valid:
                valid_color = c
                break
        if valid_color is None:
            return None # Failed to find a valid color
        colors.append(valid_color)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    n = 128
    
    # Try increasing k
    # We want to minimize the number of distinct colors used.
    # Greedy with small k is fast.
    
    for k in [3, 4, 5, 6, 8, 12, 16, 32]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random seeds for greedy
        for seed in range(50):
            if time.time() > deadline - 5:
                break
                
            colors = try_greedy(n, k, seed)
            if colors is not None:
                # Verify
                if count_violations(colors) == 0:
                    # Remap colors to 0..m-1 to minimize label count
                    unique_colors = sorted(set(colors))
                    mapping = {c: i for i, c in enumerate(unique_colors)}
                    remapped = [mapping[c] for c in colors]
                    return remapped
                    
    # Fallback: Guaranteed valid with many colors
    # Use 128 distinct colors
    colors = list(range(n))
    return colors
