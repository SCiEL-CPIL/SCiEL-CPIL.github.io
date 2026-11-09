import random
import time
from itertools import combinations

def _edge_idx(i, j):
    """Map pair (i,j) with i<j to linear index in the 120-edge list."""
    # Number of edges before row i: sum_{k=0}^{i-1} (16-1-k) = sum_{k=0}^{i-1}(15-k)
    # = i*15 - i*(i-1)//2
    return i * 15 - i * (i - 1) // 2 + (j - i - 1)

def _count_bad_k4s(colors, K):
    """Count K4s with fewer than 5 distinct colors. colors is a list of 120 ints."""
    bad = 0
    n = 16
    # Precompute edge indices for quick lookup
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    e1 = _edge_idx(a, b)
                    e2 = _edge_idx(a, c)
                    e3 = _edge_idx(a, d)
                    e4 = _edge_idx(b, c)
                    e5 = _edge_idx(b, d)
                    e6 = _edge_idx(c, d)
                    cols = {colors[e1], colors[e2], colors[e3], colors[e4], colors[e5], colors[e6]}
                    if len(cols) < 5:
                        bad += 1
    return bad

def _count_colors_used(colors):
    """Count distinct colors used."""
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Seed coloring
    colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Precompute all K4s and their edge indices
    all_k4s = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    all_k4s.append((_edge_idx(a,b), _edge_idx(a,c), _edge_idx(a,d),
                                    _edge_idx(b,c), _edge_idx(b,d), _edge_idx(c,d)))
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, k4 in enumerate(all_k4s):
        for e in k4:
            edge_to_k4s[e].append(k4_idx)
    
    # Try to reduce colors: first get to 15, then 14, then 13
    best_colors = colors[:]
    best_bad = _count_bad_k4s(colors, 17)
    
    # Target: minimize number of colors, but must have 0 bad K4s
    # Use simulated annealing with objective: bad_k4s * 1000 + num_colors_used
    
    def objective(c):
        bad = _count_bad_k4s(c, 120)
        if bad > 0:
            return bad * 10000 + _count_colors_used(c)
        return _count_colors_used(c)
    
    current = colors[:]
    current_obj = objective(current)
    best = current[:]
    best_obj = current_obj
    
    rng = random.Random(42)
    temp = 100.0
    cooling = 0.9995
    
    # Also try direct approaches: reduce color range
    # First, try to find a valid 15-coloring by remapping
    for target_K in [15, 14, 13]:
        if time.time() > deadline:
            break
        # Try to remap colors to use only target_K colors
        # Start from current best
        trial = best[:]
        # Map all colors to 0..target_K-1
        color_map = {}
        next_color = 0
        for c in trial:
            if c not in color_map:
                if next_color < target_K:
                    color_map[c] = next_color
                    next_color += 1
                else:
                    color_map[c] = rng.randint(0, target_K - 1)
        trial = [color_map[c] for c in trial]
        trial_obj = objective(trial)
        
        if trial_obj < best_obj:
            best = trial[:]
            best_obj = trial_obj
            current = trial[:]
            current_obj = trial_obj
        
        # Local search to fix bad K4s
        for step in range(20000):
            if time.time() > deadline:
                break
            if current_obj <= target_K:  # Found valid coloring with target_K or fewer colors
                break
            
            e = rng.randint(0, 119)
            old_color = current[e]
            new_color = rng.randint(0, target_K - 1)
            if new_color == old_color:
                continue
            
            current[e] = new_color
            new_obj = objective(current)
            
            delta = new_obj - current_obj
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                current_obj = new_obj
                if new_obj < best_obj:
                    best = current[:]
                    best_obj = new_obj
            else:
                current[e] = old_color
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        # If we found a valid coloring with target_K colors, we're done
        if best_obj <= target_K:
            break
    
    # Final check: if best is valid (0 bad K4s), return it
    if _count_bad_k4s(best, 120) == 0:
        return best
    
    # If no valid coloring found, try harder with more aggressive search
    # Reset and try again with different parameters
    if time.time() < deadline:
        current = colors[:]
        current_obj = objective(current)
        best = current[:]
        best_obj = current_obj
        
        temp = 50.0
        cooling = 0.999
        
        for target_K in [15, 14, 13]:
            if time.time() > deadline:
                break
            if best_obj <= target_K:
                break
            
            for step in range(30000):
                if time.time() > deadline:
                    break
                if current_obj <= target_K:
                    break
                
                e = rng.randint(0, 119)
                old_color = current[e]
                new_color = rng.randint(0, target_K - 1)
                if new_color == old_color:
                    continue
                
                current[e] = new_color
                new_obj = objective(current)
                
                delta = new_obj - current_obj
                if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best = current[:]
                        best_obj = new_obj
                else:
                    current[e] = old_color
                
                temp *= cooling
                if temp < 0.01:
                    temp = 0.01
        
        if _count_bad_k4s(best, 120) == 0:
            return best
    
    # Fallback: return the seed if nothing better found
    if _count_bad_k4s(best, 120) == 0:
        return best
    return colors
