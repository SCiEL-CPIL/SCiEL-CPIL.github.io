"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_valid(colors):
    """Check if a coloring satisfies all 4-AP constraints."""
    n = len(colors)
    for x in range(1, n + 1):
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            c0 = colors[x - 1]
            c1 = colors[x - 1 - d]
            c2 = colors[x - 1 - 2 * d]
            c3 = colors[x - 1 - 3 * d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _greedy_deterministic():
    """Seed's deterministic greedy: assign smallest valid color."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            # Check all 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                prev = {colors[x - 1 - j * d] for j in (1, 2, 3)}
                if len({color} | prev) < 3:
                    ok = False
                    break
            # Also check 4-APs where x is not the last element
            # x, x+d, x+2d, x+3d where x+3d <= 128
            # But those haven't been assigned yet, so skip
            if ok:
                colors.append(color)
                break
    return colors

def _randomized_greedy(K, seed_val):
    """Randomized greedy with K colors, trying to minimize colors used."""
    rng = random.Random(seed_val)
    colors = [None] * 128
    
    for i in range(128):
        pos = i + 1
        valid = set(range(K))
        
        # Check all 4-APs where pos is the last element
        max_d = (pos - 1) // 3
        for d in range(1, max_d + 1):
            i1 = pos - 3 * d - 1
            i2 = pos - 2 * d - 1
            i3 = pos - d - 1
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            
            # Remove colors that would make this AP have < 3 distinct colors
            for c in list(valid):
                if len({c, c1, c2, c3}) < 3:
                    valid.discard(c)
        
        if not valid:
            return None
        
        # Prefer already-used colors to minimize total colors
        used = {colors[j] for j in range(i) if colors[j] is not None}
        preferred = valid & used
        if preferred:
            c = rng.choice(list(preferred))
        else:
            c = rng.choice(list(valid))
        colors[i] = c
    
    # Verify
    if _check_valid(colors):
        return colors
    return None

def _local_search_improve(colors, K, max_iter=2000, seed_val=42):
    """Try to reduce number of distinct colors via local search."""
    rng = random.Random(seed_val)
    current = list(colors)
    current_colors = set(current)
    
    for _ in range(max_iter):
        # Try changing a random position to a color that's already in use
        # but different from current, to potentially reduce color count
        pos = rng.randint(0, 127)
        old_color = current[pos]
        
        # Try all colors in current_colors except old_color
        candidates = [c for c in current_colors if c != old_color]
        if not candidates:
            continue
        
        new_color = rng.choice(candidates)
        current[pos] = new_color
        
        if _check_valid(current):
            new_colors = set(current)
            if len(new_colors) < len(current_colors):
                current_colors = new_colors
            # Keep the change even if color count same (for diversity)
        else:
            current[pos] = old_color
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # First, get the deterministic greedy solution as fallback
    fallback = _greedy_deterministic()
    fallback_colors = len(set(fallback))
    
    best = fallback
    best_color_count = fallback_colors
    
    # Try to find a solution with fewer colors using randomized search
    # Start with K=3, then 4, etc.
    for K in range(3, best_color_count + 1):
        if time.time() > deadline:
            break
        
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            seed_val = attempt * 7919 + K
            result = _randomized_greedy(K, seed_val)
            if result is not None:
                color_count = len(set(result))
                if color_count < best_color_count:
                    best = result
                    best_color_count = color_count
                    # If we found K=3 solution, that's optimal (need at least 3)
                    if best_color_count == 3:
                        return best
                # Try to improve via local search
                if time.time() < deadline - 10:
                    improved = _local_search_improve(result, K, max_iter=500, seed_val=seed_val + 1)
                    improved_count = len(set(improved))
                    if improved_count < best_color_count:
                        best = improved
                        best_color_count = improved_count
                        if best_color_count == 3:
                            return best
    
    return best
# EVOLVE-BLOCK-END
