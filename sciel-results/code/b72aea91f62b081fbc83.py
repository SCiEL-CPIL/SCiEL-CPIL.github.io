"""Rewritten program: greedy randomized construction with local repair for 4-TP coloring."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _all_4tps(n=128):
    """Generate all non-constant 4-term arithmetic progressions in 0..n-1."""
    tps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                tps.append((a, b, c, e))
    return tps

def _violations_for_pos(pos, colors, tps, pos_tps):
    """Count how many 4-TPs containing pos have < 3 distinct colors."""
    count = 0
    for tp_idx in pos_tps[pos]:
        a, b, c, e = tps[tp_idx]
        vals = {colors[a], colors[b], colors[c], colors[e]}
        if len(vals) < 3:
            count += 1
    return count

def _total_violations(colors, tps):
    """Count total number of violated 4-TPs."""
    v = 0
    for a, b, c, e in tps:
        vals = {colors[a], colors[b], colors[c], colors[e]}
        if len(vals) < 3:
            v += 1
    return v

def _is_valid(colors, tps):
    """Check if every 4-TP has at least 3 distinct colors."""
    for a, b, c, e in tps:
        if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 128
    tps = _all_4tps(n)
    num_tps = len(tps)
    
    # Pre-compute for each position, which 4-TPs it participates in
    pos_tps = [[] for _ in range(n)]
    for i, tp in enumerate(tps):
        for pos in tp:
            pos_tps[pos].append(i)
    
    best_colors = None
    best_score = float('inf')
    
    # Strategy: greedy forward assignment with random restarts
    # For each position, try colors in random order, pick the one that
    # minimizes immediate constraint violations with already-assigned positions
    
    # Key insight: when assigning position p, the constraints that matter
    # are 4-TPs where p is the last element (a, b, c, p) - these are fully
    # determined by previous assignments. We can check these cheaply.
    
    # For each position p, precompute which 4-TPs have p as the last element
    last_pos_tps = [[] for _ in range(n)]
    for i, (a, b, c, e) in enumerate(tps):
        last_pos_tps[e].append(i)
    
    # Also track 4-TPs where p is in earlier positions (for forward-looking)
    # But for greedy, we mainly care about completing TPs
    
    def greedy_construct(k, seed):
        """Greedy forward construction with k colors."""
        random.seed(seed)
        colors = [-1] * n
        colors[0] = 0
        
        for p in range(1, n):
            # Get all 4-TPs where p is the last element
            relevant_tps = last_pos_tps[p]
            
            # For each candidate color, count how many relevant TPs would be violated
            best_color = None
            best_viol = float('inf')
            
            # Shuffle colors for randomization
            color_order = list(range(k))
            random.shuffle(color_order)
            
            for c in color_order:
                colors[p] = c
                viol = 0
                for tp_idx in relevant_tps:
                    a, b, cc, e = tps[tp_idx]
                    vals = {colors[a], colors[b], colors[cc], colors[e]}
                    if len(vals) < 3:
                        viol += 1
                        if viol >= best_viol:
                            break
                if viol < best_viol:
                    best_viol = viol
                    best_color = c
                    if viol == 0:
                        break
            
            colors[p] = best_color
        return colors
    
    def local_search(colors, k, max_iters, seed):
        """Local search to fix violations."""
        random.seed(seed)
        colors = list(colors)
        
        for _ in range(max_iters):
            # Find a violated TP
            violated_idx = -1
            for i, (a, b, c, e) in enumerate(tps):
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    violated_idx = i
                    break
            
            if violated_idx == -1:
                return colors  # No violations
            
            a, b, c, e = tps[violated_idx]
            positions = [a, b, c, e]
            
            # Try changing each position to a different color
            random.shuffle(positions)
            for pos in positions:
                for new_c in range(k):
                    if new_c == colors[pos]:
                        continue
                    
                    old_c = colors[pos]
                    colors[pos] = new_c
                    
                    # Check if this reduces violations
                    # Count violations in TPs containing pos
                    new_viol = 0
                    for tp_idx in pos_tps[pos]:
                        aa, bb, cc, ee = tps[tp_idx]
                        if len({colors[aa], colors[bb], colors[cc], colors[ee]}) < 3:
                            new_viol += 1
                        if new_viol > 3:
                            break
                    
                    if new_viol == 0:
                        break  # Keep the change
                    else:
                        colors[pos] = old_c
                
                # If we broke out of inner loop with a good change, break outer
                # (already kept the change)
            
            # Occasionally do a random move to escape local minima
            if random.random() < 0.1:
                pos = random.randint(0, n - 1)
                colors[pos] = random.randint(0, k - 1)
        
        return colors
    
    # Try different numbers of colors
    for k in range(2, 8):
        if time.time() > deadline - 5.0:
            break
        
        max_restarts = 30 if k <= 3 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3.0:
                break
            
            seed = restart * 7919 + k * 31
            
            # Greedy construction
            colors = greedy_construct(k, seed)
            
            # Local search to fix any remaining violations
            colors = local_search(colors, k, max_iters=2000, seed=seed + 1000)
            
            # Check validity
            if _is_valid(colors, tps):
                used = _count_colors(colors)
                if used < best_score:
                    best_score = used
                    best_colors = list(colors)
                    # If we found a valid coloring with k colors, try to see
                    # if we can do better with fewer (but k is already minimal in loop)
                    break
        
        # If we found valid with k colors, no need to try more
        if best_colors is not None and best_score <= k:
            break
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        if _is_valid(best_colors, tps):
            return best_colors
    
    # Fallback: sequential greedy (deterministic)
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            ok = True
            # Check all 4-TPs ending at position x-1 (0-indexed)
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                b = x - 1 - 2 * d
                c = x - 1 - d
                if a < 0:
                    continue
                # Positions a, b, c are already assigned (0-indexed: a, b, c)
                # x-1 is the new position
                existing = [colors[a], colors[b], colors[c]]
                if len(set(existing + [color])) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                found = True
                break
        
        if not found:
            # Should not happen, but fallback
            colors.append(127)
    
    return colors

# EVOLVE-BLOCK-END
