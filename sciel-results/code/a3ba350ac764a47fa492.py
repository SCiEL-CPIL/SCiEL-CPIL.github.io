import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    positions = list(range(1, n + 1))
    
    # Precompute all nonconstant 4-APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            if a + 3 * d <= n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
            else:
                break
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = {i: [] for i in positions}
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count number of violated APs (those with < 3 distinct colors)."""
        count = 0
        for ap in aps:
            vals = set(colors[p] for p in ap)
            if len(vals) < 3:
                count += 1
        return count
    
    def try_merge_colors(colors, max_iters=50000, seed=0):
        """Try to merge colors to reduce the count while maintaining validity."""
        rng = random.Random(seed)
        colors = list(colors)
        current_colors = set(colors)
        
        if len(current_colors) <= 2:
            return colors
        
        # Try merging pairs of colors
        color_list = sorted(current_colors)
        
        for _ in range(100):  # Try up to 100 merge attempts
            if time.time() > deadline:
                break
            if len(set(colors)) <= 2:
                break
            
            # Pick a random pair of colors to try merging
            c1, c2 = rng.sample(sorted(set(colors)), 2)
            
            # Try merging c2 into c1
            new_colors = [c1 if c == c2 else c for c in colors]
            
            if count_violations(new_colors) == 0:
                colors = new_colors
                continue  # Successfully merged, try again
            
            # If direct merge doesn't work, try local search to fix violations
            # by recoloring some positions
            fixed = _fix_after_merge(colors, c1, c2, rng)
            if fixed is not None:
                colors = fixed
        
        return colors
    
    def _fix_after_merge(original_colors, c1, c2, rng, max_iters=20000):
        """After attempting to merge c2->c1, try to fix violations by recoloring."""
        colors = [c1 if c == c2 else c for c in original_colors]
        
        # Find violated APs
        violated_aps = []
        for idx, ap in enumerate(aps):
            vals = set(colors[p] for p in ap)
            if len(vals) < 3:
                violated_aps.append(idx)
        
        if not violated_aps:
            return colors
        
        # Local search to fix violations
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not violated_aps:
                return colors
            
            # Pick a random violated AP
            ap_idx = rng.choice(violated_aps)
            ap = aps[ap_idx]
            
            # Try recoloring each position in the AP
            pos_order = list(ap)
            rng.shuffle(pos_order)
            
            improved = False
            for pos in pos_order:
                old_color = colors[pos]
                # Try all possible colors (0 to 127)
                for c in range(128):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    
                    # Check if this fixes the violation without creating new ones
                    new_violated = []
                    for v_idx in violated_aps:
                        v_ap = aps[v_idx]
                        vals = set(colors[p] for p in v_ap)
                        if len(vals) < 3:
                            new_violated.append(v_idx)
                    
                    # Also check APs that weren't violated before but might be now
                    for ap_i in pos_to_aps[pos]:
                        if ap_i not in violated_aps and ap_i not in new_violated:
                            ap = aps[ap_i]
                            vals = set(colors[p] for p in ap)
                            if len(vals) < 3:
                                new_violated.append(ap_i)
                    
                    if len(new_violated) < len(violated_aps):
                        violated_aps = new_violated
                        improved = True
                        break
                    else:
                        colors[pos] = old_color
                        break
                
                if improved:
                    break
            
            if not improved:
                # Random kick
                pos = rng.choice(positions)
                colors[pos] = rng.randint(0, 127)
                # Recompute violations
                violated_aps = []
                for idx, ap in enumerate(aps):
                    vals = set(colors[p] for p in ap)
                    if len(vals) < 3:
                        violated_aps.append(idx)
        
        if not violated_aps:
            return colors
        return None
    
    def try_local_search(k, max_iters=50000, seed=0):
        """Local search to find valid coloring with exactly k colors."""
        rng = random.Random(seed)
        
        # Initialize with a pattern that tends to work
        colors = {}
        for p in positions:
            # Use a modular pattern
            colors[p] = (p - 1) % k
        
        # Check initial violations
        violated = [False] * num_aps
        for idx in range(num_aps):
            ap = aps[idx]
            vals = set(colors[p] for p in ap)
            violated[idx] = len(vals) < 3
        num_violations = sum(violated)
        
        if num_violations == 0:
            return [colors[p] for p in positions]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if num_violations == 0:
                return [colors[p] for p in positions]
            
            # Pick a random violated AP
            v_indices = [i for i in range(num_aps) if violated[i]]
            if not v_indices:
                break
            ap_idx = v_indices[rng.randint(0, len(v_indices) - 1)]
            ap = aps[ap_idx]
            
            # Try to fix by recoloring one position in this AP
            pos_order = list(ap)
            rng.shuffle(pos_order)
            
            improved = False
            for pos in pos_order:
                old_color = colors[pos]
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    
                    # Update violations for APs containing pos
                    delta = 0
                    for ap_i in pos_to_aps[pos]:
                        was_v = violated[ap_i]
                        vals = set(colors[q] for q in aps[ap_i])
                        is_v = len(vals) < 3
                        violated[ap_i] = is_v
                        if is_v and not was_v:
                            delta += 1
                        elif not is_v and was_v:
                            delta -= 1
                    
                    if num_violations + delta < num_violations:
                        num_violations += delta
                        improved = True
                        break
                    else:
                        # Revert
                        colors[pos] = old_color
                        for ap_i in pos_to_aps[pos]:
                            vals = set(colors[q] for q in aps[ap_i])
                            violated[ap_i] = len(vals) < 3
                        num_violations = sum(violated)
                        break
                
                if improved:
                    break
            
            if not improved:
                # Random kick: recolor a random position
                pos = rng.choice(positions)
                new_c = rng.randint(0, k - 1)
                colors[pos] = new_c
                for ap_i in pos_to_aps[pos]:
                    vals = set(colors[q] for q in aps[ap_i])
                    violated[ap_i] = len(vals) < 3
                num_violations = sum(violated)
        
        if num_violations == 0:
            return [colors[p] for p in positions]
        return None
    
    # Strategy: Start with more colors, then merge down
    # First try to get a valid coloring with 5 colors
    best_result = None
    best_color_count = 999
    
    # Try different initial patterns and local searches
    for k in [5, 4, 3]:
        for seed in range(20):
            if time.time() > deadline:
                break
            
            # Try local search
            result = try_local_search(k, max_iters=20000, seed=seed)
            if result is not None:
                color_count = len(set(result))
                if color_count < best_color_count:
                    best_color_count = color_count
                    best_result = result
                    if color_count == 3:
                        return result
    
    # If we have a valid result, try to merge colors to reduce count
    if best_result is not None:
        for seed in range(10):
            if time.time() > deadline:
                break
            merged = try_merge_colors(best_result, max_iters=20000, seed=seed)
            if merged is not None:
                color_count = len(set(merged))
                if color_count < best_color_count:
                    best_color_count = color_count
                    best_result = merged
                    if color_count == 3:
                        return merged
    
    # Fallback: greedy construction
    if best_result is None:
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                valid = True
                # Check all 4-APs ending at x
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a < 1:
                        continue
                    ap = (a, a + d, a + 2 * d, x)
                    vals = set([color] + [colors[ap[i] - 1] for i in range(3)])
                    if len(vals) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                colors.append(0)
        best_result = colors
    
    return best_result
