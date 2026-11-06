import time
import random
import math

def construct():
    deadline = time.time() + 85
    N = 128
    
    # Precompute all nonconstant 4-term APs (0-indexed positions 0..127)
    aps = []
    for a in range(N):
        for d in range(1, N):
            if a + 3*d < N:
                aps.append((a, a+d, a+2*d, a+3*d))
    
    # For each position, list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count number of 4-APs with < 3 distinct colors."""
        count = 0
        for i, j, k, l in aps:
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                count += 1
        return count
    
    def is_valid_partial(colors, changed_idx):
        """Check if changing position changed_idx creates any violation."""
        for ap_idx in pos_to_aps[changed_idx]:
            i, j, k, l = aps[ap_idx]
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                return False
        return True
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Try to construct a 3-coloring
    # Strategy: start with a base pattern, then repair violations
    
    best_colors = None
    best_cost = 129
    
    rng = random.Random(12345)
    
    # Phase 1: Try multiple random 3-color initializations with repair
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random 3-coloring
        colors = [rng.randint(0, 2) for _ in range(N)]
        
        # Greedy repair: fix violations by recoloring
        improved = True
        max_iter = 500
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Find a violation
            viol_ap = None
            for ap_idx, (i, j, k, l) in enumerate(aps):
                if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                    viol_ap = (i, j, k, l)
                    break
            if viol_ap is None:
                break
            i, j, k, l = viol_ap
            # Try to fix by recoloring one of the positions
            fixed = False
            for pos in [i, j, k, l]:
                old_c = colors[pos]
                for new_c in [0, 1, 2]:
                    if new_c == old_c:
                        continue
                    colors[pos] = new_c
                    if is_valid_partial(colors, pos):
                        fixed = True
                        break
                if fixed:
                    break
                colors[pos] = old_c
            if not fixed:
                # Try all colors 0..127
                for pos in [i, j, k, l]:
                    old_c = colors[pos]
                    for new_c in range(128):
                        if new_c == old_c:
                            continue
                        colors[pos] = new_c
                        if is_valid_partial(colors, pos):
                            fixed = True
                            break
                    if fixed:
                        break
                    colors[pos] = old_c
                if not fixed:
                    break
        
        cost = count_distinct(colors)
        if count_violations(colors) == 0:
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
        
        # Also try to reduce colors
        if best_colors is not None and best_cost <= 3:
            # Try to see if we can do better (unlikely but check)
            pass
    
    # Phase 2: If we have a valid coloring, try to minimize colors via local search
    if best_colors is not None:
        colors = best_colors[:]
        current_cost = best_cost
        
        # Simulated annealing to reduce color count
        T = 2.0
        T_min = 0.01
        cooling = 0.9998
        
        freq = {}
        for c in colors:
            freq[c] = freq.get(c, 0) + 1
        
        while time.time() < deadline - 3 and T > T_min:
            idx = rng.randint(0, N-1)
            old_color = colors[idx]
            new_color = rng.randint(0, 127)
            if new_color == old_color:
                T *= cooling
                continue
            
            colors[idx] = new_color
            
            if is_valid_partial(colors, idx):
                freq[old_color] -= 1
                if freq[old_color] == 0:
                    del freq[old_color]
                freq[new_color] = freq.get(new_color, 0) + 1
                
                new_cost = len(freq)
                delta = new_cost - current_cost
                
                if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_colors = colors[:]
                else:
                    colors[idx] = old_color
                    freq[old_color] = freq.get(old_color, 0) + 1
                    freq[new_color] -= 1
                    if freq[new_color] == 0:
                        del freq[new_color]
            else:
                colors[idx] = old_color
            
            T *= cooling
        
        # Phase 3: Polish - try to eliminate specific colors
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            colors = best_colors[:]
            current_cost = best_cost
            freq = {}
            for c in colors:
                freq[c] = freq.get(c, 0) + 1
            
            used = sorted(freq.keys())
            for target_color in used[:]:
                if target_color not in freq:
                    continue
                positions = [i for i in range(N) if colors[i] == target_color]
                rng.shuffle(positions)
                success = True
                for pos in positions:
                    found = False
                    for new_c in used:
                        if new_c == target_color:
                            continue
                        colors[pos] = new_c
                        if is_valid_partial(colors, pos):
                            found = True
                            break
                    if not found:
                        for new_c in range(128):
                            if new_c == target_color:
                                continue
                            colors[pos] = new_c
                            if is_valid_partial(colors, pos):
                                found = True
                                break
                    if not found:
                        success = False
                        break
                    freq[target_color] -= 1
                    if freq[target_color] == 0:
                        del freq[target_color]
                    freq[new_c] = freq.get(new_c, 0) + 1
                if success:
                    new_cost = len(freq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_colors = colors[:]
    
    if best_colors is None:
        # Fallback: should not happen
        return list(range(N))
    
    return best_colors
