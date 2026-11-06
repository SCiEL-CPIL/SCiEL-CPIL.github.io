"""Rewritten program: precomputed color-count vectors + simulated annealing + color reduction."""
import time
import random
import math

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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    tps = _all_4tps(n)
    num_tps = len(tps)
    
    # Pre-compute for each position, which 4-TPs it participates in
    pos_tps = [[] for _ in range(n)]
    for i, tp in enumerate(tps):
        for pos in tp:
            pos_tps[pos].append(i)
    
    # Pre-compute for each TP, the positions (for fast color-count updates)
    # tp_positions[i] = (a, b, c, e)
    tp_positions = [(tp[0], tp[1], tp[2], tp[3]) for tp in tps]
    
    def count_violations(colors):
        """Count the number of 4-TPs with fewer than 3 distinct colors."""
        viol_count = 0
        for i in range(num_tps):
            a, b, c, e = tp_positions[i]
            c0, c1, c2, c3 = colors[a], colors[b], colors[c], colors[e]
            if len({c0, c1, c2, c3}) < 3:
                viol_count += 1
        return viol_count

    def sim_anneal(k_colors, time_limit, initial_colors=None, seed_offset=0):
        """
        Simulated annealing to find a valid coloring with k_colors.
        Returns (colors, is_valid, viol_count).
        """
        sa_start = time.time()
        
        if initial_colors is not None:
            colors = list(initial_colors)
        else:
            colors = [random.randint(0, k_colors - 1) for _ in range(n)]
        
        # Initialize violation state
        is_violated = [False] * num_tps
        viol_count = 0
        for i in range(num_tps):
            a, b, c, e = tp_positions[i]
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                is_violated[i] = True
                viol_count += 1
        
        best_colors = list(colors)
        best_viol = viol_count
        if viol_count == 0:
            return best_colors, True, 0
        
        # Temperature schedule: exponential cooling
        t0 = max(0.5, min(2.0, best_viol * 0.1))
        t_min = 0.01
        # Estimate number of steps we can do
        # We'll use a time-based approach with adaptive step rate
        
        # Pre-compute: for each TP, which positions are in it (already have tp_positions)
        # For efficiency, track color counts per TP
        # tp_color_counts[i] = dict color -> count
        tp_color_counts = []
        for i in range(num_tps):
            a, b, c, e = tp_positions[i]
            cc = {}
            for pos in (a, b, c, e):
                col = colors[pos]
                cc[col] = cc.get(col, 0) + 1
            tp_color_counts.append(cc)
        
        # Verify violation counts match
        # A TP is violated if it has < 3 distinct colors, i.e., max count >= 2 and distinct < 3
        # Actually: violated if len(distinct) < 3, which means 1 or 2 distinct colors
        # With 4 positions: 1 distinct means all same; 2 distinct means split like 3-1 or 2-2
        
        # Simulated annealing loop
        # We'll do fixed number of iterations per time slice
        temp = t0
        cooling_rate = 0.9995  # per step
        step_count = 0
        
        while time.time() < sa_start + time_limit:
            step_count += 1
            
            # Pick a random position
            pos = random.randrange(n)
            old_color = colors[pos]
            
            # Pick a new color
            new_color = random.randrange(k_colors)
            if new_color == old_color:
                continue
            
            # Update affected TPs
            affected = pos_tps[pos]
            old_states = []
            old_ccs = []
            delta_viol = 0
            
            for i in affected:
                old_viol = is_violated[i]
                old_states.append(old_viol)
                old_ccs.append(dict(tp_color_counts[i]))
                
                # Update color count
                cc = tp_color_counts[i]
                cc[old_color] -= 1
                if cc[old_color] == 0:
                    del cc[old_color]
                cc[new_color] = cc.get(new_color, 0) + 1
                
                new_distinct = len(cc)
                new_viol = new_distinct < 3
                is_violated[i] = new_viol
                
                if old_viol and not new_viol:
                    delta_viol -= 1
                elif not old_viol and new_viol:
                    delta_viol += 1
            
            new_viol = viol_count + delta_viol
            
            if new_viol < viol_count:
                # Accept
                viol_count = new_viol
                colors[pos] = new_color
                if viol_count < best_viol:
                    best_viol = viol_count
                    best_colors = list(colors)
                    if viol_count == 0:
                        return best_colors, True, 0
            else:
                # Maybe accept with probability exp(-delta/temp)
                delta = new_viol - viol_count
                if delta > 0 and temp > t_min:
                    accept_prob = math.exp(-delta / temp)
                    if random.random() < accept_prob:
                        viol_count = new_viol
                        colors[pos] = new_color
                else:
                    # Revert
                    for idx, i in enumerate(affected):
                        is_violated[i] = old_states[idx]
                        tp_color_counts[i] = old_ccs[idx]
                    # colors[pos] unchanged
            
            # Cool down
            temp *= cooling_rate
            if temp < t_min:
                temp = t_min
        
        return best_colors, (best_viol == 0), best_viol

    def reduce_colors(colors, k_current, time_limit):
        """
        Try to reduce the number of colors by merging color classes.
        Returns (reduced_colors, new_k) or (colors, k_current) if no reduction.
        """
        red_start = time.time()
        current_colors = list(colors)
        current_k = k_current
        
        # Try to merge pairs of colors
        while time.time() < red_start + time_limit:
            # Get list of colors actually used
            used_colors = set(current_colors)
            if len(used_colors) < current_k:
                current_k = len(used_colors)
            
            if current_k <= 2:
                break  # Can't go below 3 for validity, but let's try
            
            # Try merging each pair
            merged = False
            color_list = sorted(used_colors)
            for i in range(len(color_list)):
                for j in range(i + 1, len(color_list)):
                    if time.time() > red_start + time_limit:
                        break
                    ci = color_list[i]
                    cj = color_list[j]
                    
                    # Try merging cj into ci
                    test_colors = [ci if c == cj else c for c in current_colors]
                    
                    # Check validity
                    valid = True
                    for tp in tps:
                        c0, c1, c2, c3 = test_colors[tp[0]], test_colors[tp[1]], test_colors[tp[2]], test_colors[tp[3]]
                        if len({c0, c1, c2, c3}) < 3:
                            valid = False
                            break
                    
                    if valid:
                        current_colors = test_colors
                        current_k = len(set(current_colors))
                        merged = True
                        break
                if merged or time.time() > red_start + time_limit:
                    break
            
            if not merged:
                break
        
        return current_colors, current_k

    # Strategy:
    # 1. Try simulated annealing with 3 colors
    # 2. If that fails, try 4 colors
    # 3. After finding a valid coloring, try to reduce colors
    
    best_result = None
    best_result_k = 128
    
    # Try 3 colors with multiple restarts
    if time.time() < deadline - 40.0:
        for restart in range(5):
            if time.time() > deadline - 40.0:
                break
            colors, valid, viol = sim_anneal(3, 8.0, seed_offset=restart)
            if valid:
                best_result = colors
                best_result_k = 3
                break
            elif best_result is None or viol < count_violations(best_result):
                best_result = colors
                best_result_k = 3
    
    # Try 4 colors
    if best_result_k > 3 and time.time() < deadline - 20.0:
        for restart in range(3):
            if time.time() > deadline - 20.0:
                break
            colors, valid, viol = sim_anneal(4, 8.0, seed_offset=restart + 100)
            if valid:
                best_result = colors
                best_result_k = 4
                break
            elif best_result is None or viol < count_violations(best_result):
                best_result = colors
                best_result_k = 4
    
    # Try 5 colors as fallback
    if best_result_k > 4 and time.time() < deadline - 10.0:
        for restart in range(2):
            if time.time() > deadline - 10.0:
                break
            colors, valid, viol = sim_anneal(5, 5.0, seed_offset=restart + 200)
            if valid:
                best_result = colors
                best_result_k = 5
                break
            elif best_result is None or viol < count_violations(best_result):
                best_result = colors
                best_result_k = 5
    
    # If we have a valid result, try to reduce colors
    if best_result is not None and best_result_k > 3:
        if time.time() < deadline - 5.0:
            reduced, new_k = reduce_colors(best_result, best_result_k, 5.0)
            if new_k < best_result_k:
                best_result = reduced
                best_result_k = new_k
    
    # If no valid result found, return a trivially valid high-color solution
    if best_result is None:
        # Use a pattern that's likely valid: color[i] = i % k for large k
        # Actually, just use all distinct colors
        return list(range(128))
    
    # Ensure colors are in range 0..127
    # Remap colors to 0..k-1 for cleanliness (optional but nice)
    color_map = {}
    next_label = 0
    for c in best_result:
        if c not in color_map:
            color_map[c] = next_label
            next_label += 1
    remapped = [color_map[c] for c in best_result]
    
    return remapped

# EVOLVE-BLOCK-END
