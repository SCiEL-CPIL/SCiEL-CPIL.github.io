import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute all nonconstant 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, (i, j, k, l) in enumerate(aps):
        pos_aps[i].append(idx)
        pos_aps[j].append(idx)
        pos_aps[k].append(idx)
        pos_aps[l].append(idx)

    def count_violations(colors):
        count = 0
        for i, j, k, l in aps:
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                count += 1
        return count

    best_colors = None
    best_distinct = n

    # Try different numbers of colors
    for max_k in range(3, 15):
        time_limit_for_k = 10.0
        if time.time() > deadline - time_limit_for_k:
            break
            
        # Run local search for this K
        for trial in range(20):
            if time.time() > deadline - 5:
                break
                
            # Random initial coloring
            colors = [random.randint(0, max_k - 1) for _ in range(n)]
            cur_violations = count_violations(colors)
            
            if cur_violations == 0:
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors[:]
                    if best_distinct <= 3:
                        # Remap and return
                        color_map = {}
                        result = []
                        for c in best_colors:
                            if c not in color_map:
                                color_map[c] = len(color_map)
                            result.append(color_map[c])
                        return result
                break

            # Min-Conflicts Local Search
            no_improve = 0
            max_steps = 5000
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                if cur_violations == 0:
                    distinct = len(set(colors))
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = colors[:]
                        if best_distinct <= 3:
                             color_map = {}
                             result = []
                             for c in best_colors:
                                 if c not in color_map:
                                     color_map[c] = len(color_map)
                                 result.append(color_map[c])
                             return result
                    break

                # Find a random violated AP
                # To do this efficiently, we can just pick a random AP and check if it's violated.
                # Or maintain a list of violated APs.
                # Given N=128, num APs is small. We can just pick a random AP index.
                ap_idx = random.randint(0, len(aps) - 1)
                i, j, k, l = aps[ap_idx]
                
                # Check if this AP is actually violated
                if len({colors[i], colors[j], colors[k], colors[l]}) >= 3:
                    continue

                # Pick a random position in this AP
                pos = random.choice([i, j, k, l])
                old_color = colors[pos]
                
                # Find best new color for this position
                best_new_color = old_color
                best_local_violations = cur_violations
                min_violations = cur_violations
                
                for new_color in range(max_k):
                    if new_color == old_color:
                        continue
                    colors[pos] = new_color
                    # We only need to check APs involving pos
                    # But for simplicity and speed with small N, full recount is okay?
                    # No, full recount is 1000 APs * 4 ops = 4000 ops.
                    # 5000 steps * 4000 ops = 20M ops.
                    # Python is slow. 20M ops might take 2-5 seconds.
                    # So we can do ~10-20 restarts in 10s.
                    # This is acceptable.
                    
                    new_violations = count_violations(colors)
                    colors[pos] = old_color
                    
                    if new_violations < min_violations:
                        min_violations = new_violations
                        best_new_color = new_color
                
                if best_new_color != old_color:
                    colors[pos] = best_new_color
                    cur_violations = min_violations
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 200:
                        # Restart
                        break

    if best_colors is None:
        # Fallback
        best_colors = list(range(n))
        best_distinct = n

    # Remap to 0..distinct-1
    color_map = {}
    result = []
    for c in best_colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        result.append(color_map[c])
    
    return result
