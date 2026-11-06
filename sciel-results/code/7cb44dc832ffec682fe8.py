"""Rewritten program: randomized local search with incremental violation tracking."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            a3 = a + 3 * d
            if a3 <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))
    
    num_aps = len(aps)
    
    # Build index: for each position, which APs contain it
    pos_to_aps = [[] for _ in range(n)]
    for i, (p1, p2, p3, p4) in enumerate(aps):
        pos_to_aps[p1].append(i)
        pos_to_aps[p2].append(i)
        pos_to_aps[p3].append(i)
        pos_to_aps[p4].append(i)
    
    def count_violations(colors):
        """Count number of APs with < 3 distinct colors."""
        violations = 0
        for p1, p2, p3, p4 in aps:
            c1, c2, c3, c4 = colors[p1], colors[p2], colors[p3], colors[p4]
            if c1 == c2 == c3 == c4:
                violations += 1
            else:
                # Check if exactly 2 distinct colors
                distinct = len({c1, c2, c3, c4})
                if distinct < 3:
                    violations += 1
        return violations
    
    def count_violations_fast(colors):
        """Faster violation count using numpy."""
        arr = np.array(colors)
        viol = 0
        for p1, p2, p3, p4 in aps:
            c1, c2, c3, c4 = arr[p1], arr[p2], arr[p3], arr[p4]
            if c1 == c2:
                if c3 == c4:
                    viol += 1
                elif c3 == c1 or c4 == c1:
                    viol += 1
                elif c3 == c4:
                    viol += 1
            elif c1 == c3:
                if c2 == c4:
                    viol += 1
                elif c2 == c1 or c4 == c1:
                    viol += 1
            elif c1 == c4:
                if c2 == c3:
                    viol += 1
                elif c2 == c1 or c3 == c1:
                    viol += 1
            elif c2 == c3:
                if c1 == c4:
                    viol += 1
                elif c1 == c2 or c4 == c2:
                    viol += 1
            elif c2 == c4:
                if c1 == c3:
                    viol += 1
                elif c1 == c2 or c3 == c2:
                    viol += 1
            elif c3 == c4:
                if c1 == c2:
                    viol += 1
                elif c1 == c3 or c2 == c3:
                    viol += 1
        return viol
    
    def check_valid(colors):
        """Check if coloring is valid."""
        for p1, p2, p3, p4 in aps:
            c = {colors[p1], colors[p2], colors[p3], colors[p4]}
            if len(c) < 3:
                return False
        return True
    
    def local_search(colors, k, max_iters=5000):
        """Local search to minimize violations, then try to reduce colors."""
        colors = list(colors)
        cur_viol = count_violations_fast(colors)
        
        if cur_viol == 0:
            return colors, cur_viol
        
        best_colors = list(colors)
        best_viol = cur_viol
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            if cur_viol == 0:
                best_colors = list(colors)
                best_viol = 0
                break
            
            # Pick a random position to change
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try each new color
            best_new_color = old_color
            best_new_viol = cur_viol
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                # Compute delta violation
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    p1, p2, p3, p4 = aps[ap_idx]
                    old_c = [colors[p1], colors[p2], colors[p3], colors[p4]]
                    
                    # Old distinct count
                    old_distinct = len(set(old_c))
                    old_bad = 1 if old_distinct < 3 else 0
                    
                    # New distinct count
                    new_c = list(old_c)
                    # Find which position in this AP is pos
                    if p1 == pos:
                        new_c[0] = new_color
                    elif p2 == pos:
                        new_c[1] = new_color
                    elif p3 == pos:
                        new_c[2] = new_color
                    else:
                        new_c[3] = new_color
                    
                    new_distinct = len(set(new_c))
                    new_bad = 1 if new_distinct < 3 else 0
                    
                    delta += new_bad - old_bad
                
                if delta < best_new_viol - cur_viol:
                    best_new_viol = cur_viol + delta
                    best_new_color = new_color
                elif delta == best_new_viol - cur_viol and new_color != old_color:
                    # Tie-breaking: prefer colors that reduce total color count
                    if random.random() < 0.3:
                        best_new_color = new_color
            
            if best_new_color != old_color:
                colors[pos] = best_new_color
                cur_viol = best_new_viol
                
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_colors = list(colors)
            else:
                # No improvement; try random perturbation occasionally
                if it % 100 == 99:
                    pos2 = random.randint(0, n - 1)
                    colors[pos2] = random.randint(0, k - 1)
                    cur_viol = count_violations_fast(colors)
                    if cur_viol < best_viol:
                        best_viol = cur_viol
                        best_colors = list(colors)
        
        return best_colors, best_viol
    
    def try_reduce_colors(colors, k):
        """Try to merge colors to reduce total distinct colors used."""
        colors = list(colors)
        distinct = set(colors)
        k_actual = len(distinct)
        
        if k_actual <= k:
            return colors
        
        # Try to merge pairs of colors
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            
            if len(set(colors)) <= k:
                break
            
            # Pick two colors to merge
            distinct = sorted(set(colors))
            if len(distinct) <= k:
                break
            
            # Pick two random distinct colors
            i, j = random.sample(range(len(distinct)), 2)
            c1, c2 = distinct[i], distinct[j]
            
            # Try merging c2 into c1
            old_colors = list(colors)
            for idx in range(n):
                if colors[idx] == c2:
                    colors[idx] = c1
            
            if check_valid(colors):
                if len(set(colors)) < len(set(old_colors)):
                    continue  # Keep this merge
            else:
                colors = old_colors  # Revert
        
        return colors
    
    # Strategy: Start with k colors, do local search, then try to reduce
    best_result = None
    best_num_colors = 999
    
    # Try different numbers of starting colors
    for k in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 5:
            break
        
        # Multiple restarts for each k
        num_restarts = 20 if k <= 4 else 10
        for restart in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search
            colors, viol = local_search(colors, k, max_iters=3000)
            
            if viol == 0:
                # Valid! Try to reduce colors
                colors_reduced = try_reduce_colors(colors, k)
                actual_colors = len(set(colors_reduced))
                
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_result = list(colors_reduced)
                
                # If we found 3 colors, that's likely optimal
                if best_num_colors == 3:
                    return best_result
            else:
                # Even if not fully valid, check if we can get close
                # and see if reducing helps
                if viol <= 2:
                    colors_reduced = try_reduce_colors(colors, k)
                    if check_valid(colors_reduced):
                        actual_colors = len(set(colors_reduced))
                        if actual_colors < best_num_colors:
                            best_num_colors = actual_colors
                            best_result = list(colors_reduced)
    
    # If we have a valid result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, n + 1):
        placed = False
        for color in range(128):
            valid = True
            # Check all 4-term APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a >= 1:
                    idx1, idx2, idx3, idx4 = a - 1, a + d - 1, a + 2 * d - 1, x - 1
                    if idx1 < len(colors) and idx2 < len(colors) and idx3 < len(colors):
                        c1, c2, c3 = colors[idx1], colors[idx2], colors[idx3]
                        if len({color, c1, c2, c3}) < 3:
                            valid = False
                            break
            if valid:
                colors.append(color)
                placed = True
                break
        if not placed:
            colors.append(0)
    
    return colors

# EVOLVE-BLOCK-END
