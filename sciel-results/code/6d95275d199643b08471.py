import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline for safety (evaluator kills at 90s)
    
    n = 128
    
    # Precompute all 4-term arithmetic progressions
    def get_4tps():
        tps = []
        pos_to_tps = [[] for _ in range(n)]
        for a in range(1, n + 1):
            for d in range(1, n):
                a3 = a + 3 * d
                if a3 <= n:
                    tp = (a - 1, a + d - 1, a + 2 * d - 1, a3 - 1)
                    tp_idx = len(tps)
                    tps.append(tp)
                    for pos in tp:
                        pos_to_tps[pos].append(tp_idx)
        return tps, pos_to_tps
    
    all_tps, pos_to_tps = get_4tps()
    num_tps = len(all_tps)
    
    def count_distinct4(coloring, tp):
        return len({coloring[tp[0]], coloring[tp[1]], coloring[tp[2]], coloring[tp[3]]})
    
    def is_valid(coloring):
        for tp in all_tps:
            if count_distinct4(coloring, tp) < 3:
                return False
        return True
    
    def count_violations(coloring):
        count = 0
        for tp in all_tps:
            if count_distinct4(coloring, tp) < 3:
                count += 1
        return count
    
    def compute_delta_fast(coloring, pos, old_color, new_color):
        """Compute change in violations when changing position pos from old_color to new_color."""
        delta = 0
        for tp_idx in pos_to_tps[pos]:
            tp = all_tps[tp_idx]
            others = [p for p in tp if p != pos]
            other_colors = [coloring[p] for p in others]
            
            old_distinct = len({old_color} | set(other_colors))
            new_distinct = len({new_color} | set(other_colors))
            
            old_bad = 1 if old_distinct < 3 else 0
            new_bad = 1 if new_distinct < 3 else 0
            delta += new_bad - old_bad
        return delta
    
    def try_color_merge(coloring, color_a, color_b):
        """Try to merge all instances of color_b into color_a. Returns True if valid."""
        # Create a test coloring
        test_coloring = coloring[:]
        for i in range(n):
            if test_coloring[i] == color_b:
                test_coloring[i] = color_a
        
        # Check validity
        return is_valid(test_coloring)
    
    def minimize_colors(coloring):
        """Try to minimize the number of colors used while maintaining validity."""
        used_colors = sorted(set(coloring))
        current_num_colors = len(used_colors)
        
        # Iteratively try to merge colors
        improved = True
        while improved:
            improved = False
            colors_list = sorted(set(coloring))
            
            for i in range(len(colors_list)):
                for j in range(i + 1, len(colors_list)):
                    if time.time() > deadline:
                        return coloring
                    
                    ci = colors_list[i]
                    cj = colors_list[j]
                    
                    # Try merging cj into ci
                    if try_color_merge(coloring, ci, cj):
                        # Apply the merge
                        for k in range(n):
                            if coloring[k] == cj:
                                coloring[k] = ci
                        current_num_colors -= 1
                        improved = True
                        break
                if improved:
                    break
        
        return coloring
    
    # Try with increasing number of colors
    for max_colors in range(3, 129):
        if time.time() > deadline:
            break
        
        best_coloring = None
        best_violations = float('inf')
        
        # Use more restarts for smaller color counts
        num_restarts = 200 if max_colors <= 10 else 100
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Various initial colorings
            if restart == 0:
                coloring = [i % max_colors for i in range(n)]
            elif restart == 1:
                coloring = [(i * 7) % max_colors for i in range(n)]
            elif restart == 2:
                coloring = [(i * 13) % max_colors for i in range(n)]
            elif restart == 3:
                coloring = [(i * 31) % max_colors for i in range(n)]
            else:
                coloring = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Local search with simulated annealing
            current_violations = count_violations(coloring)
            temperature = 5.0
            cooling = 0.995
            
            for iteration in range(10000):
                if time.time() > deadline:
                    break
                
                if current_violations == 0:
                    best_coloring = coloring[:]
                    best_violations = 0
                    break
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
                
                # Pick a position: bias toward positions in violating TPs
                violating_positions = set()
                for tp_idx in range(num_tps):
                    tp = all_tps[tp_idx]
                    if count_distinct4(coloring, tp) < 3:
                        for pos in tp:
                            violating_positions.add(pos)
                
                if violating_positions and random.random() < 0.7:
                    pos = random.choice(list(violating_positions))
                else:
                    pos = random.randint(0, n - 1)
                
                old_color = coloring[pos]
                new_color = random.randint(0, max_colors - 1)
                if new_color == old_color:
                    continue
                
                delta = compute_delta_fast(coloring, pos, old_color, new_color)
                
                # Simulated annealing acceptance
                if delta <= 0:
                    accept = True
                else:
                    accept = random.random() < pow(2.718281828, -delta / temperature)
                
                if accept:
                    coloring[pos] = new_color
                    current_violations += delta
                
                temperature *= cooling
            
            if best_violations == 0:
                break
        
        if best_violations == 0 and best_coloring is not None:
            # We found a valid coloring with at most max_colors colors
            # Now try to minimize the actual number of colors used
            coloring = best_coloring[:]
            coloring = minimize_colors(coloring)
            
            # Verify
            if is_valid(coloring):
                return coloring
            else:
                return best_coloring
    
    # Fallback: return a simple valid coloring if we didn't find a better one
    # Use the greedy approach as fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
