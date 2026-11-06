import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    N = 128
    
    # Precompute all 4-term progressions (1-based positions)
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))  # 0-based
    
    # For each position, list of AP indices containing it
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)
    
    def count_distinct(colors, ap):
        """Count distinct colors in an AP."""
        return len(set(colors[i] for i in ap))
    
    def is_valid(colors):
        """Check if every 4-term AP has >= 3 distinct colors."""
        for ap in aps:
            if count_distinct(colors, ap) < 3:
                return False
        return True
    
    def count_violations(colors):
        """Count number of APs with < 3 distinct colors."""
        count = 0
        for ap in aps:
            if count_distinct(colors, ap) < 3:
                count += 1
        return count
    
    def count_colors_used(colors):
        return len(set(colors))
    
    def local_search(colors, max_time=None):
        """Improve coloring by local search. Returns improved coloring."""
        if max_time is None:
            max_time = deadline
        colors = colors[:]
        best_colors = colors[:]
        best_violations = count_violations(colors)
        
        if best_violations == 0:
            return best_colors
        
        # Try to fix violations by changing individual positions
        random.seed(42)
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            if random.random() < 0.1:
                # Random perturbation
                for _ in range(random.randint(1, 5)):
                    pos = random.randint(0, N - 1)
                    colors[pos] = random.randint(0, 3)
            else:
                # Find a position in a violated AP and try changing it
                violated_aps = [ap for ap in aps if count_distinct(colors, ap) < 3]
                if not violated_aps:
                    break
                ap = random.choice(violated_aps)
                pos = random.choice(ap)
                old_val = colors[pos]
                colors[pos] = random.randint(0, 3)
            
            new_violations = count_violations(colors)
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                no_improve_count = 0
                if best_violations == 0:
                    break
            else:
                # Accept with some probability (simulated annealing style)
                if random.random() < 0.3:
                    pass  # keep the change
                else:
                    colors = best_colors[:]
                no_improve_count += 1
        
        return best_colors
    
    # Strategy: Use 3 colors (0, 1, 2) and try to find a valid coloring
    # Start with all 0s, then fix violations
    
    best_result = None
    best_num_colors = N  # worst case
    
    # Try to build a valid 3-coloring
    # Initialize with a pattern that's likely to work well
    # Use modular coloring: color[i] = i % 3
    # This might not work, so we need local search
    
    # Multiple restarts with different initial patterns
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            # Pattern: i % 3
            colors = [i % 3 for i in range(N)]
        elif restart == 1:
            # Pattern: (i // 2) % 3
            colors = [(i // 2) % 3 for i in range(N)]
        elif restart == 2:
            # Pattern: (i % 4) // 2 mapped to 3 colors
            colors = [(i % 4) % 3 for i in range(N)]
        elif restart == 3:
            # Random 3-coloring
            random.seed(restart)
            colors = [random.randint(0, 2) for _ in range(N)]
        else:
            # Take best_colors and perturb
            random.seed(restart * 7 + 13)
            colors = best_colors[:] if best_result is not None else [random.randint(0, 2) for _ in range(N)]
            for _ in range(restart % 20 + 1):
                pos = random.randint(0, N - 1)
                colors[pos] = random.randint(0, 2)
        
        # Local search
        improved = local_search(colors, max_time=min(deadline - 5, time.time() + 0.5))
        
        if count_violations(improved) == 0:
            num_colors = count_colors_used(improved)
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_result = improved
                if best_num_colors == 3:
                    # Can't do better than 3
                    break
        elif best_result is None:
            # Keep track of best non-valid solution for perturbation
            best_result = improved
    
    if best_result is not None and count_violations(best_result) == 0:
        # Remap colors to be consecutive
        unique_colors = sorted(set(best_result))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        return [color_map[c] for c in best_result]
    
    # Fallback: use the seed's greedy approach
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                # Check 4-term AP ending at x: x-3d, x-2d, x-d, x
                if x - 3 * d >= 1:
                    prev_colors = {colors[x - 1 - j * d] for j in (1, 2, 3) if x - j * d >= 1}
                    if len({color} | prev_colors) < 3:
                        ok = False
                        break
            if ok:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)  # fallback
    return colors
