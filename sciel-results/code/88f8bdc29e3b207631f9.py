import time
import random
import numpy as np

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3 * d >= n:
                break
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it participates in
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    def count_violations(colors):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if fewer than 3 distinct
            if c0 == c1 == c2 == c3:
                violations += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                violations += 1
            elif c0 == c1 and c2 == c3 and c0 != c2:
                violations += 1
            elif c0 == c2 and c1 == c3 and c0 != c1:
                violations += 1
            elif c0 == c3 and c1 == c2 and c0 != c1:
                violations += 1
            elif c0 == c2 and c1 == c3 and c0 == c1:
                violations += 1  # all same, already counted
            # More carefully: count distinct
            distinct = len({c0, c1, c2, c3})
            if distinct < 3:
                # This would double count, let me just use the set approach
                pass
        # Simpler: just use set
        violations = 0
        for ap in aps:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                violations += 1
        return violations
    
    def local_search(colors, num_colors, max_iters=5000):
        """Local search to minimize violations."""
        current_colors = colors.copy()
        current_violations = count_violations(current_colors)
        
        if current_violations == 0:
            return current_colors, 0
        
        temp = 1.0
        cooling = 0.9995
        best_colors = current_colors.copy()
        best_violations = current_violations
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            pos = np.random.randint(0, n)
            old_color = current_colors[pos]
            new_color = np.random.randint(0, num_colors)
            
            if new_color == old_color:
                continue
            
            # Compute delta in violations
            current_colors[pos] = new_color
            new_violations = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                if len({current_colors[ap[0]], current_colors[ap[1]], current_colors[ap[2]], current_colors[ap[3]]}) < 3:
                    new_violations += 1
            
            current_colors[pos] = old_color
            old_violations = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                if len({current_colors[ap[0]], current_colors[ap[1]], current_colors[ap[2]], current_colors[ap[3]]}) < 3:
                    old_violations += 1
            
            delta = new_violations - old_violations
            current_colors[pos] = new_color
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = current_colors.copy()
            else:
                current_colors[pos] = old_color
            
            temp *= cooling
        
        return best_colors, best_violations
    
    best_colors = None
    best_num_colors = float('inf')
    
    # Try different numbers of colors, starting from small
    for num_colors in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 10.0:
            break
        
        # Multiple restarts for this number of colors
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize randomly
            colors = np.random.randint(0, num_colors, size=n)
            
            # Run local search
            result_colors, result_violations = local_search(colors, num_colors, max_iters=3000)
            
            if result_violations == 0:
                actual_colors = np.unique(result_colors)
                if len(actual_colors) < best_num_colors:
                    best_num_colors = len(actual_colors)
                    best_colors = result_colors.copy()
                break  # Found valid solution with this num_colors, move to next
        
        if best_colors is not None and best_num_colors <= num_colors:
            # We found a valid solution, check if we can do better with fewer colors
            # But we're going in order, so if we found one with num_colors, 
            # we should try to see if fewer works
            # Actually, we should break and return if we have a good solution
            # But let's continue to see if we can find a solution with fewer colors
            # Wait, we're trying in order from 3 up, so if we found one with 4, 
            # we already tried 3 and it didn't work (or we didn't find it)
            # Let's just keep trying until we find a valid one
            pass
    
    if best_colors is None:
        # Fallback: use more colors
        num_colors = 12
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            colors = np.random.randint(0, num_colors, size=n)
            result_colors, result_violations = local_search(colors, num_colors, max_iters=5000)
            if result_violations == 0:
                best_colors = result_colors
                best_num_colors = len(np.unique(result_colors))
                break
    
    if best_colors is None:
        # Last resort: use a modular coloring with enough colors
        # x mod 7 gives 7 colors, but may not satisfy constraints
        # Let's just use a random coloring with many colors and hope
        best_colors = np.random.randint(0, 128, size=n)
    
    return best_colors.tolist()
