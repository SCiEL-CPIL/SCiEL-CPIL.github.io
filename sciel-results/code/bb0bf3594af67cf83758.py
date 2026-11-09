import random
import time

def construct():
    N = 128
    # Precompute all 4-APs
    # Each AP is a tuple of 4 indices (0-based)
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for start in range(N - 3 * d):
            aps.append((start, start + d, start + 2 * d, start + 3 * d))
    
    num_aps = len(aps)
    
    # For each index, store the list of AP indices it belongs to
    # This allows fast update of violation counts
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)
            
    def is_violation(ap, colors):
        # Returns True if the AP has < 3 distinct colors
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Count distinct colors
        # Fast check: if all same -> 1
        if c0 == c1 == c2 == c3:
            return True
        # If 2 distinct, it's a violation. 
        # We can check if the set size is < 3
        # len({c0, c1, c2, c3}) < 3
        return len({c0, c1, c2, c3}) < 3

    def get_violation_counts(colors):
        counts = [0] * num_aps
        for idx, ap in enumerate(aps):
            if is_violation(ap, colors):
                counts[idx] = 1
        return counts

    def total_violations(counts):
        return sum(counts)

    # We want to minimize the number of colors used.
    # We will try K=3, then K=4, etc.
    # But the problem says "labels in 0..127".
    # If we use K colors, we can just use labels 0..K-1.
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    for K in range(3, 10):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for each K
        best_violations = float('inf')
        best_colors = None
        
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random colors
            colors = [random.randint(0, K - 1) for _ in range(N)]
            counts = get_violation_counts(colors)
            current_violations = total_violations(counts)
            
            if current_violations == 0:
                return colors
            
            # Local Search
            temp = 1.0
            cooling = 0.999
            best_in_run = current_violations
            best_colors_run = colors.copy()
            
            while time.time() - start_time < time_limit:
                # Pick random index
                i = random.randint(0, N - 1)
                old_c = colors[i]
                
                # Try all possible new colors
                best_new_c = old_c
                best_delta = 0
                
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    
                    # Calculate change in violations
                    delta = 0
                    for ap_idx in index_to_aps[i]:
                        ap = aps[ap_idx]
                        # Check old status
                        old_viol = counts[ap_idx]
                        
                        # Temporarily change color to check new status
                        # We need to check if the AP becomes violated or stops being violated
                        # Since only color at i changes, we can just evaluate the AP with new_c
                        # But we need the other 3 colors.
                        # Let's just compute the distinct count for the new configuration of this AP.
                        
                        # Get other colors
                        other_colors = []
                        for pos in ap:
                            if pos != i:
                                other_colors.append(colors[pos])
                        
                        # New set of colors
                        new_set = set(other_colors)
                        new_set.add(new_c)
                        
                        new_viol = 1 if len(new_set) < 3 else 0
                        
                        delta += new_viol - old_viol
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                    elif delta == best_delta and random.random() < 0.1:
                        # Tie-breaking: random choice among equal deltas
                        best_new_c = new_c
                
                # Acceptance
                if best_delta < 0:
                    # Always accept improvement
                    colors[i] = best_new_c
                    # Update counts
                    for ap_idx in index_to_aps[i]:
                        ap = aps[ap_idx]
                        counts[ap_idx] = 1 if is_violation(ap, colors) else 0
                    current_violations += best_delta
                    if current_violations < best_in_run:
                        best_in_run = current_violations
                        best_colors_run = colors.copy()
                        if current_violations == 0:
                            return colors
                else:
                    # Accept with probability exp(-delta / temp)
                    if random.random() < pow(2.71828, -best_delta / temp):
                        colors[i] = best_new_c
                        for ap_idx in index_to_aps[i]:
                            ap = aps[ap_idx]
                            counts[ap_idx] = 1 if is_violation(ap, colors) else 0
                        current_violations += best_delta
                        if current_violations < best_in_run:
                            best_in_run = current_violations
                            best_colors_run = colors.copy()
                            if current_violations == 0:
                                return colors
                
                temp *= cooling
                if temp < 0.01:
                    break
            
            if best_in_run < best_violations:
                best_violations = best_in_run
                best_colors = best_colors_run
                
        if best_violations == 0 and best_colors:
            return best_colors

    # If no solution found with K < 10, return the best one found or a fallback
    # The seed program always returns a valid solution, so we should too.
    # If we failed to find a valid solution with few colors, we can use the greedy approach from seed as fallback?
    # Or just return the best_colors from the last K tried.
    # But the task requires EVERY progression to have >= 3 colors.
    # If we return an invalid solution, score is 0.
    # So we MUST return a valid solution.
    
    # Fallback: Use the seed's greedy algorithm if local search fails?
    # The seed is valid. Let's just run the seed logic if we fail.
    # But the seed is slow? No, it's fast.
    
    # Let's implement a quick greedy fallback just in case.
    # Actually, the local search with K=3 or 4 should find a solution very quickly for N=128.
    # Probability of failure is low.
    
    # To be safe, if we didn't return, use the seed's method.
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
