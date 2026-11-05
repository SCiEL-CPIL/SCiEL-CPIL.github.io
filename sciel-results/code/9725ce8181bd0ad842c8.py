import time
import random

def construct():
    N = 128
    # Precompute APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Precompute which APs involve each index
    # ap_indices_for_pos[i] = list of indices in aps where i appears
    ap_indices_for_pos = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_for_pos[pos].append(idx)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check distinct count
            # Fast check:
            # If all same -> 1
            # If 2 same, 2 different -> 2
            # If 3 same, 1 different -> 2
            # If all diff -> 4
            # If 2 pairs -> 2
            # If 1 pair, 2 singles -> 3
            # If 3 singles, 1 pair? No, 4 items.
            
            # Simplest: len({c0, c1, c2, c3}) < 3
            if len({c0, c1, c2, c3}) < 3:
                count += 1
        return count

    def get_violations_for_pos(colors, pos, new_val):
        # Returns number of violations in APs containing pos, assuming colors[pos] = new_val
        # We must be careful not to modify colors permanently yet
        # We can just check the APs
        v_count = 0
        for ap_idx in ap_indices_for_pos[pos]:
            ap = aps[ap_idx]
            c0 = colors[ap[0]] if ap[0] != pos else new_val
            c1 = colors[ap[1]] if ap[1] != pos else new_val
            c2 = colors[ap[2]] if ap[2] != pos else new_val
            c3 = colors[ap[3]] if ap[3] != pos else new_val
            if len({c0, c1, c2, c3}) < 3:
                v_count += 1
        return v_count

    # Search for minimum k
    best_solution = None
    global_deadline = time.time() + 45.0 # Leave 15s for safety
    
    for k in range(3, 128):
        if time.time() > global_deadline:
            break
            
        # Try to find a solution with k colors
        # Local search with restarts
        # We give a time budget for each k
        # If k is small, it's harder.
        # Let's say 5s for k=3, 2s for k=4, etc.
        # Or just run until global deadline if previous failed?
        # Better: run a few restarts.
        
        start_k_time = time.time()
        found = False
        
        # Max 20 restarts or until 5s passed for this k
        for restart in range(20):
            if time.time() > global_deadline:
                break
            if time.time() - start_k_time > 5.0:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Local search
            for step in range(10000):
                if time.time() > global_deadline:
                    break
                
                # Find a position to change
                # Random is simple.
                # Better: pick a position involved in a violation.
                # We can just pick random pos.
                pos = random.randint(0, N - 1)
                
                current_val = colors[pos]
                
                # Try all k values
                best_new_val = current_val
                best_violations = get_violations_for_pos(colors, pos, current_val)
                
                for new_val in range(k):
                    if new_val == current_val:
                        continue
                    v = get_violations_for_pos(colors, pos, new_val)
                    if v < best_violations:
                        best_violations = v
                        best_new_val = new_val
                
                # If no improvement, pick random (simulated annealing style)
                if best_violations >= get_violations_for_pos(colors, pos, current_val):
                    # Maybe just pick random from k
                    best_new_val = random.randint(0, k - 1)
                    # But we need to check if it's better?
                    # Actually, the loop above finds the min.
                    # If min is >= current, we are stuck.
                    # Let's just pick a random different value to move.
                    # Or just keep current?
                    # Standard WalkSAT: pick random clause, pick random var, pick random val.
                    # Here, let's pick a random value.
                    best_new_val = random.randint(0, k - 1)
                
                colors[pos] = best_new_val
                
                # Check if solved
                # We can check total violations every few steps to save time
                if step % 10 == 0:
                    if count_violations(colors) == 0:
                        best_solution = colors
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
        
    if best_solution is None:
        # Fallback: return a valid coloring with many colors (e.g. all distinct)
        # But we need to ensure it's valid.
        # If we use N colors, every AP has 4 distinct colors, so it's valid.
        best_solution = list(range(N))
        
    return best_solution
