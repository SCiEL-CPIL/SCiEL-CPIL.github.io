"""
Optimized solver for the 4-term progression coloring problem.
Uses a greedy construction with a smarter local search to minimize colors used.
"""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Precompute all 4-term progressions (a, a+d, a+2d, a+3d) within [1, 128]
    # Note: The problem uses 0-indexed positions in the return value, 
    # but the logic in the seed used 1-indexed range(1,129).
    # Let's stick to 0-indexed for the array, positions 0..127.
    # A 4-term progression is indices i, i+d, i+2d, i+3d where i+3d < 128.
    # "Nonconstant" means d > 0.
    
    progressions = []
    for i in range(128):
        for d in range(1, (128 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < 128:
                progressions.append((i, j, k, l))
    
    n_progressions = len(progressions)
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check if all progressions have at least 3 distinct colors."""
        for i, j, k, l in progressions:
            c = {colors[i], colors[j], colors[k], colors[l]}
            if len(c) < 3:
                return False
        return True
    
    def count_violations(colors):
        """Count how many progressions have < 3 distinct colors."""
        count = 0
        for i, j, k, l in progressions:
            c = {colors[i], colors[j], colors[k], colors[l]}
            if len(c) < 3:
                count += 1
        return count

    # Strategy: 
    # 1. Try to find a valid coloring with a small number of colors.
    # 2. Start with a greedy heuristic that tries to use few colors.
    # 3. Use local search (hill climbing) to fix violations and minimize color count.
    
    best_colors = None
    best_score = 128  # Max possible is 128 distinct colors
    
    # Helper to create a random initial coloring using k colors
    def random_coloring(k):
        return [random.randint(0, k - 1) for _ in range(128)]
    
    # Helper to evaluate a coloring: 
    # Primary objective: minimize number of distinct colors used.
    # Secondary objective (constraint): must be valid (0 violations).
    # We will search for valid colorings with the minimum number of colors.
    
    # The seed program uses a greedy approach that picks the smallest color index 
    # that satisfies constraints *so far*. This naturally tends to use few colors 
    # because it prefers 0, then 1, etc. But it might get stuck in a local minimum 
    # or use more colors than necessary if a later position forces a new color.
    
    # Let's first run the seed's logic to get a baseline.
    # Seed logic:
    def seed_greedy():
        colors = []
        for x in range(1, 129):
            # x is 1-indexed position. Index is x-1.
            found = False
            for color in range(128):
                valid_so_far = True
                # Check all progressions ending at x-1? 
                # The seed checks:
                # 1. For all d, the set {color, colors[x-1-d], colors[x-1-2d], colors[x-1-3d]} has size >= 3.
                #    This ensures the progression (x-1-3d, x-1-2d, x-1-d, x-1) is valid.
                # 2. For all d, at least one of colors[x-1-d] or colors[x-1-2d] is not equal to color.
                #    This ensures the progression (x-1-2d, x-1-d, x-1, x-1+d) ... wait.
                #    The second condition in seed: 
                #    any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1, (x-1)//2+1)
                #    This checks pairs? No, it checks if adding 'color' at x-1 doesn't create a 2-color progression 
                #    with the previous two terms? 
                #    Actually, a 4-term progression involving x-1 can start earlier or be centered.
                #    The seed only checks progressions where x-1 is the LAST term (condition 1) 
                #    and ensures that the first 3 terms don't allow a 2-color extension?
                #    Let's just use the full validity check for simplicity and correctness, 
                #    even if it's slower. 128 positions is small.
                
                # Let's just check all progressions that include the current position x-1.
                idx = x - 1
                for i, j, k, l in progressions:
                    if idx not in (i, j, k, l):
                        continue
                    # Create a temporary copy to test
                    # This is expensive. Let's just assume the seed's incremental check was correct 
                    # and efficient, but we want to improve it.
                    # Instead, let's do a full local search from a good start.
                    
                # Let's abandon the seed's incremental check and use a robust local search.
                pass
            if not found:
                # Fallback: just append 0 (will be fixed later)
                colors.append(0)
                found = True
            else:
                colors.append(color)
        return colors
    
    # Better approach:
    # 1. Generate a random coloring with k colors (start with small k, e.g., 3, 4, 5).
    # 2. If invalid, use hill climbing to fix violations.
    # 3. If valid, try to reduce the number of colors by re-coloring.
    
    # Since we need to minimize the number of labels used, we should iterate over possible 
    # color counts k from 3 upwards, and try to find a valid coloring using only k colors.
    
    # However, the labels don't need to be consecutive. But minimizing the number of *distinct* 
    # labels is the goal. So we can assume WLOG that the labels are 0..k-1 for some k.
    
    # Let's try k = 3, 4, 5, ... up to some limit.
    
    # For a given k, we run a local search:
    # - Start with a random assignment of 0..k-1.
    # - While violations exist:
    #   - Pick a random violated progression.
    #   - Try to change one of its positions to a different color (within 0..k-1) to fix it.
    #   - If stuck, try a random move (simulated annealing style).
    
    # Let's implement a simple local search with restarts.
    
    def local_search(k, max_steps=100000, timeout=5.0):
        """
        Try to find a valid coloring using exactly k colors (0..k-1).
        Returns a valid coloring if found, else None.
        """
        start_ls = time.time()
        
        for restart in range(50):  # 50 restarts
            if time.time() > timeout:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(128)]
            
            # Initial violation count
            viol = count_violations(colors)
            
            if viol == 0:
                return colors
            
            steps = 0
            while viol > 0 and steps < max_steps:
                if time.time() > timeout:
                    return None
                
                # Pick a random violated progression
                # To be efficient, we can just pick a random position and check its progressions,
                # or pick a random progression index.
                # Since n_progressions is ~ 128*128/6 ~ 2700, picking a random index is fine.
                p_idx = random.randint(0, n_progressions - 1)
                i, j, k_idx, l = progressions[p_idx]
                
                c = {colors[i], colors[j], colors[k_idx], colors[l]}
                if len(c) >= 3:
                    # Not violated? Maybe it was fixed by a previous move.
                    # Recalculate violation count periodically? 
                    # For simplicity, let's just try to change one of these positions 
                    # to a random color and see if it reduces violations.
                    pass
                else:
                    # It is violated. Try to change one of the 4 positions.
                    positions = [i, j, k_idx, l]
                    random.shuffle(positions)
                    
                    moved = False
                    for pos in positions:
                        old_color = colors[pos]
                        # Try a new random color
                        new_color = random.randint(0, k - 1)
                        if new_color == old_color:
                            continue
                        
                        colors[pos] = new_color
                        new_viol = count_violations(colors)
                        
                        if new_viol < viol:
                            viol = new_viol
                            moved = True
                            break
                        else:
                            # Accept with some probability if equal? Or reject?
                            # Let's reject for now (greedy).
                            colors[pos] = old_color
                    
                    if not moved:
                        # Stuck. Try a random move to escape local minimum.
                        pos = random.randint(0, 127)
                        old_color = colors[pos]
                        new_color = random.randint(0, k - 1)
                        colors[pos] = new_color
                        new_viol = count_violations(colors)
                        # Accept if not worse? Or always accept to escape?
                        # Let's accept if new_viol <= viol + 2 (small penalty)
                        if new_viol <= viol + 2:
                            viol = new_viol
                        else:
                            colors[pos] = old_color
                            # If we can't improve or escape, restart this search
                            break
                    
                    steps += 1
            
            if viol == 0:
                return colors
            
            # Check time
            if time.time() > start_ls + timeout:
                break
        
        return None

    # Try to find a valid coloring for k = 3, 4, 5, ...
    # The seed likely uses more than 3 colors. 
    # Let's start with k=3. If it fails quickly, try k=4, etc.
    
    # We need to be careful with time. 
    # Let's allocate time per k.
    
    # First, let's see what the seed produces to get a baseline.
    # The seed is deterministic. Let's run it.
    seed_colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            # Check condition 1: 4-term progressions ending at x-1
            for d in range(1, (x - 1) // 3 + 1):
                idxs = [x - 1 - 3 * d, x - 1 - 2 * d, x - 1 - d, x - 1]
                # Note: x-1 is the current position, not in seed_colors yet.
                # The set is {color, colors[idxs[0]], colors[idxs[1]], colors[idxs[2]]}
                # Wait, the seed code:
                # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 ...)
                # This is {color, colors[x-1-d], colors[x-1-2d], colors[x-1-3d]}
                # This corresponds to the progression (x-1-3d, x-1-2d, x-1-d, x-1).
                # So it checks progressions where x-1 is the LAST term.
                
                c = {color, seed_colors[x - 1 - d], seed_colors[x - 1 - 2 * d], seed_colors[x - 1 - 3 * d]}
                if len(c) < 3:
                    ok = False
                    break
            
            if not ok:
                continue
                
            # Check condition 2: 
            # all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This checks for each d, that NOT (colors[x-1-d] == color AND colors[x-1-2d] == color).
            # Why? If colors[x-1-d] == color and colors[x-1-2d] == color, 
            # and we set colors[x-1] = color, then we have a 3-term constant sequence?
            # Or does it prevent a future violation?
            # Actually, if we have (a, a+d, a+2d, a+3d) and we are at a+3d, 
            # the previous check covered it.
            # What about progressions where x-1 is NOT the last term?
            # E.g., (x-1, x-1+d, x-1+2d, x-1+3d). These are not checked yet.
            # The second condition seems to be a heuristic to avoid creating "bad" pairs 
            # that might lead to violations later.
            
            for d in range(1, (x - 1) // 2 + 1):
                if seed_colors[x - 1 - d] == color and seed_colors[x - 1 - 2 * d] == color:
                    ok = False
                    break
            
            if ok:
                seed_colors.append(color)
                break
        else:
            # Should not happen if 128 colors are available
            seed_colors.append(0)
            
    seed_valid = is_valid(seed_colors)
    seed_color_count = count_colors(seed_colors)
    
    # Print for debugging (remove in final)
    # print(f"Seed valid: {seed_valid}, colors: {seed_color_count}")
    
    # Now, try to improve using local search.
    # Start with the seed coloring as a candidate.
    best_colors = seed_colors if seed_valid else None
    best_count = seed_color_count if seed_valid else 128
    
    # Try to find colorings with fewer colors.
    # We know the answer is likely small (3, 4, or 5).
    
    # Let's try k from 3 to best_count - 1.
    # If we find a valid one with k colors, update best.
    
    # We will use the local_search function defined above.
    # But it's slow. Let's optimize it.
    
    # Optimized local search:
    # - Maintain a list of violated progressions.
    # - When a position changes, update the violation status of all progressions involving it.
    
    # This is complex. Given the small size (128), a simple brute-force local search 
    # with restarts might be fast enough if we limit steps.
    
    # Let's try a different approach: 
    # Randomized greedy construction with backtracking? No, too slow.
    
    # Let's stick to the local search but make it faster.
    # Precompute for each position, the list of progressions it belongs to.
    
    pos_to_progs = [[] for _ in range(128)]
    for idx, (i, j, k, l) in enumerate(progressions):
        pos_to_progs[i].append(idx)
        pos_to_progs[j].append(idx)
        pos_to_progs[k].append(idx)
        pos_to_progs[l].append(idx)
    
    def fast_local_search(k, max_steps=50000, timeout=5.0):
        start_ls = time.time()
        
        for restart in range(100):
            if time.time() > timeout:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(128)]
            
            # Calculate initial violations
            viol_count = 0
            viol_set = set()
            for idx, (i, j, k_idx, l) in enumerate(progressions):
                if len({colors[i], colors[j], colors[k_idx], colors[l]}) < 3:
                    viol_count += 1
                    viol_set.add(idx)
            
            if viol_count == 0:
                return colors
            
            steps = 0
            while viol_count > 0 and steps < max_steps:
                if time.time() > timeout:
                    return None
                
                # Pick a random violated progression
                if not viol_set:
                    break
                p_idx = random.choice(list(viol_set))
                i, j, k_idx, l = progressions[p_idx]
                
                # Try to fix it by changing one of the 4 positions
                positions = [i, j, k_idx, l]
                random.shuffle(positions)
                
                moved = False
                for pos in positions:
                    old_color = colors[pos]
                    # Try all k colors? Or random?
                    # Let's try random first, then all if stuck.
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        
                        colors[pos] = new_color
                        
                        # Update violation count
                        # Only progressions involving 'pos' can change status.
                        delta = 0
                        affected = pos_to_progs[pos]
                        
                        # We need to check the status of these progressions.
                        # This is O(deg(pos)) which is ~ 128/4 * 3 ~ 100? No, 
                        # each pos is in ~ 3*128/4 ~ 100 progressions? 
                        # Actually, a position is in at most ~ 128/1 + 128/2 + ... progressions?
                        # No, it's fixed. For a given pos, d can be 1..128/3, and pos can be 
                        # 1st, 2nd, 3rd, or 4th term. So ~ 4 * 128/3 ~ 170 progressions.
                        
                        # This is too slow for inner loop.
                        # Let's just recalculate total violations if k is small?
                        # No, that's O(N) = 2700.
                        # 50000 steps * 2700 = 135M ops. Might be too slow in Python.
                        
                        # Let's go back to the simpler version but with a time check.
                        pass
                    
                    # Revert
                    colors[pos] = old_color
                
                # If we didn't move, try a random kick
                pos = random.randint(0, 127)
                old_color = colors[pos]
                new_color = random.randint(0, k - 1)
                colors[pos] = new_color
                
                # Recalculate violations
                new_viol = count_violations(colors)
                
                # Accept if better or equal
                if new_viol <= viol_count:
                    viol_count = new_viol
                    # Update viol_set? Too expensive. Just rely on random picks.
                    # If viol_count is 0, we are done.
                else:
                    # Revert if worse? Or accept with prob?
                    # Let's revert for now.
                    colors[pos] = old_color
                    break  # Restart this search
                
                steps += 1
            
            if viol_count == 0:
                return colors
        
        return None

    # The fast_local_search is still messy. Let's use a cleaner, albeit slower, 
    # but reliable local search that works within the time limit.
    
    def robust_local_search(k, max_restarts=20, timeout=5.0):
        start_ls = time.time()
        
        for _ in range(max_restarts):
            if time.time() > timeout:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(128)]
            
            # Hill climbing
            for step in range(20000):
                if time.time() > timeout:
                    return None
                    
                viol = count_violations(colors)
                if viol == 0:
                    return colors
                
                # Pick a random position
                pos = random.randint(0, 127)
                old_color = colors[pos]
                
                # Try a random new color
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_viol = count_violations(colors)
                
                if new_viol < viol:
                    # Accept
                    pass
                elif new_viol == viol:
                    # Accept with prob 0.5
                    if random.random() < 0.5:
                        pass
                    else:
                        colors[pos] = old_color
                else:
                    # Reject
                    colors[pos] = old_color
            
            # Check final
            if count_violations(colors) == 0:
                return colors
        
        return None

    # Try k = 3, 4, 5, ...
    # Start with the seed's color count as the upper bound.
    # If seed is valid, we have a baseline.
    
    current_best = seed_colors if seed_valid else None
    current_best_count = seed_color_count if seed_valid else 128
    
    # Try smaller k
    for k in range(3, current_best_count):
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid coloring with k colors
        result = robust_local_search(k, max_restarts=10, timeout=3.0)
        
        if result is not None:
            # Verify
            if is_valid(result):
                new_count = count_colors(result)
                if new_count < current_best_count:
                    current_best_count = new_count
                    current_best = result
                    # If we found 3 colors, that's likely optimal (or close).
                    # Break early if k=3?
                    if k == 3:
                        break

    # If we didn't find anything better, use the seed.
    if current_best is None:
        # Fallback to a random valid coloring if seed was invalid
        # (Seed should be valid, but just in case)
        current_best = seed_colors
        
    # Final check
    if not is_valid(current_best):
        # This should not happen
        pass
        
    return current_best

# Test
if __name__ == "__main__":
    res = construct()
    print(len(set(res)))
    print(is_valid(res))
