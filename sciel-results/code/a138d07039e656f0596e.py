import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 128
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                progressions.append((a - 1, b - 1, c - 1, e - 1))
    
    num_progs = len(progressions)
    
    # Map position to list of (prog_index, other_positions)
    pos_progs = [[] for _ in range(n)]
    for pi, (a, b, c, e) in enumerate(progressions):
        for pos in (a, b, c, e):
            others = [p for p in (a, b, c, e) if p != pos]
            pos_progs[pos].append((pi, others))
    
    # We will try to find a 3-coloring.
    # State for each progression:
    # assigned_colors: bitmask of colors used (3 bits for 3 colors)
    # unassigned_count: number of positions not yet assigned
    
    colors = [-1] * n
    prog_assigned_masks = [0] * num_progs
    prog_unassigned_counts = [4] * num_progs
    
    # To speed up backtracking, we should pick the variable with the fewest remaining values (MRV)
    # or the one involved in the most constraints.
    # Given n=128, a simple index order might be too slow if it hits dead ends late.
    # Let's try a heuristic: order positions by their degree (number of progressions they are in).
    # Actually, positions near the middle are in more progressions.
    # Let's just use the natural order first, but with strong pruning.
    
    def is_dead(pi, assigned_mask, unassigned_count, available_colors_mask):
        # available_colors_mask: colors that *could* still be assigned? 
        # Actually, any color from 0..2 can be assigned to any unassigned slot.
        # So max distinct colors we can add is unassigned_count.
        # Current distinct = popcount(assigned_mask)
        # Max possible distinct = popcount(assigned_mask) + unassigned_count
        # If this is < 3, it's dead.
        # Note: If assigned_mask already has 3 colors, it's safe (satisfied).
        # If assigned_mask has 2 colors, we need at least 1 new color.
        # If assigned_mask has 1 color, we need at least 2 new colors.
        # If assigned_mask has 0 colors, we need at least 3 new colors.
        
        # Wait, "new colors" means colors NOT in assigned_mask.
        # Since we have 3 colors total, if assigned_mask has 1 color, there are 2 other colors available.
        # So we can definitely add 2 distinct colors if we have 2 slots.
        # The constraint is simply: Can we reach 3 distinct colors?
        # Yes, if (current_distinct + unassigned_count) >= 3 is NOT sufficient if the unassigned slots are forced to be same color? No, we control the assignment.
        # The only reason it's dead is if we *can't* get 3 distinct colors.
        # Since we can pick any of the 3 colors for any slot, the only limit is the number of slots.
        # So:
        current_distinct = bin(assigned_mask).count('1')
        if current_distinct == 3:
            return False # Already satisfied
        if current_distinct + unassigned_count < 3:
            return True
        return False

    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
            
        # Heuristic: Try to assign the position that is most constrained?
        # For simplicity and speed, we stick to idx order but prune aggressively.
        # However, if we fail, we backtrack.
        
        pos = idx
        # Try colors 0, 1, 2
        for color in range(3):
            # Check if assigning this color creates an immediate conflict or dead-end
            valid = True
            # We need to update the state for all progressions containing pos
            # and check if any become dead.
            
            # To avoid modifying global state and reverting, we can simulate or just check.
            # Since we are modifying arrays, we must revert.
            
            # Optimization: Pre-check if this color is "forced" or "forbidden" by any progression?
            # A progression is violated if it's complete and has < 3 colors.
            # A progression is dead if it can't reach 3 colors.
            
            # Let's just do the update and revert.
            # This is O(degree) per step.
            
            affected_progs = pos_progs[pos]
            changes = []
            
            for pi, others in affected_progs:
                old_mask = prog_assigned_masks[pi]
                new_mask = old_mask | (1 << color)
                old_count = prog_unassigned_counts[pi]
                new_count = old_count - 1
                
                # Check if new state is dead
                if is_dead(pi, new_mask, new_count, 0):
                    valid = False
                    # Revert changes made so far in this loop iteration
                    for p_idx, m_idx in changes:
                        prog_assigned_masks[p_idx] = m_idx
                    break
                
                # Also check if it's complete and invalid
                if new_count == 0 and bin(new_mask).count('1') < 3:
                    valid = False
                    for p_idx, m_idx in changes:
                        prog_assigned_masks[p_idx] = m_idx
                    break
                
                # Apply change
                changes.append((pi, old_mask))
                prog_assigned_masks[pi] = new_mask
                prog_unassigned_counts[pi] = new_count
            
            if valid:
                colors[pos] = color
                if backtrack(idx + 1):
                    return True
                colors[pos] = -1
                # Revert changes
                for p_idx, m_idx in changes:
                    prog_assigned_masks[p_idx] = m_idx
                    # We also need to revert counts. 
                    # We didn't store old counts in 'changes'. 
                    # Let's fix 'changes' to store (pi, old_mask, old_count)
            
            # If we broke early due to invalid, we already reverted masks.
            # We need to revert counts too.
            # Let's refactor the revert logic.

        return False

    # The above backtrack function has a bug in revert logic for counts.
    # Let's rewrite the inner loop to be cleaner.

    def backtrack_v2(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
            
        pos = idx
        
        for color in range(3):
            valid = True
            # Store old states for reverting
            old_states = []
            
            for pi, others in pos_progs[pos]:
                old_mask = prog_assigned_masks[pi]
                old_count = prog_unassigned_counts[pi]
                
                new_mask = old_mask | (1 << color)
                new_count = old_count - 1
                
                # Check validity
                # 1. If complete, must have >= 3 colors
                if new_count == 0:
                    if bin(new_mask).count('1') < 3:
                        valid = False
                        break
                # 2. If not complete, check if it's dead
                else:
                    current_distinct = bin(new_mask).count('1')
                    # We can add at most new_count distinct colors
                    if current_distinct + new_count < 3:
                        valid = False
                        break
                
                # Apply
                old_states.append((pi, old_mask, old_count))
                prog_assigned_masks[pi] = new_mask
                prog_unassigned_counts[pi] = new_count
            
            if valid:
                colors[pos] = color
                if backtrack_v2(idx + 1):
                    return True
                colors[pos] = -1
                # Revert
                for pi, old_mask, old_count in old_states:
                    prog_assigned_masks[pi] = old_mask
                    prog_unassigned_counts[pi] = old_count
            
            # If not valid, old_states is partially filled or empty.
            # If we broke early, we need to revert the ones we DID apply.
            # The loop above breaks before appending if invalid.
            # So if valid is False, we might have appended some valid ones before hitting an invalid one?
            # No, the loop breaks immediately on invalid.
            # So if valid is False, we have appended 0..k states.
            # We must revert them.
            else:
                for pi, old_mask, old_count in old_states:
                    prog_assigned_masks[pi] = old_mask
                    prog_unassigned_counts[pi] = old_count

        return False

    # Initialize
    # We can use symmetry breaking: colors[0] = 0
    colors[0] = 0
    # Update progressions containing 0
    for pi, others in pos_progs[0]:
        prog_assigned_masks[pi] |= 1 # color 0 is bit 0
        prog_unassigned_counts[pi] -= 1

    if backtrack_v2(1):
        return colors

    # If 3 colors failed (unlikely or timeout), try 4 colors?
    # The task says "Minimize the number of labels".
    # If 3 fails, we must try 4.
    # But 3 should work. Let's add a fallback to a known good pattern if search fails.
    # Pattern: i % 3 ?
    # Let's check if i % 3 works.
    # AP: a, a+d, a+2d, a+3d.
    # Colors: a%3, (a+d)%3, (a+2d)%3, (a+3d)%3 = a%3.
    # If d=1: 0,1,2,0 -> {0,1,2} OK.
    # If d=2: 0,2,1,0 -> {0,1,2} OK.
    # If d=3: 0,0,0,0 -> {0} FAIL.
    # So i%3 does not work for d=3.
    
    # How about i % 4?
    # d=1: 0,1,2,3 -> 4 colors.
    # d=2: 0,2,0,2 -> {0,2} FAIL.
    
    # How about a random restart strategy if backtracking is too slow?
    # Given the 60s limit, and the fact that 3-coloring exists, the backtracker should find it.
    # The bottleneck is the depth-first search exploring dead ends.
    # To improve, we can randomize the order of colors or positions.
    
    # Let's try a few random restarts of the backtracker with different color orderings.
    # Actually, let's just return the best effort or a heuristic if it times out.
    # But we must return a valid coloring.
    
    # Fallback: Greedy with random restarts
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        random_colors = [random.randint(0, 2) for _ in range(n)]
        # Check validity
        valid = True
        for a, b, c, e in progressions:
            if len({random_colors[a], random_colors[b], random_colors[c], random_colors[e]}) < 3:
                valid = False
                break
        if valid:
            return random_colors
            
    # If all else fails, return a 128-color solution (identity map) which is always valid.
    # Identity: i -> i.
    # AP: a, a+d, a+2d, a+3d are distinct if d!=0.
    # So 4 distinct colors. Valid.
    # Number of colors used: 128.
    # This is a valid fallback.
    return list(range(n))
