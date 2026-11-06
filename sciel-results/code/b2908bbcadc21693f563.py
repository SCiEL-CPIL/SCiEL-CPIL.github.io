import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 128

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # pos_to_aps: for each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    # Helper to count distinct colors in an AP
    def count_distinct_ap(ap_idx):
        a, b, c, d_pos = ap_list[ap_idx]
        return len({colors[a], colors[b], colors[c], colors[d_pos]})
        
    def count_distinct_ap_val(val_list):
        return len(set(val_list))

    def is_valid_coloring(colors):
        for ap_idx in range(len(ap_list)):
            a, b, c, d_pos = ap_list[ap_idx]
            if count_distinct_ap_val([colors[a], colors[b], colors[c], colors[d_pos]]) < 3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # 1. Greedy Construction (Sequential)
    def greedy_construct(random_order=False):
        colors = [-1] * n
        if random_order:
            # Randomized greedy might escape local minima or find different structures?
            # Actually, for APs, the structure is heavily constrained by local positions.
            # Let's stick to left-to-right for deterministic behavior first, 
            # but maybe try a few random color orderings.
            pass
        
        for x in range(n):
            # Try colors 0..127
            found = False
            for color in range(128):
                valid = True
                # Check all 4-term APs ending at x (or involving x where x is the max index)
                # An AP (a, a+d, a+2d, a+3d) involves x if x is one of them.
                # Since we fill left to right, when placing color at x,
                # only APs where x is the last element (a+3d) or part of a set where others are unassigned?
                # No, we must ensure that the APs that are *completed* (all 4 indices <= x) are valid.
                # APs involving x and future indices are not completed yet.
                # So we only check APs (a, b, c, d) such that d == x.
                # d = a + 3d_step => a = x - 3d_step.
                for d_step in range(1, (x // 3) + 1):
                    a = x - 3 * d_step
                    if a < 0: continue
                    b = a + d_step
                    c = a + 2 * d_step
                    # d = x
                    val_set = {colors[a], colors[b], colors[c], color}
                    if len(val_set) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    found = True
                    break
            if not found:
                # Fallback: pick a random color, though with 128 colors this shouldn't happen often
                colors[x] = random.randint(0, 127)
        return colors

    # Initial greedy solution
    best_colors = greedy_construct()
    best_count = count_colors(best_colors)
    
    # If we already have 3, we are optimal.
    if best_count <= 3:
        return best_colors

    # 2. Local Search (Simulated Annealing / Hill Climbing)
    # Goal: Minimize number of colors used.
    # Strategy: 
    #   - Start with a valid coloring using K colors.
    #   - Try to recolor a random position to a color < max_used_color to reduce max color or remove a color.
    #   - Or more directly: try to change a color to another existing color to see if we can drop a color.
    
    # Better approach: 
    #   Keep a set of allowed colors. 
    #   Start with allowed = set(best_colors). 
    #   Try to remove a color c from allowed.
    #   If removing c makes the coloring invalid, we need to recolor positions with c.
    #   We can recolor them to other allowed colors if valid.
    
    # Let's implement a "Color Elimination" local search.
    
    def try_eliminate_color(colors, target_color, allowed_colors):
        """
        Try to recolor all positions with target_color to other colors in allowed_colors\{target_color}
        such that the coloring remains valid.
        Returns new_colors if successful, else None.
        """
        new_colors = list(colors)
        positions_with_target = [i for i, c in enumerate(new_colors) if c == target_color]
        
        # We need to assign new colors to these positions.
        # This is a CSP itself. Let's use a simple backtracking for these specific positions.
        # Since the number of positions might be large, this could be slow.
        # However, if we do this randomly or with a limit, it might work.
        
        # Alternative: Simulated Annealing on the whole vector, with a cost function
        # Cost = number of colors used + M * (number of invalid APs)
        
    # Let's switch to Simulated Annealing with a composite cost.
    # Cost(C) = num_colors(C) * 1000 + num_invalid_aps(C)
    # We want to minimize this.
    
    def calc_cost(colors):
        num_colors = len(set(colors))
        num_invalid = 0
        for ap_idx in range(len(ap_list)):
            a, b, c, d_pos = ap_list[ap_idx]
            if count_distinct_ap_val([colors[a], colors[b], colors[c], colors[d_pos]]) < 3:
                num_invalid += 1
        return num_colors * 1000 + num_invalid

    # Initialize
    current_colors = list(best_colors)
    current_cost = calc_cost(current_colors)
    best_sa_colors = list(current_colors)
    best_sa_cost = current_cost
    
    # SA Parameters
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.95
    max_iter = 200000  # Adjust based on time
    
    def get_neighbor(colors):
        # Pick a random position and a random color
        i = random.randint(0, n - 1)
        # Pick a color from 0..127? Or from existing + new?
        # To minimize colors, we should prefer existing colors.
        # Let's pick a color from the set of currently used colors, or 0..3 to encourage consolidation.
        # Actually, allowing any color 0..127 lets us discover new structures, but we want to *reduce* colors.
        # Let's pick from 0..5 initially, or from the set of used colors.
        used_colors = set(colors)
        if used_colors:
            # 80% chance pick from used, 20% chance pick from 0..7 (to try to consolidate to low indices)
            if random.random() < 0.8:
                c = random.choice(list(used_colors))
            else:
                c = random.randint(0, 7)
        else:
            c = 0
        new_colors = list(colors)
        new_colors[i] = c
        return new_colors

    random.seed(42)
    iter_count = 0
    while time.time() < deadline - 5.0:
        neighbor = get_neighbor(current_colors)
        neighbor_cost = calc_cost(neighbor)
        
        delta_cost = neighbor_cost - current_cost
        
        if delta_cost < 0:
            current_colors = neighbor
            current_cost = neighbor_cost
            if current_cost < best_sa_cost:
                best_sa_cost = current_cost
                best_sa_colors = list(current_colors)
        else:
            # Accept with probability exp(-delta/T)
            if T > T_min and random.random() < pow(2.71828, -delta_cost / T):
                current_colors = neighbor
                current_cost = neighbor_cost
        
        T *= cooling_rate
        iter_count += 1
        
        # If we found a valid coloring with fewer colors than best, we can stop early if we reach 3
        if best_sa_cost // 1000 == 3 and best_sa_cost % 1000 == 0:
            break
            
        # If time is running out, break
        if time.time() > deadline - 2.0:
            break

    # If SA found a better solution, use it
    if best_sa_cost < best_cost * 1000:
        best_colors = best_sa_colors
        best_count = best_sa_cost // 1000
        if best_sa_cost % 1000 == 0:
             # It's a valid coloring
             return best_colors
        else:
             # It's invalid, so we can't return it directly. 
             # We need to repair it or fall back to the last valid one found during SA?
             # My SA didn't track the last valid state specifically, only the best cost.
             # If best cost has penalty, it's invalid.
             # We should keep track of the best *valid* coloring found.
             
             # Let's re-run SA but only accept valid moves? No, that's hill climbing and gets stuck.
             # Let's just check if the current `best_sa_colors` is valid.
             if is_valid_coloring(best_sa_colors):
                 return best_sa_colors
             else:
                 # Fallback to greedy if SA got stuck in invalid state
                 pass
                 
    # If SA didn't improve or failed, try a specific "Color Reduction" search.
    # Try to map colors to a smaller set.
    
    def attempt_k_color_solution(k):
        """
        Try to find a valid coloring using exactly colors 0..k-1.
        Uses backtracking with MRV and forward checking.
        Optimized for k=3,4,5.
        """
        if time.time() > deadline - 2.0:
            return None
            
        colors = [-1] * n
        ap_distinct_counts = [0] * len(ap_list) # Count of distinct colors in AP
        ap_unassigned_counts = [4] * len(ap_list)
        
        # To efficiently check consistency, we need to know the set of colors in each AP.
        # Storing sets is expensive. 
        # Alternative: 
        # When assigning a color to pos, for each AP containing pos:
        #   - If this is the first assignment to this AP, init set.
        #   - Add color to set.
        #   - Decrement unassigned count.
        #   - Check if constraint is violated.
        
        ap_color_sets = [set() for _ in range(len(ap_list))]
        
        def is_consistent(pos, color):
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                # Trial assignment
                new_set = set(ap_color_sets[ap_idx])
                new_set.add(color)
                
                # If the set of colors already in the AP (excluding pos) + this color 
                # results in a set that *cannot* reach 3 distinct colors with remaining slots?
                
                current_set = ap_color_sets[ap_idx]
                remaining_unassigned = ap_unassigned_counts[ap_idx] - 1 # After assigning pos
                
                # If pos was already in the set? No, pos is unassigned, so its color is not in current_set.
                
                # If len(new_set) >= 3: Safe.
                if len(new_set) >= 3:
                    continue
                    
                # If len(new_set) == 2:
                #   We need at least 1 new color from the remaining unassigned slots.
                #   If remaining_unassigned == 0: Fail (only 2 colors).
                #   If remaining_unassigned >= 1: Possible.
                if len(new_set) == 2:
                    if remaining_unassigned == 0:
                        return False
                    continue
                    
                # If len(new_set) == 1:
                #   We need at least 2 new colors from the remaining unassigned slots.
                #   If remaining_unassigned == 0: Fail (1 color).
                #   If remaining_unassigned == 1: Fail (max 2 colors).
                #   If remaining_unassigned >= 2: Possible.
                if len(new_set) == 1:
                    if remaining_unassigned < 2:
                        return False
                    continue
                    
                # If len(new_set) == 0 (impossible as we added color)
                
            return True

        def backtrack(pos):
            if time.time() > deadline - 2.0:
                return False
                
            if pos == n:
                return True
            
            # MRV Heuristic: Find unassigned position with fewest valid colors.
            # Since we are going in order 0..n-1, MRV requires scanning ahead or maintaining domains.
            # Scanning ahead is expensive. 
            # Let's stick to sequential order but with a good pruning.
            # Or, implement dynamic variable ordering.
            
            # For simplicity and speed in Python, let's try sequential first with strong pruning.
            # If it's too slow, we can switch to a more complex scheduler.
            
            # Find next unassigned position
            next_pos = -1
            for i in range(pos, n):
                if colors[i] == -1:
                    next_pos = i
                    break
            
            if next_pos == -1:
                return True
                
            # Try colors 0..k-1
            for c in range(k):
                if is_consistent(next_pos, c):
                    # Assign
                    colors[next_pos] = c
                    for ap_idx in pos_to_aps[next_pos]:
                        ap_color_sets[ap_idx].add(c)
                        ap_unassigned_counts[ap_idx] -= 1
                    
                    if backtrack(next_pos + 1):
                        return True
                        
                    # Unassign
                    colors[next_pos] = -1
                    for ap_idx in pos_to_aps[next_pos]:
                        ap_color_sets[ap_idx].remove(c)
                        ap_unassigned_counts[ap_idx] += 1
                        
            return False

        if backtrack(0):
            return colors
        return None

    # Try k=3
    result = attempt_k_color_solution(3)
    if result is not None:
        return result

    # Try k=4
    result = attempt_k_color_solution(4)
    if result is not None:
        return result

    # Try k=5
    result = attempt_k_color_solution(5)
    if result is not None:
        return result

    # If all else fails, return the best valid coloring found
    # If best_sa_colors is valid, return it.
    if is_valid_coloring(best_sa_colors):
        return best_sa_colors
        
    return best_colors

