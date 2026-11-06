import time
import random
import sys

def construct():
    """
    Simulated Annealing approach for 4-AP coloring.
    
    Strategy:
    1. Precompute all 4-APs and their constituent positions.
    2. Use an incremental violation counter: for each AP, track the count of each color.
       A violation occurs if the number of distinct colors in the AP < 3.
    3. Simulated Annealing loop:
       - Pick a random position.
       - Pick a new color (biased towards colors that fix violations, or random).
       - Compute delta in total violations.
       - Accept if delta < 0 or with probability exp(-delta / T).
       - Cool T over time.
    4. Once a valid solution (0 violations) is found, attempt to merge colors to minimize the count.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    n = 128
    
    # Precompute all nonconstant 4-term arithmetic progressions (a, a+d, a+2d, a+3d) within 1..128
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps.append((a - 1, b - 1, c - 1, e - 1))
    
    num_aps = len(aps)
    
    # Map each position to the list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, (i, j, k, l) in enumerate(aps):
        for pos in (i, j, k, l):
            pos_to_aps[pos].append(idx)
    
    def count_violations_full(colors):
        """Full recount of violations (used for initial check and final verification)."""
        violations = 0
        for (i, j, k, l) in aps:
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                violations += 1
        return violations

    def try_simulated_annealing(initial_colors, max_iters=200000):
        colors = list(initial_colors)
        
        # Initialize AP state: for each AP, store a dict of color -> count
        ap_color_counts = [{} for _ in range(num_aps)]
        ap_violations = [0] * num_aps
        total_violations = 0
        
        for idx, (i, j, k, l) in enumerate(aps):
            counts = {}
            for pos in (i, j, k, l):
                c = colors[pos]
                counts[c] = counts.get(c, 0) + 1
            ap_color_counts[idx] = counts
            num_distinct = len(counts)
            if num_distinct < 3:
                ap_violations[idx] = 1
                total_violations += 1
        
        if total_violations == 0:
            return colors
        
        best_colors = list(colors)
        best_violations = total_violations
        best_num_colors = len(set(colors))
        
        # Annealing parameters
        T = 50.0
        cooling_rate = 0.99995  # Slow cooling to explore more
        min_T = 0.1
        
        iters = 0
        while iters < max_iters:
            if time.time() > deadline - 5:
                break
            iters += 1
            
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Pick a new color
            # Strategy: 50% chance to pick a random color from 0..7 (small palette for efficiency)
            # 50% chance to pick a color that is currently "needed" to fix a violation in one of its APs
            # For simplicity and speed, let's just pick from a small set of candidate colors
            # If we want to minimize colors, we should mostly use existing colors or 0,1,2
            used_colors = set(colors)
            candidate_colors = list(used_colors)
            # Add a few new colors if we haven't found a solution yet?
            # Actually, to minimize colors, we should stick to a small set.
            # Let's try colors 0..5 initially.
            for c in range(6):
                if c not in candidate_colors:
                    candidate_colors.append(c)
            
            new_color = random.choice(candidate_colors)
            if new_color == old_color:
                continue
            
            # Calculate delta in violations
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                counts = ap_color_counts[ap_idx]
                
                # Remove old color
                counts[old_color] -= 1
                if counts[old_color] == 0:
                    del counts[old_color]
                
                # Add new color
                if new_color not in counts:
                    counts[new_color] = 1
                else:
                    counts[new_color] += 1
                
                old_num_distinct = len(counts) - 1 if old_color != new_color else len(counts) # This logic is slightly off, let's just recompute distinct count
                # Simpler: just count distinct in the modified dict
                new_num_distinct = len(counts)
                
                # We need to know if this AP was a violation before and is now, or vice versa
                # ap_violations[ap_idx] is 1 if violation, 0 otherwise
                was_viol = ap_violations[ap_idx]
                is_viol = 1 if new_num_distinct < 3 else 0
                
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
                
                # Store the new violation state for this AP (tentatively)
                # We will revert if we don't accept
                # To keep it clean, let's just calculate the delta and if we accept, we update the state.
                # But we modified counts in place. We need to revert if we reject.
                # Let's store the old state to revert.
                # Optimization: Only revert if rejected.
            
            # Decision: Accept or Reject
            accepted = False
            if delta < 0:
                accepted = True
            else:
                # Probability of accepting worse state
                # If T is very small, exp(-delta/T) is ~0
                if T > min_T:
                    prob = random.random()
                    # Use log to avoid underflow? delta is small integer.
                    # exp(-delta / T)
                    if delta == 0:
                        accepted = True
                    else:
                        # Simple check
                        if random.random() < 0.01: # Small chance to accept neutral
                            accepted = True
                        else:
                            # Calculate exp
                            # delta is usually small (1, 2, 3)
                            # T starts at 50
                            # exp(-1/50) ~ 0.98
                            # exp(-3/50) ~ 0.94
                            # As T drops to 0.1, exp(-1/0.1) = exp(-10) ~ 0.00004
                            try:
                                import math
                                if delta * T < 700: # Avoid overflow in exp? No, exp(-x) is safe for positive x
                                    p_accept = math.exp(-delta / T)
                                else:
                                    p_accept = 0.0
                                if random.random() < p_accept:
                                    accepted = True
                            except:
                                pass
            
            if accepted:
                # Commit the change
                colors[pos] = new_color
                for ap_idx in pos_to_aps[pos]:
                    # The counts were already updated in the loop above.
                    # We just need to update ap_violations
                    counts = ap_color_counts[ap_idx]
                    new_num_distinct = len(counts)
                    is_viol = 1 if new_num_distinct < 3 else 0
                    if is_viol != ap_violations[ap_idx]:
                        ap_violations[ap_idx] = is_viol
                        total_violations += is_viol - (1 - is_viol) # Wait, delta was already calculated.
                        # Actually, total_violations += delta is correct if delta was calculated correctly.
                        # But I calculated delta by iterating. Let's just use delta.
                        pass
                total_violations += delta
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = list(colors)
                    best_num_colors = len(set(colors))
                    
                if total_violations == 0:
                    # Try to merge colors to minimize distinct count
                    used = sorted(set(colors))
                    if len(used) <= 3:
                        return colors
                    # Attempt local merge
                    merged = False
                    for a in range(len(used)):
                        for b in range(a + 1, len(used)):
                            # Try merging b into a
                            test_colors = list(colors)
                            for p in range(n):
                                if test_colors[p] == used[b]:
                                    test_colors[p] = used[a]
                            # Check if still valid
                            # Fast check: only check APs affected by change
                            # For simplicity, full check if close to time limit? No, full check is O(N_aps) ~ 1000. Fast enough.
                            if count_violations_full(test_colors) == 0:
                                return test_colors
            else:
                # Revert the change in ap_color_counts
                for ap_idx in pos_to_aps[pos]:
                    counts = ap_color_counts[ap_idx]
                    # Revert new_color
                    counts[new_color] -= 1
                    if counts[new_color] == 0:
                        del counts[new_color]
                    # Revert old_color
                    if old_color not in counts:
                        counts[old_color] = 1
                    else:
                        counts[old_color] += 1
            
            # Cool down
            T *= cooling_rate
        
        return best_colors if best_violations == 0 else best_colors

    # Main logic
    best_result = None
    best_score = float('inf')
    
    # Strategy: Start with a small number of colors and anneal
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initial coloring with k colors
        k = 3 if attempt < 10 else 4
        initial = [random.randint(0, k - 1) for _ in range(n)]
        
        result = try_simulated_annealing(initial, max_iters=50000)
        
        if result is not None:
            v = count_violations_full(result)
            if v == 0:
                num_colors = len(set(result))
                if num_colors < best_score:
                    best_score = num_colors
                    best_result = list(result)
                if best_score == 3:
                    # 3 is the theoretical minimum (2 colors can't satisfy >=3 distinct in 4-AP)
                    # Try a few more to ensure stability or find a better structure? No, 3 is min.
                    # But maybe we can find a solution with 3 colors that is more "robust"?
                    # The score is just the number of colors. So 3 is the best possible.
                    # We can break early if we find 3, but let's do a couple more to be sure.
                    if attempt >= 5:
                        break
    
    if best_result is None:
        # Fallback: Greedy construction
        colors = []
        for x in range(1, n + 1):
            found = False
            for color in range(128):
                valid = True
                # Check all APs ending at x? No, check all APs containing x.
                # But x is the last element added, so we only need to check APs where x is one of the elements.
                # Since we build sequentially, x is the largest index.
                # APs containing x can have x as a, a+d, a+2d, or a+3d.
                # If x is a, then a+d, a+2d, a+3d are > x, not colored yet. So no constraint from future.
                # If x is a+d, then a < x, a+2d > x, a+3d > x. Constraint on a, x, future, future.
                # Wait, the condition is on the set of colors in the AP.
                # If some positions are not colored yet, we can't evaluate the constraint fully.
                # So greedy from 1 to 128 only works if we ensure that when we color x,
                # all APs that are "complete" (all 4 positions <= x) are valid.
                # APs that are not complete can be violated now but fixed later?
                # No, if we color x, and an AP has x and 3 previous positions, it is complete.
                # If an AP has x and some future positions, it is not complete.
                # So we must ensure that for any AP (a, b, c, d) where d == x, the colors of a,b,c,d are valid.
                # Also, if x is a, b, or c, the AP is not complete yet.
                # So we only check APs where x is the last element (a+3d = x).
                
                # Actually, we need to check if adding color `color` at position `x` creates a violation
                # in any AP that is now complete.
                # An AP is complete if its max index is x.
                # Since indices are 1..128, and we are at x, the only new APs completed are those where x is the max.
                # In a 4-AP (a, a+d, a+2d, a+3d), the max is a+3d.
                # So we check all (a, d) such that a+3d = x.
                
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a >= 1:
                        b = a + d
                        c = a + 2 * d
                        # Colors are 0-indexed in list `colors`
                        # a-1, b-1, c-1 are in `colors`
                        # x-1 is current
                        ap_colors = {colors[a - 1], colors[b - 1], colors[c - 1], color}
                        if len(ap_colors) < 3:
                            valid = False
                            break
                
                if valid:
                    colors.append(color)
                    found = True
                    break
            
            if not found:
                colors.append(random.randint(0, 127))
        
        v = count_violations_full(colors)
        if v == 0:
            num_colors = len(set(colors))
            if num_colors < best_score:
                best_score = num_colors
                best_result = list(colors)
    
    if best_result is not None:
        return best_result
    
    return list(range(n))
