"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # Compute degree of each position
    degree = [len(pos_to_aps[i]) for i in range(n)]

    # Order positions by degree descending (most constrained first)
    order = sorted(range(n), key=lambda x: -degree[x])

    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v

    def try_reduce_colors_aggressive(colors, time_limit):
        """
        Aggressively try to reduce the number of colors by merging color classes.
        Returns (best_colors, best_num_colors, best_violations).
        """
        best_colors = colors[:]
        best_num_colors = len(set(colors))
        best_violations = count_violations(colors)
        
        # We'll try to merge pairs of colors into one
        # Strategy: repeatedly find a color pair whose merge causes fewest new violations
        # and merge it if it reduces or maintains the number of colors while keeping violations low
        
        colors = colors[:]
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1
        
        # Get list of colors actually used
        used_colors = sorted([c for c in range(128) if color_counts[c] > 0])
        
        # We'll try to reduce by merging colors
        # Keep a list of "active" colors
        active = set(used_colors)
        
        while time.time() < time_limit and len(active) > 3:
            # Try all pairs of active colors to see which merge is best
            active_list = sorted(active)
            best_pair = None
            best_score = float('inf')
            
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    c1, c2 = active_list[i], active_list[j]
                    
                    # Simulate merging c2 into c1
                    # Count how many positions would change
                    positions_to_change = [p for p in range(n) if colors[p] == c2]
                    
                    # For each such position, check impact on violations
                    delta_violations = 0
                    # We need to be careful: merging c2->c1 might create new violations
                    # or remove existing ones (if c1 and c2 were the only two colors in some AP)
                    
                    # Quick estimate: for each AP containing a position with color c2,
                    # check if the merge would reduce distinct colors below 3
                    
                    # Build set of affected APs
                    affected_ap_set = set()
                    for p in positions_to_change:
                        for ap_idx in pos_to_aps[p]:
                            affected_ap_set.add(ap_idx)
                    
                    # Count violations before and after for affected APs
                    v_before = 0
                    v_after = 0
                    for ap_idx in affected_ap_set:
                        ap = aps[ap_idx]
                        # Before
                        cs_before = set(colors[ap[k]] for k in range(4))
                        if len(cs_before) < 3:
                            v_before += 1
                        
                        # After (simulate)
                        cs_after = set()
                        for k in range(4):
                            pos = ap[k]
                            if colors[pos] == c2:
                                cs_after.add(c1)
                            else:
                                cs_after.add(colors[pos])
                        if len(cs_after) < 3:
                            v_after += 1
                    
                    delta_v = v_after - v_before
                    # Score: prefer merges that don't increase violations, and prefer merging larger groups
                    score = delta_v * 1000 - min(color_counts[c1], color_counts[c2])
                    
                    if score < best_score:
                        best_score = score
                        best_pair = (c1, c2)
            
            if best_pair is None:
                break
            
            c1, c2 = best_pair
            # Check if this merge would increase violations too much
            # We allow some increase if it reduces color count significantly
            
            # Perform the merge
            for p in range(n):
                if colors[p] == c2:
                    colors[p] = c1
            
            color_counts[c2] = 0
            color_counts[c1] += color_counts[c2] if color_counts[c2] > 0 else 0
            # Actually need to recompute since we changed positions
            color_counts = [0] * 128
            for c in colors:
                color_counts[c] += 1
            
            active.discard(c2)
            
            # Update best if improved
            v_now = count_violations(colors)
            nc_now = len(set(colors))
            if v_now == 0 and nc_now < best_num_colors:
                best_colors = colors[:]
                best_num_colors = nc_now
                best_violations = v_now
            elif v_now < best_violations or (v_now == best_violations and nc_now < best_num_colors):
                best_colors = colors[:]
                best_num_colors = nc_now
                best_violations = v_now
        
        return best_colors, best_num_colors, best_violations

    def local_search_refine(colors, time_limit):
        """Simulated annealing to reduce color count while maintaining validity."""
        colors = colors[:]
        current_violations = count_violations(colors)
        current_colors_used = len(set(colors))
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1

        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used

        temp = 3.0
        cooling = 0.997
        min_temp = 0.01

        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]

            used = set(colors)
            candidates = list(used)
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)

            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0

            affected_aps = pos_to_aps[pos]

            curr_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                if len(c) < 3:
                    curr_invalid += 1

            for new_color in candidates:
                if new_color == old_color:
                    continue

                colors[pos] = new_color

                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    if len(c) < 3:
                        new_invalid += 1

                delta_v = new_invalid - curr_invalid
                delta_c = 0
                if color_counts[old_color] == 1:
                    delta_c -= 1
                if color_counts[new_color] == 0:
                    delta_c += 1

                score = delta_v * 100 + delta_c
                best_score = best_delta_v * 100 + best_delta_c

                if score < best_score:
                    best_score = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c

            colors[pos] = old_color

            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                energy = best_delta_v * 100 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob

            if accept:
                colors[pos] = best_new_color
                color_counts[old_color] -= 1
                color_counts[best_new_color] += 1
                current_violations += best_delta_v
                current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)

                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]

                if current_violations == 0 and current_colors_used <= 3:
                    break

            temp *= cooling

        return best_colors, best_violations, best_colors_used

    # Strategy: 
    # 1. Start with a random valid-ish coloring
    # 2. Use local search to fix violations and reduce colors
    # 3. Use aggressive color reduction
    # 4. Repeat with different seeds

    best_result = None
    best_num_colors = 999

    # Try multiple random initializations
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(seed)
        
        # Random initialization with few colors
        num_init_colors = rng.choice([3, 4, 5])
        colors = [rng.randint(0, num_init_colors - 1) for _ in range(n)]
        
        # Local search to fix violations and reduce colors
        ls_time_limit = min(time.time() + 2.0, deadline - 3.0)
        refined, v, nc = local_search_refine(colors, ls_time_limit)
        
        if v == 0 and nc < best_num_colors:
            best_result = refined
            best_num_colors = nc
            
            # If we have 3 colors, we're done
            if nc <= 3:
                break
        
        # Try aggressive reduction if we have a valid coloring
        if best_result is not None and time.time() < deadline - 3.0:
            aggr_time_limit = min(time.time() + 3.0, deadline - 2.0)
            aggr_result, aggr_nc, aggr_v = try_reduce_colors_aggressive(best_result, aggr_time_limit)
            if aggr_v == 0 and aggr_nc < best_num_colors:
                best_result = aggr_result
                best_num_colors = aggr_nc
                if nc <= 3:
                    break

    # Final local search on best result
    if best_result is not None and time.time() < deadline - 2.0:
        refined, final_v, final_c = local_search_refine(best_result, deadline - 1.0)
        if final_v == 0 and final_c < best_num_colors:
            best_result = refined
            best_num_colors = final_c

    # Fallback if nothing found
    if best_result is None:
        # Greedy construction
        colors = [-1] * n
        for x in range(n):
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] == -1:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
            if colors[x] == -1:
                colors[x] = x

        v = count_violations(colors)
        nc = len(set(colors))
        if v == 0:
            best_result = colors
            best_num_colors = nc

    # Final fallback
    if best_result is None:
        return list(range(n))

    return best_result
# EVOLVE-BLOCK-END
