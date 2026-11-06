import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute APs
    # aps is a list of tuples (i, j, k, l)
    # index_to_aps[i] is a list of indices in aps where i appears
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            index_to_aps[a].append(ap_idx)
            index_to_aps[b].append(ap_idx)
            index_to_aps[c].append(ap_idx)
            index_to_aps[e].append(ap_idx)

    def is_bad_ap(ap, colors):
        c1 = colors[ap[0]]
        c2 = colors[ap[1]]
        c3 = colors[ap[2]]
        c4 = colors[ap[3]]
        if len({c1, c2, c3, c4}) < 3:
            return True
        return False

    def calculate_penalty(colors):
        penalty = 0
        for ap in aps:
            if is_bad_ap(ap, colors):
                penalty += 1
        return penalty

    def greedy_construct(random_seed=0):
        random.seed(random_seed)
        colors = [0] * N
        # To allow for randomization in greedy, we can shuffle the order of colors we try
        # or just use the smallest valid color.
        # Smallest valid color is deterministic.
        # To get variety, we can randomly permute the color labels 0..127
        # and try them in that order.
        
        color_order = list(range(128))
        random.shuffle(color_order)
        
        for x in range(N):
            valid_color = None
            for c in color_order:
                # Check if c is valid at x
                ok = True
                for ap_idx in index_to_aps[x]:
                    ap = aps[ap_idx]
                    # Temporarily set colors[x] = c
                    old_c = colors[x]
                    colors[x] = c
                    if is_bad_ap(ap, colors):
                        colors[x] = old_c
                        ok = False
                        break
                if ok:
                    valid_color = c
                    break
            
            if valid_color is not None:
                colors[x] = valid_color
            else:
                # Should not happen
                colors[x] = 0
                
        # Remap to 0..K-1
        used = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(used)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    def try_reduce_colors(colors, time_limit):
        best_colors = list(colors)
        best_count = len(set(best_colors))
        
        # Precompute current AP badness
        ap_bad = [is_bad_ap(ap, best_colors) for ap in aps]
        current_penalty = sum(ap_bad)
        
        while time.time() < time_limit and best_count > 1:
            used_colors = sorted(set(best_colors))
            if len(used_colors) <= 1:
                break
            
            # Try to remove the color with fewest instances
            counts = {c: best_colors.count(c) for c in used_colors}
            target_color = min(counts, key=counts.get)
            positions = [i for i, c in enumerate(best_colors) if c == target_color]
            random.shuffle(positions)
            
            improved = False
            for pos in positions:
                old_c = best_colors[pos]
                # Try to reassign to any other used color
                for new_c in used_colors:
                    if new_c == old_c:
                        continue
                    
                    # Check if move is valid
                    valid = True
                    delta_penalty = 0
                    for ap_idx in index_to_aps[pos]:
                        was_bad = ap_bad[ap_idx]
                        # Temporarily set
                        best_colors[pos] = new_c
                        is_now_bad = is_bad_ap(aps[ap_idx], best_colors)
                        # Revert
                        best_colors[pos] = old_c
                        
                        if was_bad and not is_now_bad:
                            delta_penalty -= 1
                        elif not was_bad and is_now_bad:
                            delta_penalty += 1
                            valid = False
                            break
                    
                    if valid and delta_penalty == 0:
                        # Apply move
                        for ap_idx in index_to_aps[pos]:
                            was_bad = ap_bad[ap_idx]
                            best_colors[pos] = new_c
                            is_now_bad = is_bad_ap(aps[ap_idx], best_colors)
                            if was_bad and not is_now_bad:
                                current_penalty -= 1
                                ap_bad[ap_idx] = False
                            elif not was_bad and is_now_bad:
                                current_penalty += 1
                                ap_bad[ap_idx] = True
                        best_colors[pos] = new_c
                        improved = True
                        break # Move to next position in target color
            
            if not improved:
                # Try random color
                target_color = random.choice(used_colors)
                positions = [i for i, c in enumerate(best_colors) if c == target_color]
                random.shuffle(positions)
                improved = False
                for pos in positions:
                    old_c = best_colors[pos]
                    for new_c in used_colors:
                        if new_c == old_c: continue
                        valid = True
                        delta_penalty = 0
                        for ap_idx in index_to_aps[pos]:
                            was_bad = ap_bad[ap_idx]
                            best_colors[pos] = new_c
                            is_now_bad = is_bad_ap(aps[ap_idx], best_colors)
                            best_colors[pos] = old_c
                            if was_bad and not is_now_bad:
                                delta_penalty -= 1
                            elif not was_bad and is_now_bad:
                                delta_penalty += 1
                                valid = False
                                break
                        if valid and delta_penalty == 0:
                            for ap_idx in index_to_aps[pos]:
                                was_bad = ap_bad[ap_idx]
                                best_colors[pos] = new_c
                                is_now_bad = is_bad_ap(aps[ap_idx], best_colors)
                                if was_bad and not is_now_bad:
                                    current_penalty -= 1
                                    ap_bad[ap_idx] = False
                                elif not was_bad and is_now_bad:
                                    current_penalty += 1
                                    ap_bad[ap_idx] = True
                            best_colors[pos] = new_c
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            # Update counts
            new_used = set(best_colors)
            if len(new_used) < len(used_colors):
                used_colors = sorted(new_used)
                best_count = len(used_colors)

        return best_colors

    # Main
    best_solution = None
    best_count = float('inf')
    
    # Greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 10.0:
            break
        sol = greedy_construct(seed)
        count = len(set(sol))
        if count < best_count:
            best_count = count
            best_solution = sol
            
    # Reduce colors
    if best_solution:
        reduced = try_reduce_colors(best_solution, deadline - 2.0)
        count = len(set(reduced))
        if count < best_count:
            best_count = count
            best_solution = reduced

    if best_solution:
        return best_solution
        
    return greedy_construct(0)
