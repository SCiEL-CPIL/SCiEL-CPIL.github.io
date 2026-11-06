import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4_edges.append(e_indices)
        
    num_k4s = len(k4_edges)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initial colors: (u + v) % 17
    colors = [0] * num_edges
    for i, (u, v) in enumerate(edges):
        colors[i] = (u + v) % 17

    best_colors = colors[:]
    best_num_colors = max(colors) + 1

    # Helper to count distinct colors in a K4
    # We want to minimize the number of colors used globally.
    # The constraint is: every K4 must have >= 5 distinct colors.
    
    # Optimization: Precompute color counts per K4?
    # That's 1820 K4s * 17 colors = ~30k integers. Very feasible.
    # But updating is O(1) per edge change if we maintain counts.
    
    # Let's maintain:
    # k4_color_counts[k4_idx][c] = count of edges in k4_idx with color c
    # k4_distinct[k4_idx] = number of colors with count > 0
    
    k4_color_counts = [[0] * 20 for _ in range(num_k4s)]  # 20 to be safe, max color 17
    k4_distinct = [0] * num_k4s
    
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            c = colors[e_idx]
            if k4_color_counts[k4_idx][c] == 0:
                k4_distinct[k4_idx] += 1
            k4_color_counts[k4_idx][c] += 1

    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    # Re-initialize counts after compaction if needed
    def rebuild_counts(colors):
        nonlocal k4_color_counts, k4_distinct
        k4_color_counts = [[0] * 20 for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        for k4_idx, e_list in enumerate(k4_edges):
            for e_idx in e_list:
                c = colors[e_idx]
                if c >= 20:
                    # Expand if necessary, though unlikely
                    pass 
                if k4_color_counts[k4_idx][c] == 0:
                    k4_distinct[k4_idx] += 1
                k4_color_counts[k4_idx][c] += 1

    # Check if a K4 is valid
    def is_k4_valid(k4_idx):
        return k4_distinct[k4_idx] >= 5

    # Check if changing edge e_idx from old_c to new_c is valid
    # Returns True if all K4s containing e_idx remain valid
    def can_change_edge(e_idx, new_c):
        old_c = colors[e_idx]
        if old_c == new_c:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            # Current state
            old_distinct = k4_distinct[k4_idx]
            old_count_old = k4_color_counts[k4_idx][old_c]
            old_count_new = k4_color_counts[k4_idx][new_c]
            
            # Simulate change
            # old_c count decreases by 1
            # new_c count increases by 1
            
            new_distinct = old_distinct
            if old_count_old == 1:
                # old_c disappears
                new_distinct -= 1
            if old_count_new == 0:
                # new_c appears
                new_distinct += 1
            
            if new_distinct < 5:
                return False
                
        return True

    def apply_change(e_idx, new_c):
        old_c = colors[e_idx]
        if old_c == new_c:
            return
        colors[e_idx] = new_c
        for k4_idx in edge_to_k4s[e_idx]:
            # Update counts
            if k4_color_counts[k4_idx][old_c] == 1:
                k4_distinct[k4_idx] -= 1
            k4_color_counts[k4_idx][old_c] -= 1
            
            if k4_color_counts[k4_idx][new_c] == 0:
                k4_distinct[k4_idx] += 1
            k4_color_counts[k4_idx][new_c] += 1

    # Local Search Loop
    # We want to minimize the number of unique colors used.
    # Strategy: 
    # 1. Try to lower colors of edges to reuse existing low colors.
    # 2. If stuck, kick by changing a random edge to a random valid color (possibly higher).
    # 3. Periodically compact and check for improvement.
    
    iter_count = 0
    no_improve_count = 0
    max_color_bound = best_num_colors + 5
    
    while time.time() < deadline:
        # Try to improve
        improved = False
        
        # Iterate through edges in random order? Or sequential?
        # Random is better for avoiding local minima patterns.
        # But we can also do a pass.
        
        for _ in range(2000):
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            # Try to lower
            # Find the minimum valid color < current_c
            # We can try 0, 1, ... up to current_c - 1
            # Optimization: check if current_c is the only color in some K4s?
            # No, we just check validity.
            
            min_valid = -1
            # Try from 0 upwards
            for c in range(0, current_c):
                if can_change_edge(e_idx, c):
                    min_valid = c
                    break
            
            if min_valid != -1:
                apply_change(e_idx, min_valid)
                improved = True
        
        if improved:
            no_improve_count = 0
            # Check if we have a new best
            # We can approximate by checking if max(colors) < best_num_colors
            # If so, compact and update
            # But max(colors) might be high even if unique count is low.
            # Let's just compact periodically or when max drops significantly.
            # Actually, if we lower colors, max(colors) can only decrease or stay same.
            # If max(colors) < best_num_colors, it's a strong signal.
            # But what if max is 10, best is 9, but unique count is 8?
            # We should compact if we suspect improvement.
            # Let's compact every 1000 iterations or if max drops.
            
            if max(colors) < best_num_colors:
                compacted = compact_colors(colors)
                num_unique = max(compacted) + 1
                if num_unique < best_num_colors:
                    best_num_colors = num_unique
                    best_colors = compacted
                    colors = compacted
                    rebuild_counts(colors)
                    max_color_bound = best_num_colors + 5
        else:
            no_improve_count += 1
            
            # Kick: Try to change a random edge to a random valid color
            # This might increase the color count, but allows escape from local minima
            e_idx = random.randint(0, num_edges - 1)
            current_c = colors[e_idx]
            
            # Try a few random candidates
            found_kick = False
            # Range: 0 to max_color_bound
            for _ in range(20):
                new_c = random.randint(0, max_color_bound)
                if new_c == current_c:
                    continue
                if can_change_edge(e_idx, new_c):
                    apply_change(e_idx, new_c)
                    found_kick = True
                    break
            
            if not found_kick:
                # If we can't even kick, maybe we are in a very tight spot.
                # Try a more aggressive kick: allow larger colors?
                # Or just continue.
                pass

        iter_count += 1
        if iter_count % 1000 == 0:
            # Periodic compaction of current state to tidy up
            compacted = compact_colors(colors)
            num_unique = max(compacted) + 1
            if num_unique < best_num_colors:
                best_num_colors = num_unique
                best_colors = compacted
                max_color_bound = best_num_colors + 5
            # Always use compacted state for search to keep color space small
            colors = compacted
            rebuild_counts(colors)

    # Final compact best
    final_compact = compact_colors(best_colors)
    return final_compact
