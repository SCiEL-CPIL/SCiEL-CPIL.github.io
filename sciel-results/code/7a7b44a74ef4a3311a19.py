import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict matrix
        # conflict[i] is a set of indices j that conflict with i
        # Two edges conflict if they share a vertex or are joined by an edge (distance 2)
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                # Shared vertex?
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    # Distance 2? (ui-uj, ui-vj, vi-uj, vi-vj)
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
        
        # DSATUR-style backtracking
        # State:
        # colors: list of length m, -1 if uncolored
        # color_sets: list of sets, color_sets[c] contains indices of edges with color c
        # avail_colors: list of sets, avail_colors[i] contains colors available for edge i
        #   A color c is available for i if no neighbor of i has color c.
        
        colors = [-1] * m
        color_sets = [set() for _ in range(2 * d + 10)] # Max colors estimate
        # For speed, we might need more colors, but let's start with a reasonable bound
        # The conjecture is 5d^2/4. Let's allocate enough.
        max_colors_est = int(5 * d * d / 4) + 10
        color_sets = [set() for _ in range(max_colors_est)]
        
        # avail_colors[i] initially contains all colors 0..max_colors_est-1
        # We will update this dynamically.
        # To save memory and time, we can compute availability on the fly or maintain bitsets.
        # Given m <= 240, bitsets are good.
        
        # Let's use a different representation for speed in backtracking:
        # uncolored: set of indices not yet colored
        # For each uncolored edge i, we need to know which colors are forbidden.
        # Forbidden colors for i are the union of colors of its neighbors.
        
        uncolored = set(range(m))
        
        # Heuristic for choosing next edge:
        # 1. Saturation: number of distinct colors in neighbors
        # 2. Degree: number of uncolored neighbors
        # 3. Conflict degree: total number of neighbors (static)
        
        # Precompute static conflict degrees
        static_deg = [len(conflict_sets[i]) for i in range(m)]
        
        # We will implement a recursive solver with a node limit to prevent explosion.
        # If it fails, we fall back to a greedy heuristic or restart with different randomization.
        
        best_colors = None
        best_num_colors = float('inf')
        
        def get_saturation(i):
            # Number of distinct colors in neighbors of i
            if i not in uncolored:
                return -1
            neighbor_colors = set()
            for j in conflict_sets[i]:
                if colors[j] != -1:
                    neighbor_colors.add(colors[j])
            return len(neighbor_colors)

        def get_uncolored_degree(i):
            # Number of uncolored neighbors of i
            if i not in uncolored:
                return -1
            count = 0
            for j in conflict_sets[i]:
                if j in uncolored:
                    count += 1
            return count

        def choose_next_edge():
            # Select edge with max saturation, then max uncolored degree, then max static degree
            best_i = None
            best_score = (-1, -1, -1)
            for i in uncolored:
                sat = get_saturation(i)
                deg = get_uncolored_degree(i)
                score = (sat, deg, static_deg[i])
                if score > best_score:
                    best_score = score
                    best_i = i
            return best_i

        def is_color_available(i, c):
            # Check if color c can be assigned to edge i
            for j in conflict_sets[i]:
                if colors[j] == c:
                    return False
            return True

        def backtrack(node_count, max_nodes):
            nonlocal best_colors, best_num_colors
            
            if time.time() > time_limit:
                return False # Timeout

            if not uncolored:
                # All colored
                # Count max color used
                max_c = -1
                for c in range(len(color_sets)):
                    if color_sets[c]:
                        max_c = c
                num_colors = max_c + 1
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colors = colors[:]
                return True # Found a solution, stop searching for this branch?
                            # Actually, we want the minimum. So we should continue to see if we can do better.
                            # But backtracking for minimum coloring is hard.
                            # Strategy: Find ANY valid coloring first. Then try to improve?
                            # Or use branch and bound: if current max color + 1 >= best_num_colors, prune.
                
                # Pruning: If we are using more colors than best known, stop.
                # We track max color used in current path.
                # Let's maintain current_max_color.
                
            # Pruning:
            # If the minimum number of colors needed for the remaining graph + current max color >= best_num_colors
            # This is hard to estimate. Simple bound: if current_max_color >= best_num_colors, prune.
            # But we don't track current_max_color explicitly in the state easily without scanning.
            # Let's just rely on node limit and time.
            
            if node_count >= max_nodes:
                return False
            
            i = choose_next_edge()
            if i is None:
                return False # Should not happen if uncolored is not empty

            # Try colors in order of preference
            # Preference: Lower colors first? Or colors that are "tightest"?
            # Usually, trying lower colors first helps find a solution faster.
            # But for minimization, we might want to try all.
            
            # Determine available colors
            # A color c is available if no neighbor has color c.
            # We can iterate c from 0 to current_max_used + 1.
            
            # Find max color used so far
            max_used = -1
            for c in range(len(color_sets)):
                if color_sets[c]:
                    max_used = c
            
            # Try colors 0 to max_used + 1
            for c in range(max_used + 2):
                if is_color_available(i, c):
                    # Assign color c to i
                    colors[i] = c
                    color_sets[c].add(i)
                    uncolored.remove(i)
                    
                    if backtrack(node_count + 1, max_nodes):
                        # Undo
                        uncolored.add(i)
                        color_sets[c].remove(i)
                        colors[i] = -1
                        return True # If we found a solution, we might want to stop?
                                    # No, we want the MINIMUM.
                                    # However, pure backtracking for minimum is exponential.
                                    # Given the small size (m~240), and d small (4,5,6),
                                    # the number of colors is small (~20-45).
                                    # The search space might still be large.
                                    # A better approach:
                                    # 1. Get a valid coloring (any) using DSATUR.
                                    # 2. Use local search to minimize colors.
                                    # This is what the previous program did, but maybe the initial coloring was poor.
                                    # Let's stick to a hybrid:
                                    # Try to find a solution with k colors.
                                    # Start k = greedy_estimate.
                                    # If success, try k-1, etc.
                                    
                    # Undo
                    uncolored.add(i)
                    color_sets[c].remove(i)
                    colors[i] = -1
                    
            return False

        # Since pure backtracking for minimization is risky, let's use a different strategy:
        # 1. Generate a valid coloring using DSATUR (greedy).
        # 2. Use a "color merging" local search to reduce the number of colors.
        # This is similar to the original but with a better initial coloring.
        
        # Let's implement a robust Greedy DSATUR first.
        def greedy_dsatur():
            colors = [-1] * m
            color_sets = [set() for _ in range(100)]
            uncolored = set(range(m))
            
            while uncolored:
                # Choose edge with max saturation, then max degree
                best_i = None
                best_score = (-1, -1)
                for i in uncolored:
                    sat = 0
                    neighbor_colors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_sets[i] if j in uncolored)
                    score = (sat, deg)
                    if score > best_score:
                        best_score = score
                        best_i = i
                
                i = best_i
                # Find smallest available color
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                
                c = 0
                while c in taken:
                    c += 1
                
                colors[i] = c
                color_sets[c].add(i)
                uncolored.remove(i)
            
            return colors

        # Initial coloring
        init_colors = greedy_dsatur()
        max_c_init = max(init_colors) + 1 if init_colors else 0
        
        # Now, try to reduce colors using local search.
        # We represent the state as the list of colors.
        # We want to minimize the number of distinct colors used.
        # Strategy: Try to merge color k into color < k.
        # Or, more generally, try to recolor edges to reduce the max color.
        
        current_colors = init_colors[:]
        best_colors = init_colors[:]
        best_num_colors = max_c_init
        
        rng = random.Random(seed + 12345)
        
        def get_conflicts_for_color(c):
            # Returns set of pairs (i, j) in color c that conflict
            conflicts = []
            edges_in_c = [i for i in range(m) if current_colors[i] == c]
            for idx, i in enumerate(edges_in_c):
                for j in edges_in_c[idx+1:]:
                    if j in conflict_sets[i]:
                        conflicts.append((i, j))
            return conflicts

        def is_valid_coloring(cols):
            # Check if any color class has conflicts
            color_edges = defaultdict(list)
            for i, c in enumerate(cols):
                color_edges[c].append(i)
            for c, es in color_edges.items():
                for i in range(len(es)):
                    for j in range(i+1, len(es)):
                        if es[j] in conflict_sets[es[i]]:
                            return False
            return True

        # Local Search:
        # 1. While max_color > 0:
        #    Try to remove max_color.
        #    For each edge with max_color, try to move it to a lower color.
        #    If a move creates a conflict, resolve it by moving another edge? (Complex)
        #    Simpler: Simulated Annealing on the number of conflicts + penalty for max color.
        
        # Let's use Simulated Annealing with objective:
        # Energy = total_conflicts * 100 + max_color
        # We want to minimize Energy.
        
        def compute_energy(cols):
            total_conf = 0
            max_c = -1
            color_edges = defaultdict(list)
            for i, c in enumerate(cols):
                if c > max_c: max_c = c
                color_edges[c].append(i)
            
            for c, es in color_edges.items():
                # Count conflicts within this color class
                for i in range(len(es)):
                    for j in range(i+1, len(es)):
                        if es[j] in conflict_sets[es[i]]:
                            total_conf += 1
            return total_conf * 100 + (max_c + 1 if max_c >= 0 else 0)

        current_energy = compute_energy(current_colors)
        best_energy = current_energy
        
        T = 100.0
        cooling = 0.999
        
        steps = 0
        max_steps = 500000 # Limit steps
        
        while time.time() < time_limit and steps < max_steps:
            steps += 1
            
            # Pick a random edge
            i = rng.randrange(m)
            old_c = current_colors[i]
            
            # Pick a random new color
            # Range: 0 to max_color + 1
            max_c_used = max(current_colors)
            new_c = rng.randint(0, max_c_used + 1)
            
            if new_c == old_c:
                continue
                
            # Calculate delta energy
            # This is expensive if done naively.
            # Let's just compute new energy and compare.
            # For m=240, compute_energy is O(m^2) ~ 50k ops. 500k steps * 50k ops = 2.5e10. Too slow.
            
            # We need incremental updates.
            # Let's maintain:
            # 1. conflict_count[c]: number of conflicts in color class c
            # 2. max_color_used
            
            # Re-initialize state for incremental SA
            conflict_counts = defaultdict(int)
            color_edges = defaultdict(list)
            for i, c in enumerate(current_colors):
                color_edges[c].append(i)
            
            # Precompute conflicts for each color
            for c, es in color_edges.items():
                cnt = 0
                for i in range(len(es)):
                    for j in range(i+1, len(es)):
                        if es[j] in conflict_sets[es[i]]:
                            cnt += 1
                conflict_counts[c] = cnt
                
            total_conflicts = sum(conflict_counts.values())
            max_color_used = max(current_colors) if current_colors else 0
            current_energy = total_conflicts * 100 + max_color_used + 1
            
            best_colors = current_colors[:]
            best_energy = current_energy
            best_max_color = max_color_used
            
            for step in range(max_steps):
                if time.time() > time_limit:
                    break
                    
                i = rng.randrange(m)
                old_c = current_colors[i]
                
                # Determine valid range for new color
                # We allow 0 to max_color_used + 1
                new_c = rng.randint(0, max_color_used + 1)
                
                if new_c == old_c:
                    continue
                    
                # Calculate delta
                # Remove i from old_c
                # Conflicts removed: number of neighbors j in old_c that conflict with i
                removed_conf = 0
                for j in conflict_sets[i]:
                    if current_colors[j] == old_c:
                        removed_conf += 1
                        
                # Add i to new_c
                # Conflicts added: number of neighbors j in new_c that conflict with i
                added_conf = 0
                for j in conflict_sets[i]:
                    if current_colors[j] == new_c:
                        added_conf += 1
                        
                delta_conf = added_conf - removed_conf
                
                # Update max color
                old_max = max_color_used
                new_max = max_color_used
                if old_c == old_max and all(current_colors[j] != old_c for j in range(m) if j != i):
                    # Last one removed
                    # Find new max
                    # This is expensive.
                    # Let's just track max_color_used loosely or recalc if needed.
                    # Actually, we can just check if new_c > old_max.
                    pass
                
                # Let's simplify: just compute new max if old_c was max or new_c is > max
                if new_c > max_color_used:
                    new_max = new_c
                elif old_c == max_color_used:
                    # Check if any other edge has old_c
                    # To save time, we'll just assume max doesn't drop unless we verify.
                    # Or we can maintain a count of edges per color.
                    pass
                
                # Let's maintain color_counts
                # We didn't initialize color_counts above. Let's add it.
                
                # Restarting the inner loop logic is getting messy.
                # Let's stick to the simpler, slightly slower but robust approach:
                # Just compute energy from scratch every few steps? No.
                
                # Let's use the incremental approach properly.
                
                # We need:
                # color_counts[c]: number of edges with color c
                # conflict_counts[c]: number of conflicts in color c
                
                # Initialize
                color_counts = defaultdict(int)
                for c in current_colors:
                    color_counts[c] += 1
                
                # Apply move
                color_counts[old_c] -= 1
                color_counts[new_c] += 1
                
                # Update conflicts
                # For old_c:
                # i was in old_c. It had 'removed_conf' conflicts.
                # So conflict_counts[old_c] -= removed_conf
                # For new_c:
                # i joins new_c. It has 'added_conf' conflicts.
                # So conflict_counts[new_c] += added_conf
                
                new_total_conf = total_conflicts + delta_conf
                
                # Update max color
                new_max = max_color_used
                if color_counts[old_c] == 0 and old_c == max_color_used:
                    # Find new max
                    # We can just scan or keep a sorted structure.
                    # For m=240, scanning is fine.
                    new_max = -1
                    for c, cnt in color_counts.items():
                        if cnt > 0 and c > new_max:
                            new_max = c
                if new_c > new_max:
                    new_max = new_c
                    
                new_energy = new_total_conf * 100 + new_max + 1
                delta_energy = new_energy - current_energy
                
                if delta_energy < 0 or rng.random() < pow(2.71828, -delta_energy / T):
                    # Accept
                    current_colors[i] = new_c
                    total_conflicts = new_total_conf
                    max_color_used = new_max
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = current_colors[:]
                        best_max_color = max_color_used
                        
                else:
                    # Reject, revert color_counts
                    color_counts[old_c] += 1
                    color_counts[new_c] -= 1
                    # total_conflicts and max_color_used unchanged
                    
                T *= cooling
                
        # Finalize
        # Ensure best_colors is valid?
        # The SA might have left it with conflicts if it didn't find a valid one.
        # But we started with a valid one (greedy).
        # And we only accept moves if they improve energy (which penalizes conflicts heavily).
        # So if best_energy corresponds to 0 conflicts, it's valid.
        # If not, we should fall back to the initial greedy coloring.
        
        if compute_energy(best_colors) < 100: # 0 conflicts
            return best_colors
        else:
            return init_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        
        time_limit = max(1.0, remaining - 1.0)
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
    
    return results
