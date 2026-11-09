import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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
            
        # Precompute conflict matrix as bitsets for speed
        # conflict[i] is a set of indices j that conflict with i
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
                    # Distance 2?
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
        
        # Initial coloring: Greedy First Fit
        colors = [0] * m
        max_color = 0
        for i in range(m):
            # Find smallest available color
            # A color c is invalid if any edge with color c conflicts with edge i
            taken = set()
            for j in range(i):
                if j in conflict_sets[i]:
                    taken.add(colors[j])
            
            c = 0
            while c in taken:
                c += 1
            colors[i] = c
            max_color = max(max_color, c)
        
        # State for CDLS
        # edge_conflicts[i] = number of conflicts for edge i
        # color_conflicts[c] = number of conflicts involving color c
        # total_conflicts = total number of conflicting pairs
        
        def compute_initial_state(cols):
            ec = [0] * m
            cc = defaultdict(int)
            total = 0
            for i in range(m):
                ci = cols[i]
                for j in conflict_sets[i]:
                    if j > i and cols[j] == ci:
                        ec[i] += 1
                        ec[j] += 1
                        cc[ci] += 1
                        total += 1
            return ec, cc, total

        edge_conflicts, color_conflicts, total_conflicts = compute_initial_state(colors)
        
        best_total_conflicts = total_conflicts
        best_max_color = max_color
        best_colors = colors[:]
        
        rng = random.Random(seed + 54321)
        
        # Phase 1: Eliminate conflicts
        # Phase 2: Minimize colors
        
        while time.time() < time_limit:
            # If we have 0 conflicts, try to reduce colors
            if total_conflicts == 0:
                # Try to recolor an edge to a lower color or merge colors
                # Heuristic: pick an edge with high degree in its color class or random
                i = rng.randrange(m)
                old_c = colors[i]
                
                # Try to move to a lower color if possible
                # Or try to merge color old_c into another color if old_c is the highest
                # Simple approach: try random recoloring to see if we can reduce max color
                # If max_color > 0, try to remove max_color usage
                
                # If we are at a local minimum for conflicts=0, do random perturbation
                # to escape local minima in color space
                if rng.random() < 0.1:
                    # Perturbation: pick a random edge and change its color to a random valid one
                    # This might increase conflicts temporarily
                    new_c = rng.randint(0, max_color)
                    if new_c != old_c:
                        # Calculate delta
                        # Remove old conflicts
                        for j in conflict_sets[i]:
                            if colors[j] == old_c:
                                edge_conflicts[i] -= 1
                                edge_conflicts[j] -= 1
                                color_conflicts[old_c] -= 1
                                total_conflicts -= 1
                                
                        colors[i] = new_c
                        max_color = max(max_color, new_c)
                        
                        # Add new conflicts
                        for j in conflict_sets[i]:
                            if colors[j] == new_c:
                                edge_conflicts[i] += 1
                                edge_conflicts[j] += 1
                                color_conflicts[new_c] += 1
                                total_conflicts += 1
                        
                        # Check if we improved color count (if old_c was max and no longer used)
                        # This is a heuristic, we accept if conflicts are low
                        if total_conflicts > 10: # Too many conflicts, revert
                            # Revert
                            for j in conflict_sets[i]:
                                if colors[j] == new_c:
                                    edge_conflicts[i] -= 1
                                    edge_conflicts[j] -= 1
                                    color_conflicts[new_c] -= 1
                                    total_conflicts -= 1
                            colors[i] = old_c
                            # Recalc max_color is expensive, keep old max
                        else:
                            # Accept
                            pass
                else:
                    # Try to reduce max_color
                    if max_color > 0:
                        # Find an edge with color max_color
                        # If we can recolor it to a lower color without creating conflicts, do it
                        # If not, try to find a valid color < max_color
                        candidates = [i for i in range(m) if colors[i] == max_color]
                        if candidates:
                            i = rng.choice(candidates)
                            old_c = colors[i]
                            # Try colors 0 to max_color-1
                            for new_c in range(max_color):
                                # Check if new_c is valid for i
                                valid = True
                                for j in conflict_sets[i]:
                                    if colors[j] == new_c:
                                        valid = False
                                        break
                                if valid:
                                    # Apply move
                                    for j in conflict_sets[i]:
                                        if colors[j] == old_c:
                                            edge_conflicts[i] -= 1
                                            edge_conflicts[j] -= 1
                                            color_conflicts[old_c] -= 1
                                            total_conflicts -= 1
                                    
                                    colors[i] = new_c
                                    
                                    for j in conflict_sets[i]:
                                        if colors[j] == new_c:
                                            edge_conflicts[i] += 1
                                            edge_conflicts[j] += 1
                                            color_conflicts[new_c] += 1
                                            total_conflicts += 1
                                    
                                    # Check if max_color is still used
                                    if color_conflicts[old_c] == 0:
                                        max_color -= 1
                                        # Update best
                                        if max_color < best_max_color:
                                            best_max_color = max_color
                                            best_colors = colors[:]
                                            best_total_conflicts = 0 # Implicitly 0 if we are in this branch
                                    break
            else:
                # Phase 1: Resolve conflicts
                # Pick an edge involved in a conflict
                # Weighted random selection: pick i with probability proportional to edge_conflicts[i]
                # For speed, just pick random i, if it has no conflicts, skip or pick another
                # Better: maintain a list of conflicted edges? Or just random try.
                
                # Random try up to 10 times
                found = False
                for _ in range(10):
                    i = rng.randrange(m)
                    if edge_conflicts[i] > 0:
                        found = True
                        break
                
                if not found:
                    # Should not happen if total_conflicts > 0, but just in case
                    continue

                old_c = colors[i]
                
                # Strategy: Move edge i to a color that reduces conflicts
                # Candidates:
                # 1. A color c that is not used by any neighbor of i (conflict-free)
                # 2. A color c that is used by fewer neighbors (conflict-reducing)
                
                # Count conflicts per color for edge i
                # conf_count[c] = number of neighbors j of i such that colors[j] == c
                conf_counts = defaultdict(int)
                for j in conflict_sets[i]:
                    conf_counts[colors[j]] += 1
                
                # Current conflicts for i: edge_conflicts[i]
                # If we move to c, new conflicts for i = conf_counts[c]
                # Delta for total conflicts:
                # - Remove conflicts with old_c: edge_conflicts[i] (since all old conflicts are with old_c? No, only those with same color)
                # Wait, edge_conflicts[i] counts ALL conflicts.
                # When i has color old_c, its conflicts are exactly the neighbors j where colors[j] == old_c.
                # So removing old color removes edge_conflicts[i] conflicts.
                # Adding new color c adds conf_counts[c] conflicts.
                # Delta = conf_counts[c] - edge_conflicts[i]
                
                # We want Delta < 0 (reduction) or Delta == 0 (neutral, helps color minimization)
                
                # Find best c
                best_c = -1
                best_delta = 0 # We accept if delta < 0. If no delta < 0, maybe accept delta == 0 if it reduces color count?
                
                # Try colors 0 to max_color + 1 (allow new color)
                # Limit search to small range for speed
                search_range = range(max_color + 2)
                
                # Randomize order to avoid bias
                candidates = list(search_range)
                rng.shuffle(candidates)
                
                # First pass: look for Delta < 0
                for c in candidates:
                    if c == old_c:
                        continue
                    delta = conf_counts.get(c, 0) - edge_conflicts[i]
                    if delta < 0:
                        best_c = c
                        best_delta = delta
                        break # Take first improvement (greedy)
                        
                # If no improvement, look for Delta == 0 that might help color reduction
                if best_c == -1:
                    for c in candidates:
                        if c == old_c:
                            continue
                        delta = conf_counts.get(c, 0) - edge_conflicts[i]
                        if delta == 0:
                            # Check if this reduces max color
                            if c < max_color and color_conflicts.get(max_color, 0) == 1 and colors[i] == max_color:
                                # This move removes the last use of max_color
                                best_c = c
                                best_delta = 0
                                break
                            # Or if c > max_color, it increases, so skip
                            # If c < old_c and old_c is max, and we are removing last use?
                            # Actually, if delta == 0, conflict count doesn't change.
                            # We prefer lower colors.
                            if c < old_c:
                                best_c = c
                                best_delta = 0
                                break

                if best_c == -1:
                    # No improving or neutral move found.
                    # Do a random perturbation to escape local minimum
                    # Pick a random color
                    new_c = rng.randint(0, max_color + 1)
                    if new_c == old_c:
                        continue
                    delta = conf_counts.get(new_c, 0) - edge_conflicts[i]
                    # Accept with probability based on delta (simulated annealing style)
                    # T is not explicitly managed, use a fixed small probability for worse moves
                    if delta > 0:
                        if rng.random() < 0.1: # 10% chance to accept worse move
                            pass
                        else:
                            continue
                    best_c = new_c
                    best_delta = delta

                # Apply move best_c
                # Update data structures
                # 1. Remove old color old_c
                for j in conflict_sets[i]:
                    if colors[j] == old_c:
                        edge_conflicts[i] -= 1
                        edge_conflicts[j] -= 1
                        color_conflicts[old_c] -= 1
                        total_conflicts -= 1
                        
                colors[i] = best_c
                max_color = max(max_color, best_c)
                
                # 2. Add new color best_c
                for j in conflict_sets[i]:
                    if colors[j] == best_c:
                        edge_conflicts[i] += 1
                        edge_conflicts[j] += 1
                        color_conflicts[best_c] += 1
                        total_conflicts += 1
                        
                # Update best
                if total_conflicts == 0:
                    if max_color < best_max_color:
                        best_max_color = max_color
                        best_colors = colors[:]
                        best_total_conflicts = 0
                elif total_conflicts < best_total_conflicts:
                    best_total_conflicts = total_conflicts
                    # Only update best colors if we are conflict-free? 
                    # The task requires valid coloring. If best has conflicts, it's invalid.
                    # We must return a valid coloring.
                    # So we should only track best_colors when total_conflicts == 0.
                    # If we never reach 0 conflicts, we return the last valid one?
                    # The greedy start is valid. So best_colors is always valid.
                    # We only update best_colors when total_conflicts == 0.

        # Final cleanup: ensure we return a valid coloring
        # If best_colors is from a state with 0 conflicts, it is valid.
        # If we never found 0 conflicts (unlikely with greedy start), best_colors is the initial greedy one.
        
        # Renumber colors to 0..k-1
        color_map = {}
        next_c = 0
        final_colors = []
        for c in best_colors:
            if c not in color_map:
                color_map[c] = next_c
                next_c += 1
            final_colors.append(color_map[c])
        
        return final_colors

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
