import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
        
        # Precompute adjacency for the graph
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict matrix
        # Two edges i, j conflict if they share a vertex or are joined by an edge (distance 2)
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
        
        # Heuristic: Greedy Maximum Induced Matching Extraction
        # We repeatedly pick the "best" edge, assign it a new color, and remove it 
        # along with all its conflicting edges from the pool of available edges for that color.
        # Wait, that's not right. A color class is an induced matching.
        # So, for a specific color, we want a subset of edges that are pairwise non-conflicting.
        # This is the Maximum Independent Set problem on the conflict graph.
        # Greedy MIS heuristic:
        # 1. Start with all edges.
        # 2. Pick edge e with minimum degree in the *current* conflict graph (or some heuristic).
        # 3. Add e to the color class.
        # 4. Remove e and all its neighbors from the current conflict graph.
        # 5. Repeat until empty.
        # This gives one color class.
        # Then remove these edges from the main problem and repeat.
        
        # To improve: 
        # Instead of just min-degree, we can use a weighted heuristic.
        # Or, we can use backtracking to find a *better* MIS for the first few colors.
        
        # Let's implement a function that extracts a color class (induced matching)
        # from a set of available edges, using a greedy heuristic.
        
        def extract_color_class(available_edges_set, conflict_sets):
            # available_edges_set: set of indices
            if not available_edges_set:
                return []
            
            # Compute degrees within the available set
            # deg[e] = number of conflicts of e that are also in available_edges_set
            degs = {}
            for e in available_edges_set:
                degs[e] = len(conflict_sets[e] & available_edges_set)
            
            # Priority queue: (degree, edge)
            # We want to pick edges with low degree to keep the graph sparse?
            # Or high degree to eliminate many conflicts?
            # For MIS, usually picking high degree nodes first (and removing them) is bad.
            # Picking low degree nodes (leafs) is good for finding a large independent set?
            # Actually, the standard greedy MIS is: pick vertex, remove it and neighbors.
            # To maximize size, we often pick the vertex with the *smallest* degree first?
            # No, that's for coloring. For MIS, it's tricky.
            # Let's try picking the edge with the minimum number of remaining conflicts.
            
            heap = [(degs[e], e) for e in available_edges_set]
            heapq.heapify(heap)
            
            selected = []
            remaining = set(available_edges_set)
            
            while heap:
                deg, e = heapq.heappop(heap)
                if e not in remaining:
                    continue
                # Re-check degree if necessary? 
                # Since we only remove nodes, degrees only decrease.
                # If e is still in remaining, its current degree is <= deg.
                # Is it safe to just pick it?
                # Yes, if we maintain the invariant that heap contains (deg_at_insert, e).
                # When we remove neighbors, their degrees drop.
                # We don't update the heap entries. So we might pop an old entry.
                # We check `if e not in remaining`.
                # But what if e is in remaining, but its degree has dropped significantly?
                # It doesn't matter for validity. It might matter for optimality.
                # For a heuristic, this is fine.
                
                # Select e
                selected.append(e)
                remaining.remove(e)
                
                # Remove neighbors
                for nb in conflict_sets[e]:
                    if nb in remaining:
                        remaining.remove(nb)
                        # We don't need to update heap for removed nodes, 
                        # they will be skipped when popped.
                        
            return selected

        # The strategy:
        # 1. Start with all edges available.
        # 2. Extract a color class using the heuristic above.
        # 3. Assign this color to all edges in the class.
        # 4. Remove these edges from the available set.
        # 5. Repeat until all edges are colored.
        
        # This produces a valid coloring.
        # To minimize the number of colors, we want each color class to be as large as possible.
        # The greedy MIS heuristic is decent, but can be improved.
        
        # Improvement: 
        # Use a "lookahead" or "randomized restarts" for the MIS extraction.
        # Or, use a limited backtracking search for the first few color classes.
        
        # Let's implement a simple local search on the coloring to merge colors.
        # Actually, the "extract class" approach naturally minimizes colors if the classes are large.
        
        # Let's try a hybrid:
        # 1. Greedy extraction to get an initial coloring.
        # 2. Try to merge the highest color class into lower ones.
        
        def greedy_coloring():
            colors = [-1] * m
            available = set(range(m))
            current_color = 0
            
            while available:
                # Extract a class
                cls = extract_color_class(available, conflict_sets)
                for e in cls:
                    colors[e] = current_color
                    available.remove(e)
                current_color += 1
                
            return colors, current_color

        init_colors, init_num_colors = greedy_coloring()
        
        # Local Search: Color Merging
        # Try to remove the highest color (init_num_colors - 1).
        # For each edge in that color, try to move it to a lower color.
        # If moving creates a conflict, try to move the conflicting edge? (2-swap)
        
        current_colors = init_colors[:]
        current_num_colors = init_num_colors
        
        # We will try to reduce current_num_colors one by one.
        # Target: 0 conflicts, min colors.
        
        # Let's use a more robust local search:
        # Objective: Minimize (num_colors * 1000 + total_conflicts)
        # Moves: Recolor a single edge.
        
        # Precompute color lists for fast access
        color_lists = defaultdict(list)
        for i, c in enumerate(current_colors):
            color_lists[c].append(i)
            
        def get_conflicts_in_color(c):
            # Returns number of conflicts in color class c
            es = color_lists[c]
            count = 0
            for i in range(len(es)):
                for j in range(i+1, len(es)):
                    if es[j] in conflict_sets[es[i]]:
                        count += 1
            return count

        # Initial state: 0 conflicts (from greedy extraction)
        # We want to reduce num_colors.
        
        # Algorithm:
        # While num_colors > 1:
        #   Try to eliminate color num_colors - 1.
        #   For each edge e in that color:
        #      Try to move e to color 0..num_colors-2.
        #      If valid, move it.
        #      If invalid, check if a 2-swap can resolve it.
        #   If we managed to move all edges from the top color, num_colors--.
        #   Else, break (or try a different order).
        
        # This is deterministic and might get stuck.
        # Let's add randomization:
        # Shuffle the order of edges in the top color.
        # Shuffle the order of target colors.
        
        best_colors = current_colors[:]
        best_num_colors = current_num_colors
        
        rng = random.Random(seed + 999)
        
        for attempt in range(50): # Limited attempts
            if time.time() > time_limit:
                break
                
            # Try to reduce colors
            # We'll try to merge the top color into the rest.
            top_color = current_num_colors - 1
            edges_in_top = color_lists[top_color][:]
            rng.shuffle(edges_in_top)
            
            success = True
            for e in edges_in_top:
                if time.time() > time_limit:
                    success = False
                    break
                    
                # Try to move e to a lower color
                moved = False
                # Try colors in random order
                target_colors = list(range(top_color))
                rng.shuffle(target_colors)
                
                for tc in target_colors:
                    # Check if e can be moved to tc
                    # e conflicts with some edges in tc.
                    # If no conflicts, move it.
                    has_conflict = False
                    for nb in conflict_sets[e]:
                        if current_colors[nb] == tc:
                            has_conflict = True
                            break
                            
                    if not has_conflict:
                        # Move e
                        current_colors[e] = tc
                        color_lists[top_color].remove(e)
                        color_lists[tc].append(e)
                        moved = True
                        break
                        
                if not moved:
                    # Try 2-swap:
                    # Find a conflict partner p in some target color tc.
                    # If we move p to another color, e can move to tc.
                    # This is complex. Let's just fail this attempt for this edge.
                    # Or, try to find a swap.
                    
                    # Let's try a simple swap:
                    # Find a color tc where e has exactly 1 conflict p.
                    # If p can be moved to another color td (validly), do it.
                    swapped = False
                    for tc in target_colors:
                        conflicts_in_tc = []
                        for nb in conflict_sets[e]:
                            if current_colors[nb] == tc:
                                conflicts_in_tc.append(nb)
                                
                        if len(conflicts_in_tc) == 1:
                            p = conflicts_in_tc[0]
                            # Can p move to some other color td != tc, td != top_color?
                            # Actually p is in tc. We want to move p to td.
                            # Then e can move to tc.
                            # We need td to be valid for p.
                            possible_tds = list(range(top_color))
                            possible_tds.remove(tc)
                            rng.shuffle(possible_tds)
                            
                            for td in possible_tds:
                                # Check if p can move to td
                                p_conflict = False
                                for nb in conflict_sets[p]:
                                    if current_colors[nb] == td:
                                        p_conflict = True
                                        break
                                        
                                if not p_conflict:
                                    # Do the swap
                                    # Move p to td
                                    current_colors[p] = td
                                    color_lists[tc].remove(p)
                                    color_lists[td].append(p)
                                    
                                    # Move e to tc
                                    current_colors[e] = tc
                                    color_lists[top_color].remove(e)
                                    color_lists[tc].append(e)
                                    
                                    swapped = True
                                    break
                                    
                            if swapped:
                                moved = True
                                break
                                
                    if not moved:
                        success = False
                        break
                        
            if success:
                # We successfully merged the top color
                current_num_colors -= 1
                if current_num_colors < best_num_colors:
                    best_num_colors = current_num_colors
                    best_colors = current_colors[:]
                # Update color_lists to remove empty top color?
                # color_lists[top_color] should be empty now.
                # We can just continue. The next iteration will target new top_color.
            else:
                # Failed to merge. 
                # We could try a different heuristic or restart.
                # For now, break out of the reduction loop for this attempt.
                # But we might have partially modified current_colors?
                # In my implementation, I only commit moves if they are valid.
                # If I fail on an edge, I haven't moved it.
                # So current_colors is still valid, but num_colors didn't drop.
                # I should probably break the attempt loop or try a different random seed.
                pass
                
        # Final validation
        # Check if best_colors is valid
        def is_valid(cols):
            color_edges = defaultdict(list)
            for i, c in enumerate(cols):
                color_edges[c].append(i)
            for c, es in color_edges.items():
                for i in range(len(es)):
                    for j in range(i+1, len(es)):
                        if es[j] in conflict_sets[es[i]]:
                            return False
            return True

        if is_valid(best_colors):
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
