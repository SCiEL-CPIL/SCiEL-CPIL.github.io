import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

        # Precompute conflict sets
        # Two edges conflict if they share a vertex or are connected by an edge (induced matching condition)
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

        # Convert conflict sets to sorted lists for faster iteration
        conflict_lists = [sorted(cs) for cs in conflict_sets]

        # Heuristic for finding a large Independent Set (Induced Matching) in the conflict graph
        # We use a greedy approach with a twist: pick the vertex with minimum degree in the current subgraph
        # This is known to perform well on sparse graphs.
        
        def find_large_induced_matching(available_edges):
            if not available_edges:
                return []
            
            # available_edges is a set of edge indices
            # We want to find a large independent set in the induced subgraph of conflict_graph[available_edges]
            
            # Greedy: repeatedly pick the vertex with the smallest degree in the remaining subgraph
            # This helps "unblock" other vertices.
            remaining = set(available_edges)
            selected = []
            
            # To make it faster, we can use a priority queue or just scan.
            # Given m <= 160, scanning is fine.
            
            while remaining:
                # Find vertex with min degree in remaining
                min_deg = float('inf')
                best_v = None
                for v in remaining:
                    deg = 0
                    for u in conflict_lists[v]:
                        if u in remaining:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                        if deg == 0:
                            break # Can't get better than 0
                
                if best_v is None:
                    break
                    
                selected.append(best_v)
                remaining.remove(best_v)
                # Remove all neighbors of best_v from remaining
                for u in conflict_lists[best_v]:
                    if u in remaining:
                        remaining.remove(u)
            
            return selected

        # Better Heuristic: Local Search / Hill Climbing on the Set Cover
        # We start with a greedy coloring and then try to swap colors to reduce the total count.
        
        def greedy_coloring():
            colors = [-1] * m
            uncolored = set(range(m))
            # Order edges by degree in conflict graph (descending) to handle constrained ones first?
            # Or ascending? Let's try descending degree first (most constrained).
            order = sorted(range(m), key=lambda x: len(conflict_lists[x]), reverse=True)
            
            for i in order:
                if i not in uncolored:
                    continue
                taken = set()
                for j in conflict_lists[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        # Initial Solution
        best_colors = greedy_coloring()
        best_num_colors = max(best_colors) + 1 if best_colors else 0

        # Improvement Phase: Try to remove the highest color class
        # Strategy: 
        # 1. Identify edges with the highest color C_max.
        # 2. Try to recolor these edges to lower colors using local search (swaps).
        # 3. If successful, C_max is removed, reducing total colors.
        
        # We will use a simple local search: 
        # Pick an edge e with color C. Try to change its color to c < C.
        # If c is taken by a neighbor, try to swap colors between e and a neighbor?
        # Actually, a simpler approach: 
        # For each edge e with color C_max, try to assign it a color < C_max.
        # If blocked, try to "move" the blocking edges.
        
        # Let's implement a specific "Color Removal" algorithm.
        # Reference: Algorithm for edge coloring / graph coloring improvement.
        
        def try_remove_color(colors, C_max):
            # Returns new colors if C_max is removed, else None
            # We make a copy
            new_colors = colors[:]
            
            # Edges with color C_max
            edges_to_recolor = [i for i in range(m) if new_colors[i] == C_max]
            
            # We will try to recolor each of these edges to a color < C_max.
            # We use a recursive backtracking with limited depth or just a greedy retry.
            # Given the small size, we can try a simple DFS for each edge.
            
            # However, doing this for all edges at once is complex.
            # Let's try a simpler heuristic: 
            # Randomly permute the order of edges_to_recolor.
            # For each edge, try to find a valid color < C_max.
            # If it fails, try to swap with a neighbor?
            
            # Actually, a known technique is "Kempe chains" or simple swaps.
            # Let's try a basic local search:
            # 1. Pick an edge e with color C_max.
            # 2. Try to assign it a color c < C_max.
            # 3. If c is used by neighbor n, try to swap colors of e and n? No, n has color c.
            #    We need to move n's color.
            
            # Let's use a simpler, robust method:
            # Iteratively try to recolor edges of the highest color.
            # If stuck, try a random restart of the greedy phase with a different seed/order?
            # Or use the "Largest First" heuristic for set cover directly.
            
            # Alternative Strategy: Greedy Set Cover with Max Induced Matching
            # This is the "different design family" requested.
            # Instead of coloring edges one by one, we build color classes one by one.
            
            return None # Placeholder, we will implement the set cover approach below

        # --- Structural Change: Greedy Set Cover with Max Induced Matching ---
        
        def greedy_set_cover():
            # Universe: all edges
            # Sets: all possible induced matchings? No, we generate them greedily.
            # Algorithm:
            # 1. available = set(all edges)
            # 2. colors = []
            # 3. while available:
            #    a. Find a "good" induced matching M from available.
            #       Heuristic: Maximize size? Or maximize "removal" of high-degree nodes?
            #       Let's try to find a large independent set in the conflict graph induced by available.
            #    b. Assign current color to all edges in M.
            #    c. Remove M from available.
            #    d. Append color to list.
            # 4. Return colors.
            
            available = set(range(m))
            colors = [-1] * m
            current_color = 0
            
            while available:
                # Find a large induced matching in the subgraph induced by 'available'
                # We use the min-degree greedy heuristic for Independent Set.
                
                # To improve quality, we can try multiple random orderings or use a better IS heuristic.
                # Let's use a "Weighted" greedy where we prefer edges that have many neighbors in 'available'
                # (i.e., picking them removes many conflicts, which is good for set cover).
                # Wait, standard greedy for Set Cover picks the set that covers the most uncovered elements.
                # Here, the "set" is an induced matching. The "elements" are edges.
                # An induced matching covers its own edges.
                # So we want the induced matching that covers the most edges? 
                # That's just the largest induced matching.
                
                # However, picking the largest induced matching might not be optimal for the *next* steps.
                # But it's a solid baseline.
                
                # Heuristic for Large Independent Set in Conflict Graph (induced by available):
                # 1. Calculate degree of each vertex in the induced subgraph.
                # 2. Pick vertex with min degree? Or max degree?
                #    - Min degree: leaves are safe, might leave large clusters for later.
                #    - Max degree: removes many conflicts, might be better for "set cover" efficiency?
                #    Let's try Min Degree first (standard for IS).
                
                remaining = set(available)
                selected = []
                
                # We'll do a simple greedy IS selection.
                # To make it deterministic but varied, we can break ties by index.
                
                while remaining:
                    # Find vertex with min degree in remaining
                    min_deg = float('inf')
                    best_v = None
                    for v in remaining:
                        deg = 0
                        for u in conflict_lists[v]:
                            if u in remaining:
                                deg += 1
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                            if deg == 0:
                                break
                    
                    if best_v is None:
                        break
                        
                    selected.append(best_v)
                    remaining.remove(best_v)
                    for u in conflict_lists[best_v]:
                        if u in remaining:
                            remaining.remove(u)
                
                if not selected:
                    # Fallback: just pick one edge
                    selected = [next(iter(available))]
                
                for e in selected:
                    colors[e] = current_color
                
                available -= set(selected)
                current_color += 1
                
                if time.time() > deadline - 5.0:
                    break

            # If we ran out of time, we might have uncolored edges.
            # We must ensure all edges are colored.
            # The loop above should finish quickly for m<=160.
            # But if it didn't, we need a fallback for remaining edges.
            if available:
                # Greedy color the rest with the next colors
                c = current_color
                for e in sorted(available):
                    colors[e] = c
                    c += 1
            
            return colors

        # Let's run the Greedy Set Cover approach
        set_cover_colors = greedy_set_cover()
        set_cover_num = max(set_cover_colors) + 1 if set_cover_colors else 0

        # Compare with initial greedy
        if set_cover_num < best_num_colors:
            best_colors = set_cover_colors
            best_num_colors = set_cover_num

        # Further Improvement: Local Search on the Set Cover Result
        # Try to reduce the number of colors by merging or recoloring.
        # We can try to remove the highest color class as before.
        
        def improve_by_recoloring(colors, num_colors, time_limit):
            # Try to remove color num_colors - 1
            # We will use a simple backtracking search with a time limit.
            # This is essentially trying to recolor the edges of the last class into lower classes.
            
            if num_colors <= 1:
                return colors, num_colors
            
            target_color = num_colors - 1
            edges_in_target = [i for i in range(m) if colors[i] == target_color]
            
            if not edges_in_target:
                return colors, num_colors - 1
            
            # We want to assign each edge in edges_in_target a color in [0, target_color-1]
            # such that no conflicts arise.
            
            # This is a subgraph coloring problem.
            # We can use a simple DFS with pruning.
            
            # To keep it fast, we limit the depth or use a greedy assignment with backtracking only on conflicts.
            
            # Let's try a greedy assignment first.
            new_colors = colors[:]
            
            # Order edges in target by degree in the subgraph induced by edges_in_target?
            # Or just arbitrary.
            
            assigned = {}
            
            def dfs(idx):
                if time.time() > time_limit:
                    return False
                if idx == len(edges_in_target):
                    return True
                
                e = edges_in_target[idx]
                
                # Try colors 0 to target_color - 1
                # Shuffle order? No, keep deterministic.
                for c in range(target_color):
                    # Check if c is valid for e
                    valid = True
                    for neighbor in conflict_lists[e]:
                        if new_colors[neighbor] == c:
                            valid = False
                            break
                        # Also check if neighbor is in edges_in_target and already assigned?
                        # No, we are updating new_colors.
                        # Wait, if neighbor is in edges_in_target and assigned in this step, new_colors[neighbor] is updated.
                        # If neighbor is NOT in edges_in_target, new_colors[neighbor] is from the previous state.
                    
                    if valid:
                        old_color = new_colors[e]
                        new_colors[e] = c
                        if dfs(idx + 1):
                            return True
                        new_colors[e] = old_color
                
                return False
            
            # To improve success rate, we can try different orderings of edges_in_target.
            # Let's try ordering by degree in the conflict graph (descending).
            edges_in_target.sort(key=lambda x: len(conflict_lists[x]), reverse=True)
            
            if dfs(0):
                return new_colors, target_color
            
            # If DFS failed, try a different heuristic?
            # For now, return original.
            return colors, num_colors

        # Apply improvement iteratively
        improved_colors = best_colors[:]
        improved_num = best_num_colors
        
        # We have time left?
        # Let's allocate remaining time to this improvement phase.
        # The set cover is fast. The improvement might be slower.
        
        # Let's try to remove colors one by one.
        # We need a time budget for the backtracking.
        # Let's give it 5 seconds per color removal attempt.
        
        current_colors = improved_colors
        current_num = improved_num
        
        while current_num > 1:
            if time.time() > deadline - 2.0:
                break
            
            # Try to remove the highest color
            new_colors, new_num = improve_by_recoloring(current_colors, current_num, deadline - 2.0)
            
            if new_num < current_num:
                current_colors = new_colors
                current_num = new_num
            else:
                # If we couldn't remove the highest color, maybe we can't remove any?
                # Not necessarily, but it's a good sign to stop.
                # We could try to remove a different color, but that's complex.
                break
        
        if current_num < best_num_colors:
            best_colors = current_colors
            best_num_colors = current_num

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
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
        
        # Allocate time
        # The set cover is very fast. The improvement is the main cost.
        # Let's give 40s to each instance, but cap the improvement part.
        time_limit = 40.0
        if len(results) == 2:
             time_limit = remaining - 2.0
        
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
        
        if time.time() > deadline - 2.0:
             # If we are running out of time, ensure we return valid results for remaining
             # by using the fallback greedy for any unsolved instances.
             pass

    return results
