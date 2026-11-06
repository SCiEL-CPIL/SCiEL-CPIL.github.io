"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_and_edges(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        # Heuristic: Repeatedly find long cycles
        # We can use a simple DFS-based cycle finder or try to find 2-factors.
        # Given small graphs (n<=30), we can afford some search.
        # Let's try to find a cycle, remove its edges, repeat.
        # To get "few" pieces, we want "long" cycles.
        # A simple greedy approach: find any cycle, but try to make it long.
        # Better: Find a cycle, remove it. If no cycle, remove an edge.
        # To improve: Prioritize removing edges that are part of many cycles?
        # Or: Find a maximum length cycle in the remaining graph?
        
        # Let's try a randomized local search or a more structured extraction.
        # Strategy: 
        # 1. While edges exist:
        #    a. Find a cycle. If found, remove it. 
        #    b. To maximize length, we can try to find a long cycle via DFS with backtracking limited by depth.
        #    c. If no cycle, remove a single edge.
        
        # Let's implement a DFS that tries to find the longest simple cycle.
        # Since n is small (<=30), we can do this efficiently.
        
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find longest cycle in remaining_edges
            # Convert to adjacency
            curr_adj = defaultdict(list)
            for u, v in remaining_edges:
                curr_adj[u].append(v)
                curr_adj[v].append(u)
                
            # Find longest cycle
            # Use DFS from each node, trying to close the cycle.
            # To keep it fast, limit search depth or use heuristic.
            # Actually, finding the longest cycle is NP-hard, but for n=30 it's manageable with pruning.
            # Let's try a simpler approach: Find ANY cycle, but try to extend it.
            # Or: Find a 2-factor?
            
            # Let's try to find a cycle using a randomized walk that tries to return to start.
            # Or better: Use a standard algorithm to find a cycle, then try to extend it.
            
            # Simple approach:
            # 1. Pick a random edge (u,v).
            # 2. Try to find a path from v to u in G - {(u,v)}.
            # 3. If found, we have a cycle.
            # 4. To make it long, we can try to find a longest path from v to u in G - {(u,v)}.
            
            # Let's implement a function to find a long path between two nodes.
            def find_long_path(start, end, adj, forbidden_edges):
                # DFS with backtracking, maximizing path length
                best_path = None
                best_len = -1
                visited = {start}
                path = [start]
                
                def dfs(node, current_path, current_visited):
                    nonlocal best_path, best_len
                    if node == end and len(current_path) > 2: # Cycle must have at least 3 vertices
                        if len(current_path) > best_len:
                            best_len = len(current_path)
                            best_path = list(current_path)
                        return
                    
                    # Pruning: if max possible length <= best_len, stop
                    # Max possible length is len(current_path) + (n - len(current_visited))
                    # This is a weak bound, but helps.
                    
                    for neighbor in adj[node]:
                        if neighbor in current_visited:
                            continue
                        # Check if edge (node, neighbor) is forbidden
                        edge = tuple(sorted((node, neighbor)))
                        if edge in forbidden_edges:
                            continue
                        
                        current_visited.add(neighbor)
                        current_path.append(neighbor)
                        dfs(neighbor, current_path, current_visited)
                        current_path.pop()
                        current_visited.remove(neighbor)
                
                dfs(start, path, visited)
                return best_path

            # Try to find a cycle by picking a random edge and finding a long path between its endpoints
            # To be efficient, pick edges that are likely to be in long cycles?
            # Let's just pick a random edge and see.
            # If we find a cycle, remove it.
            
            # Optimization: Try multiple starting edges if the first one doesn't yield a good cycle?
            # Or just take the first cycle found?
            # Given the time limit, let's take a reasonable approach.
            
            # Let's try to find the longest cycle by iterating over all edges as potential "chords" to close?
            # No, that's too slow.
            
            # Alternative: Find a cycle, remove it. Repeat.
            # To bias towards longer cycles, we can try to find a cycle that includes high-degree nodes?
            
            # Let's stick to: Find a cycle using DFS that tries to maximize length.
            # We'll pick a random edge (u,v), remove it from consideration for the path, and find longest path v->u.
            
            found_cycle = None
            # Try a few random edges to find a long cycle
            edge_list = list(remaining_edges)
            random.shuffle(edge_list)
            
            for u, v in edge_list[:5]: # Try up to 5 random edges
                # Find longest path from v to u avoiding (u,v)
                path = find_long_path(v, u, curr_adj, {(u, v)})
                if path:
                    # The cycle is path + [u]
                    cycle_vertices = path + [u]
                    # Check if this cycle is valid (edges exist)
                    valid = True
                    cycle_edges = set()
                    for i in range(len(cycle_vertices) - 1):
                        a, b = cycle_vertices[i], cycle_vertices[i+1]
                        e = tuple(sorted((a, b)))
                        if e not in remaining_edges:
                            valid = False
                            break
                        cycle_edges.add(e)
                    if valid:
                        # Check if this cycle is longer than the current best found in this iteration
                        if not found_cycle or len(cycle_edges) > len(found_cycle[1]):
                            found_cycle = (cycle_vertices, cycle_edges)
                            
            if found_cycle:
                cycle_vertices, cycle_edges = found_cycle
                pieces.append(cycle_vertices)
                remaining_edges -= cycle_edges
            else:
                # No cycle found (or search failed), remove a single edge
                # Pick an edge that is in the fewest cycles? Or just random?
                # Let's pick an edge that is a bridge if possible?
                # Just pick a random edge for simplicity.
                edge_to_remove = random.choice(list(remaining_edges))
                pieces.append(list(edge_to_remove))
                remaining_edges.remove(edge_to_remove)
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Return what we have so far? No, must return 3 lists.
            # If we run out of time, we should return a valid decomposition for the remaining graphs.
            # The safest is to return edge-per-piece for the remaining ones.
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        pieces = find_cycles_and_edges(edges)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
