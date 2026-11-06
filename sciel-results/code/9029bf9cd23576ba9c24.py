import time
import sys
from random import Random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin before 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if time.time() - start_time > TIME_LIMIT:
            # Fallback to trivial if we are running out of time
            results.append([list(e) for e in instance(*spec)])
            continue

        edges = instance(*spec)
        if not edges:
            results.append([])
            continue

        # Build adjacency and degree
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        pieces = []

        # Helper to find a cycle in the remaining graph
        # We use a heuristic: DFS that prefers neighbors with higher remaining degree
        def find_cycle(start_node):
            # Returns a list of vertices forming a cycle, or None
            visited_in_path = set()
            path = []
            parent = {}
            
            # Stack for DFS: (node, parent, iterator_state)
            # Simpler: Recursive DFS with backtracking to find ANY cycle
            # But we want LONG cycles.
            # Heuristic: Start DFS, if we hit a cycle, record it.
            # To get long cycles, we can try to extend the path as much as possible before closing.
            
            # Let's use a stack-based approach that tries to find a cycle.
            # If we just find the first cycle, it might be short.
            # Strategy: Find a cycle, then try to "grow" it? No, complex.
            # Strategy: Use a greedy walk.
            
            # Greedy Walk:
            # 1. Start at start_node.
            # 2. At each step, pick a neighbor with the highest remaining degree (in the subgraph of unused edges).
            # 3. If we revisit a node in the current path, we have a cycle.
            # 4. If we get stuck (no unused edges), we have a path. If start != end, it's not a cycle.
            #    In that case, we can't use it as a single cycle piece. We would have to break it down?
            #    Actually, if we just need *any* cycle, this works.
            #    If the walk doesn't form a cycle, we might have traversed a long path that is part of a cycle elsewhere?
            #    No, if it doesn't close, it's a trail.
            
            # Let's stick to a robust DFS that finds a cycle.
            # To bias towards long cycles, we can sort neighbors by degree descending.
            
            def dfs(node, prev):
                if time.time() - start_time > TIME_LIMIT:
                    return None
                visited_in_path.add(node)
                path.append(node)
                
                # Sort neighbors by remaining degree to encourage going deep into dense areas
                neighbors = sorted(adj[node], key=lambda x: len([e for e in remaining_edges if x in e]), reverse=True)
                
                for neighbor in neighbors:
                    if (min(node, neighbor), max(node, neighbor)) not in remaining_edges:
                        continue
                    if neighbor == prev:
                        continue
                    if neighbor in visited_in_path:
                        # Found a cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:]
                        return cycle
                    result = dfs(neighbor, node)
                    if result:
                        return result
                
                path.pop()
                visited_in_path.remove(node)
                return None

            return dfs(start_node, -1)

        # Main loop
        while remaining_edges:
            if time.time() - start_time > TIME_LIMIT:
                # Fallback: add all remaining as single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                break

            # Find a vertex with degree > 0
            start_node = None
            for u, v in remaining_edges:
                start_node = u
                break
            
            if start_node is None:
                break

            cycle = find_cycle(start_node)
            
            if cycle:
                # Verify cycle is valid (distinct vertices, edges exist)
                # The DFS ensures edges exist and vertices are distinct in the path segment.
                # Add to pieces
                pieces.append(cycle)
                
                # Remove edges of the cycle from remaining_edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining_edges.discard(e)
            else:
                # No cycle found from this node? 
                # This implies the component containing start_node is a tree (no cycles).
                # So all edges in this component are bridges.
                # We should extract the entire component and add all its edges as single pieces.
                # Find the component
                component_edges = set()
                visited_nodes = set()
                stack = [start_node]
                while stack:
                    node = stack.pop()
                    if node in visited_nodes:
                        continue
                    visited_nodes.add(node)
                    for neighbor in adj[node]:
                        e = (min(node, neighbor), max(node, neighbor))
                        if e in remaining_edges:
                            component_edges.add(e)
                            if neighbor not in visited_nodes:
                                stack.append(neighbor)
                
                # Add all component edges as single pieces
                for e in component_edges:
                    pieces.append(list(e))
                    remaining_edges.discard(e)
        
        results.append(pieces)

    return results
