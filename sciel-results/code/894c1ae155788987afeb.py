import random
import time

def construct():
    # Fallback solution: Two disjoint K3,3 components.
    # Cycle lengths: 4, 6. Distinct: 2.
    fallback_edges = []
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            fallback_edges.append([u, v])
            
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            fallback_edges.append([u, v])
            
    # Try to find a graph with 1 distinct cycle length.
    # Target: Only 4-cycles.
    # Strategy: Backtracking to build a 3-regular graph on 12 vertices.
    # Constraint: No triangles. No cycles of length != 4.
    
    n = 12
    start_time = time.time()
    time_limit = 5.0  # seconds
    
    # We will try to construct a bipartite graph (to avoid odd cycles)
    # with girth 4 and no 6-cycles.
    # If we find one, we check if it has ONLY 4-cycles.
    
    # Simple backtracking search
    # Vertices 0..5 in Left, 6..11 in Right?
    # 3-regular bipartite graph on 12 vertices.
    
    def get_cycle_lengths(edges):
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all simple cycles
        # For n=12, this is feasible.
        cycles = set()
        
        def dfs(start, current, visited, path):
            if len(path) > 1 and current == start:
                if len(path) >= 4: # Minimum girth 4
                    # Normalize cycle representation
                    # Rotate to min vertex
                    min_idx = path.index(min(path))
                    normalized = tuple(path[min_idx:] + path[:min_idx])
                    cycles.add(len(path))
                    return
            for neighbor in adj[current]:
                if neighbor == start and len(path) > 1:
                    if len(path) >= 4:
                        min_idx = path.index(min(path))
                        normalized = tuple(path[min_idx:] + path[:min_idx])
                        cycles.add(len(path))
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
                    
        for i in range(n):
            visited = {i}
            dfs(i, i, visited, [i])
            
        return list(cycles)
        
    # Search for a 3-regular bipartite graph with only 4-cycles
    # We can try to generate random 3-regular bipartite graphs and check them.
    # There are not too many.
    
    # Generate random 3-regular bipartite graphs
    # Left: 0-5, Right: 6-11
    # Each left node connects to 3 right nodes.
    # Each right node connects to 3 left nodes.
    
    import itertools
    
    # We can use a simple rejection sampling
    # But 12 vertices is small.
    # Number of 3-regular bipartite graphs on 6+6 vertices is manageable?
    # Total edges = 18.
    # We can try a few hundred random instances.
    
    best_edges = None
    best_distinct = 100
    
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random 3-regular bipartite graph
        # Start with a random permutation for each left node
        # Ensure column degrees are 3.
        
        # Use a simple algorithm:
        # 1. Assign 3 random right neighbors to each left node.
        # 2. Check if right degrees are all 3.
        # 3. If not, shuffle/repair.
        
        # Simpler: Use networkx if available? No, stdlib only.
        # Let's just try random assignments and check degrees.
        
        left = list(range(6))
        right = list(range(6, 12))
        
        edges = []
        deg_right = {v: 0 for v in right}
        
        # Randomly assign 3 neighbors to each left node
        # This might not result in regular right degrees.
        # We need a proper 3-regular bipartite graph.
        
        # Algorithm:
        # 1. Create a list of 18 "slots" for left nodes, each labeled 0..5 (3 each).
        # 2. Create a list of 18 "slots" for right nodes, each labeled 6..11 (3 each).
        # 3. Randomly permute the right slots.
        # 4. Connect left slots to right slots in order.
        
        left_slots = []
        for i in range(6):
            left_slots.extend([i, i, i])
            
        right_slots = []
        for i in range(6, 12):
            right_slots.extend([i, i, i])
            
        random.shuffle(right_slots)
        
        for i, u in enumerate(left_slots):
            v = right_slots[i]
            edges.append([u, v])
            
        # Check if simple (no duplicate edges)
        if len(set(tuple(sorted(e)) for e in edges)) != 18:
            continue
            
        # Check triangle-free (bipartite, so yes)
        # Check cycle lengths
        cl = get_cycle_lengths(edges)
        
        if len(cl) == 1:
            best_edges = edges
            best_distinct = 1
            break
        elif len(cl) < best_distinct:
            best_distinct = len(cl)
            best_edges = edges
            
    if best_edges and best_distinct == 1:
        return best_edges
    else:
        return fallback_edges
