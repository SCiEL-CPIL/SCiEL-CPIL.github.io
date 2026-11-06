import random
import time
import itertools

def has_triangle(edges, n):
    """
    Check if the graph contains a triangle.
    Returns the triangle if found, else None.
    edges is a list of (u, v).
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Optimization: Check edges one by one
    for u, v in edges:
        common = adj[u] & adj[v]
        if common:
            w = next(iter(common))
            return (u, v, w)
    return None

def max_independent_set_size(edges, n, limit=10):
    """
    Check if there is an independent set of size >= limit.
    Returns True if such a set exists.
    """
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Use a simple branch and bound / backtracking
    # To speed up, we can use a greedy coloring bound or similar, 
    # but for n<=40, simple backtracking with pruning is often fast enough 
    # if the graph is dense.
    
    # Heuristic: Order vertices by degree (descending) to prune faster?
    # Or just use the natural order.
    
    def backtrack(current_set, candidates):
        if len(current_set) >= limit:
            return True
        
        # Pruning: If len(current_set) + len(candidates) < limit, return False
        if len(current_set) + len(candidates) < limit:
            return False
        
        if not candidates:
            return False
            
        # Pick a vertex from candidates
        # Heuristic: pick vertex with lowest degree in the induced subgraph of candidates?
        # For simplicity, pick the first one.
        v = candidates[0]
        
        # Option 1: Include v
        new_candidates = [u for u in candidates[1:] if u not in adj[v]]
        if backtrack(current_set + [v], new_candidates):
            return True
            
        # Option 2: Exclude v
        if backtrack(current_set, candidates[1:]):
            return True
            
        return False

    return backtrack([], list(range(n)))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try n from 35 down to 27
    for n in range(35, 26, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        max_tries = 2000
        for _ in range(max_tries):
            if time.time() > deadline:
                break
                
            # Generate a random graph
            # To encourage density, start with a high edge probability
            # But we will remove edges to make it triangle-free.
            # A better start might be a random regular graph or just random.
            
            # Let's try starting with a random graph with p=0.5
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.append((i, j))
            
            # Make it triangle-free
            # We need to remove edges to break all triangles.
            # We do this iteratively.
            changed = True
            while changed:
                changed = False
                # Find a triangle
                # To speed up, we can check triangles more efficiently
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Find one triangle
                tri = None
                # Iterate through edges
                for u, v in edges:
                    common = adj[u] & adj[v]
                    if common:
                        w = next(iter(common))
                        tri = (u, v, w)
                        break
                
                if tri:
                    # Remove one edge from the triangle
                    # Randomly pick which edge to remove
                    u, v, w = tri
                    edge_to_remove = random.choice([(u, v), (v, w), (w, u)])
                    # Ensure correct ordering for removal
                    if edge_to_remove[0] > edge_to_remove[1]:
                        edge_to_remove = (edge_to_remove[1], edge_to_remove[0])
                    
                    try:
                        edges.remove(edge_to_remove)
                        changed = True
                    except ValueError:
                        # Edge might have been removed already? 
                        # This shouldn't happen if we update adj properly, 
                        # but edges list is modified.
                        # Let's rebuild adj every time to be safe, though slow.
                        pass
                else:
                    # No triangles
                    break
            
            # Check independent set
            if not max_independent_set_size(edges, n, 10):
                # Valid graph found
                return {'n': n, 'edges': [[u, v] for u, v in edges]}
                
    # Fallback: Return the Andrásfai graph And(9)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
