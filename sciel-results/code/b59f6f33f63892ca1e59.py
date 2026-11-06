import random
import time
import sys

def construct():
    """
    Finds a triangle-free graph on 16 vertices that is not 3-colorable 
    and has the maximum number of edges.
    """
    n = 16
    best_edges = []
    best_count = 0
    start_time = time.time()
    timeout = 115  # Leave 5s buffer

    def is_triangle_free(adj, u, v):
        """Check if adding edge (u, v) creates a triangle."""
        # adj is a list of bitmasks
        # neighbors of u: adj[u]
        # neighbors of v: adj[v]
        # common neighbors: adj[u] & adj[v]
        # if common neighbors is not empty, triangle exists
        return (adj[u] & adj[v]) == 0

    def get_maximal_extension(core_edges, seed=None):
        """
        Greedily extends core_edges to a maximal triangle-free graph.
        Uses randomization to explore different maximal extensions.
        Returns (edge_count, edges_list)
        """
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # List of all possible edges
        all_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                all_edges.append((u, v))
        
        # Filter to addable edges
        addable = []
        for u, v in all_edges:
            if not (adj[u] >> v) & 1:
                if is_triangle_free(adj, u, v):
                    addable.append((u, v))
        
        # Greedy addition with random order
        random.shuffle(addable)
        
        current_edges = list(core_edges)
        current_count = len(core_edges)
        
        for u, v in addable:
            # Check if still addable (might have been forbidden by previous additions)
            if not (adj[u] >> v) & 1 and is_triangle_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append([u, v])
                current_count += 1
                
        return current_count, current_edges

    def is_3_colorable(edges):
        """
        Exact backtracking 3-colorability check using bitmasks.
        """
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            
            # Get available colors
            used_colors = 0
            neighbors = adj[v]
            for u in range(n):
                if (neighbors >> u) & 1:
                    if color[u] != -1:
                        used_colors |= (1 << color[u])
                        
            for c in range(3):
                if not (used_colors >> c) & 1:
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    # Candidate 1: Groetzsch Graph
    groetzsch_edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        groetzsch_edges.append([i, (i + 1) % 5])
    # Edges from cycle to copies
    for i in range(5):
        u = i
        w = (i + 1) % 5
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    # Edges from copies to root 10
    for i in range(5):
        groetzsch_edges.append([5 + i, 10])
    
    # Evaluate Grötzsch core
    if not is_3_colorable(groetzsch_edges):
        count, edges = get_maximal_extension(groetzsch_edges)
        if count > best_count:
            best_count = count
            best_edges = edges

    # Random Search for better cores
    # Strategy: Mutate the Grötzsch core and find its maximal extension
    for _ in range(2000):
        if time.time() - start_time > timeout:
            break
        
        # Start with Grötzsch edges
        current_core = list(groetzsch_edges)
        
        # Randomly add/remove edges to create a different core
        # Ensure it remains triangle-free
        adj = [0] * n
        for u, v in current_core:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Try to add random edges
        for _ in range(10):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] >> v) & 1:
                continue
            if is_triangle_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_core.append([u, v])
                
        # Check if this new core is not 3-colorable
        if not is_3_colorable(current_core):
            count, edges = get_maximal_extension(current_core)
            if count > best_count:
                best_count = count
                best_edges = edges

    return best_edges

# Run construct to verify it works and returns something
if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
