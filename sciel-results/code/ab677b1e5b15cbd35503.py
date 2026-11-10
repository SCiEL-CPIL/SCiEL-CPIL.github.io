import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    
    # Bitmask helpers
    def make_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free_add(adj, u, v):
        # Check if (u,v) is already an edge
        if (adj[u] >> v) & 1:
            return True # It's already there, technically "free" but no change
        # Check if common neighbor exists
        return (adj[u] & adj[v]) == 0

    def count_edges(adj):
        # Sum of bits in all adj[i] / 2
        return sum(bin(x).count('1') for x in adj) // 2

    def is_3_colorable(adj):
        """
        Exact backtracking to check if graph is 3-colorable.
        Returns True if 3-colorable, False otherwise.
        """
        colors = [-1] * n
        
        # Order vertices by degree descending for better pruning
        # But since adj is bitmask, we can just iterate 0..n-1 or use a heuristic
        # For n=16, simple order is often fast enough if we use bitmasks for available colors?
        # Let's stick to simple backtracking with forward checking.
        
        def backtrack(i):
            if i == n:
                return True
            
            # Determine available colors for vertex i
            used_colors = 0
            for j in range(n):
                if (adj[i] >> j) & 1:
                    if colors[j] != -1:
                        used_colors |= (1 << colors[j])
            
            # Try each color 0, 1, 2
            for c in range(3):
                if not (used_colors & (1 << c)):
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = -1
            
            return False
        
        # Heuristic: Start with vertex of highest degree?
        # Let's just use natural order for simplicity, n=16 is small.
        # Actually, ordering by degree can help.
        # Let's create a list of vertices sorted by degree descending.
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        # We need to map the backtrack index to the actual vertex
        # Let's modify backtrack to take an index into 'order'
        
        def backtrack_idx(idx):
            if idx == n:
                return True
            
            v = order[idx]
            used_colors = 0
            # Check neighbors
            mask = adj[v]
            # Iterate neighbors
            # Since n is small, loop is fine
            for u in range(n):
                if (mask >> u) & 1:
                    if colors[u] != -1:
                        used_colors |= (1 << colors[u])
            
            for c in range(3):
                if not (used_colors & (1 << c)):
                    colors[v] = c
                    if backtrack_idx(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack_idx(0)

    def greedy_fill(adj):
        """
        Add all possible edges that maintain triangle-freeness.
        """
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if is_triangle_free_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
        return adj

    # Grötzsch graph edges (11 vertices)
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    best_edges_count = 0
    best_adj = None
    
    # Initial seed: Grötzsch + 5 isolated vertices
    init_edges = groetzsch_edges[:]
    # Vertices 11..15 are isolated initially
    
    init_adj = make_adj_from_edges(init_edges)
    init_adj = greedy_fill(init_adj)
    
    if not is_3_colorable(init_adj):
        best_edges_count = count_edges(init_adj)
        best_adj = init_adj[:]
    else:
        # If for some reason it's 3-colorable (shouldn't be with Groetzsch core),
        # we need to find a 4-chromatic one.
        # But Groetzsch is 4-chromatic. Adding edges keeps it 4-chromatic.
        # So it should be fine.
        pass
    
    # Local Search
    # We will try to remove an edge and re-greedy-fill.
    # If the new graph has more edges, we accept it.
    # To diversify, we also try random restarts from the best found.
    
    current_adj = best_adj[:] if best_adj else init_adj[:]
    
    # Precompute list of edges in current_adj for removal
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask &= mask - 1
        return edges
    
    steps_without_improvement = 0
    max_steps_without_improvement = 50
    
    while time.time() < deadline:
        current_edges = get_edges(current_adj)
        
        if not current_edges:
            break
            
        # Pick a random edge to remove
        # Prioritize removing edges that are "redundant" or part of many triangles? 
        # No, triangle-free.
        # Just random.
        u, v = random.choice(current_edges)
        
        # Remove edge
        test_adj = current_adj[:]
        test_adj[u] &= ~(1 << v)
        test_adj[v] &= ~(1 << u)
        
        # Greedy fill
        test_adj = greedy_fill(test_adj)
        
        new_count = count_edges(test_adj)
        
        # Check 3-colorability of the new graph
        # If it becomes 3-colorable, we must reject it (we need chi >= 4)
        # Note: Removing an edge can make a 4-chromatic graph 3-colorable.
        if new_count > best_edges_count and not is_3_colorable(test_adj):
            best_edges_count = new_count
            best_adj = test_adj[:]
            current_adj = test_adj[:]
            steps_without_improvement = 0
        else:
            # If we didn't improve, or it became 3-colorable, we revert.
            # But we should also check if current_adj is still valid?
            # current_adj is always valid by construction.
            
            # If we are stuck, try a restart from best_adj with a perturbation
            if steps_without_improvement > max_steps_without_improvement:
                # Perturb best_adj: remove a random edge from best_adj and re-fill
                best_edges_list = get_edges(best_adj)
                if best_edges_list:
                    u, v = random.choice(best_edges_list)
                    perturbed_adj = best_adj[:]
                    perturbed_adj[u] &= ~(1 << v)
                    perturbed_adj[v] &= ~(1 << u)
                    perturbed_adj = greedy_fill(perturbed_adj)
                    
                    if not is_3_colorable(perturbed_adj):
                        current_adj = perturbed_adj
                        # We don't update best unless it's better, but we switch current to explore
                        # Actually, if it's worse, we just use it as a starting point for local search
                        # The local search will try to add edges back.
                        
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                
    # Final check
    if best_adj is None:
        # Fallback: return Grötzsch edges
        return [[u, v] for u, v in groetzsch_edges]
        
    return [[u, v] for u, v in get_edges(best_adj)]
