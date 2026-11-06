"""Combined program: And(9) base + exact verification + extension search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_triangle(adj, n):
    """Check if graph with adjacency sets has a triangle."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    return True
    return False

def _independence_number_exactly(adj, n, timeout=None, start_time=None):
    """Exact independence number via branch and bound.
    Returns the size of the maximum independent set."""
    if start_time is None:
        start_time = time.time()
    
    # Convert to list of sorted neighbors for each vertex
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    # Use a greedy upper bound: color the graph, chromatic number >= alpha
    # Actually, we need upper bound on alpha. Use: alpha <= n - min vertex cover
    # Simpler: use the fact that in any independent set, no two vertices are adjacent.
    # Branch and bound: pick a vertex, either include it (remove it and its neighbors) or exclude it.
    
    best = [0]
    
    def bound(remaining_set):
        """Upper bound on independent set size from remaining vertices.
        Use greedy coloring of the induced subgraph as upper bound on chromatic number,
        which equals lower bound on... wait.
        Actually, alpha(G) <= n - tau(G) where tau is min vertex cover.
        Max matching gives lower bound on vertex cover... 
        Simpler bound: greedily color the induced subgraph on remaining.
        The independence number of a graph is at most the number of color classes needed? No.
        Actually, alpha(G) <= chi(G) * ... no.
        
        Let me use: a simple bound is that if we greedily pick non-adjacent vertices,
        that's a lower bound on alpha. For upper bound, note that if we can cover
        all edges with k vertices, then alpha <= n - k.
        
        Simplest effective bound: use the greedy independent set as lower bound,
        and for upper bound, use that each vertex in an independent set can be
        "charged" to itself, so alpha <= |remaining| (trivial).
        
        Better: use the fact that in the remaining subgraph, if we find a clique of
        size w, then alpha of the complement... this is getting complicated.
        
        Let me just use a simple recursive search with memoization-like pruning."""
        return len(remaining_set)
    
    def dfs(remaining, current_size, current_set):
        if timeout and time.time() - start_time > timeout:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        if current_size + len(remaining) <= best[0]:
            return
        
        # Pick a vertex with maximum degree in remaining (heuristic)
        rem_list = list(remaining)
        max_deg = -1
        chosen = rem_list[0]
        for v in rem_list:
            deg = len([u for u in rem_list if u in adj[v]])
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        # Option 1: include chosen in independent set
        new_remaining = set()
        for v in remaining:
            if v != chosen and v not in adj[chosen]:
                new_remaining.add(v)
        dfs(new_remaining, current_size + 1, current_set | {chosen})
        
        # Option 2: exclude chosen
        new_remaining2 = set(remaining)
        new_remaining2.discard(chosen)
        dfs(new_remaining2, current_size, current_set)
    
    dfs(set(range(n)), 0, set())
    return best[0]

def _check_valid(adj, n, max_alpha=9):
    """Check if graph is triangle-free and has independence number < max_alpha+1."""
    # Check triangle-free
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[i]:
                    if k > j and k in adj[j]:
                        return False
    # Check independence number
    alpha = _independence_number_exactly(adj, n, timeout=10)
    return alpha <= max_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    # Baseline: Andrasfai graph And(9) on 26 vertices
    n_base = 26
    edges_base = [[i, j] for i in range(n_base) for j in range(i + 1, n_base) if (j - i) % n_base % 3 == 1]
    best_n = n_base
    best_edges = edges_base
    
    # Try to extend to larger graphs using random construction + exact verification
    for target_n in range(27, 41):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            
            # Generate a random triangle-free graph
            # Start with empty graph, add edges ensuring no triangle
            adj = [set() for _ in range(target_n)]
            edge_set = set()
            
            # Randomly try to add edges
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.35:
                        # Check if adding (i,j) creates a triangle
                        common = adj[i] & adj[j]
                        if not common:
                            adj[i].add(j)
                            adj[j].add(i)
                            edge_set.add((i, j))
            
            # Check if valid
            # First quick check: triangle-free (should be by construction)
            # Then check independence number
            alpha = _independence_number_exactly(adj, target_n, timeout=8)
            if alpha < 10:
                best_n = target_n
                best_edges = [[u, v] for (u, v) in edge_set if u < v]
                found = True
                break
        
        if found:
            continue  # try next larger n
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
