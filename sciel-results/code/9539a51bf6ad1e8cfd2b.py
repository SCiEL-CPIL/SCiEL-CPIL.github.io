import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph C4-free.
        adj is a list of sets of neighbors.
        u, v are not adjacent.
        """
        # 1. |N(u) ∩ N(v)| <= 1
        common_uv = adj[u].intersection(adj[v])
        if len(common_uv) > 1:
            return False
        
        # 2. For all w in N(u), N(w) ∩ N(v) must be empty
        #    Because if z in N(w) ∩ N(v), then w-z-v-u-w is a C4?
        #    Wait, u-w-z-v-u.
        #    Edges: (u,w), (w,z), (z,v), (v,u).
        #    Yes, this is a C4.
        for w in adj[u]:
            if adj[w].intersection(adj[v]):
                return False
        
        # 3. For all w in N(v), N(w) ∩ N(u) must be empty
        for w in adj[v]:
            if adj[w].intersection(adj[u]):
                return False
        
        return True

    def exact_alpha(adj, n):
        """
        Exact maximum independent set size using branch and bound.
        adj: list of sets of neighbors.
        """
        best_alpha = 0
        # Order vertices by degree descending to prune faster
        vertices = list(range(n))
        vertices.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # Map original index to new index for efficiency? 
        # Actually, just using the set is fine for n<=43.
        
        def dfs(current_set, remaining):
            nonlocal best_alpha
            if not remaining:
                if len(current_set) > best_alpha:
                    best_alpha = len(current_set)
                return
            
            # Upper bound: len(current_set) + len(remaining)
            if len(current_set) + len(remaining) <= best_alpha:
                return
            
            # Pick a vertex from remaining
            # Heuristic: pick vertex with highest degree in induced subgraph on remaining?
            # Or just pick the first one.
            v = next(iter(remaining))
            
            # Option 1: Include v
            # New remaining: remaining - {v} - N(v)
            new_remaining = remaining - {v} - adj[v]
            current_set.add(v)
            dfs(current_set, new_remaining)
            current_set.remove(v)
            
            # Option 2: Exclude v
            new_remaining2 = remaining - {v}
            dfs(current_set, new_remaining2)
            
        dfs(set(), set(vertices))
        return best_alpha

    def greedy_color_upper_bound(adj, n):
        """
        Greedy coloring to get an upper bound on alpha.
        Actually, chromatic number chi >= n / alpha, so alpha <= n / chi.
        We want a lower bound on chi to get an upper bound on alpha.
        Greedy coloring gives an upper bound on chi, which is not useful for upper bounding alpha directly in this way.
        However, a greedy independent set gives a lower bound on alpha.
        To check if alpha < 11, we need to prove alpha <= 10.
        Exact search is better for n=40.
        """
        pass

    def build_random_c4_free(n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Try to add all possible edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
        return adj, edges

    # Try n from 43 down to 30
    for n in range(43, 29, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Multiple attempts for each n
        attempts = 50
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            adj, edges = build_random_c4_free(n, seed=42 + attempt * 1000 + n)
            
            # Check alpha
            # For n > 35, exact alpha might be slow if not pruned well.
            # But n=40 is small enough for a good B&B.
            alpha = exact_alpha(adj, n)
            
            if alpha < 11:
                return {'n': n, 'edges': edges}
        
    # Fallback: Return the best known static construction (30 vertices)
    # Or try a bit harder for n=39 with more attempts if time permits
    if time.time() - start_time < time_limit - 5:
        # Try n=39 with more attempts
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
            adj, edges = build_random_c4_free(39, seed=9999 + attempt * 77)
            alpha = exact_alpha(adj, 39)
            if alpha < 11:
                return {'n': 39, 'edges': edges}

    # Final fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
