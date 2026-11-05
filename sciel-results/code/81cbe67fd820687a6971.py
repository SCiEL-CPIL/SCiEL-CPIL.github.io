"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = 20
    # We want to maximize edges in a K2,3-free graph on 20 vertices.
    # A K2,3-free graph means no two vertices have 3 or more common neighbors.
    
    # Strategy:
    # 1. Start with a good heuristic construction (e.g., based on finite geometry or random greedy).
    # 2. Use local search (edge addition/removal) to improve.
    # 3. Randomized restarts with different initial seeds.
    
    def is_valid(edges_set, adj):
        """Check if the graph is K2,3-free."""
        # For every pair of vertices (u, v), count common neighbors.
        # If count >= 3, it's invalid.
        for u in range(n):
            for v in range(u + 1, n):
                # Common neighbors are in intersection of adj[u] and adj[v]
                # Since n is small (20), we can just iterate
                common = 0
                for w in range(n):
                    if w != u and w != v:
                        if w in adj[u] and w in adj[v]:
                            common += 1
                            if common >= 3:
                                return False
        return True
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v."""
        count = 0
        for w in range(n):
            if w != u and w != v:
                if w in adj[u] and w in adj[v]:
                    count += 1
        return count
    
    def try_add_edge(adj, edges_set, u, v):
        """Try to add edge (u, v). Returns True if valid."""
        if u in adj[v]:
            return False  # Already exists
        
        # Check if adding (u, v) creates a K2,3
        # A K2,3 is formed if there exist vertices x, y such that:
        # - x and y are both neighbors of u and v (common neighbors)
        # - AND there is a third common neighbor? No.
        # K2,3 consists of two vertices (say u, v) and three vertices (a, b, c) 
        # such that u and v are both connected to a, b, c.
        # So if we add edge (u, v), we need to ensure that u and v do NOT have 3 common neighbors.
        # Before adding, let common = count_common_neighbors(adj, u, v).
        # If common >= 3, adding the edge doesn't change the common neighbors count?
        # Wait. The common neighbors of u and v are vertices w such that (u,w) in E and (v,w) in E.
        # Adding (u,v) does not change the set of common neighbors.
        # However, adding (u,v) creates new K2,3s where u and v are part of the "2" side?
        # No, a K2,3 has bipartition {u, v} and {a, b, c}.
        # The edges are (u,a), (u,b), (u,c), (v,a), (v,b), (v,c).
        # The edge (u,v) is NOT part of the K2,3 structure itself.
        # So adding (u,v) is safe with respect to K2,3s where u and v are the "2" side,
        # UNLESS the definition of K2,3-free considers the graph structure differently.
        # Standard definition: A graph is K2,3-free if it does not contain a subgraph isomorphic to K2,3.
        # K2,3 is a bipartite graph with partitions of size 2 and 3.
        # So the condition is: for any pair of vertices {u, v}, the number of common neighbors must be <= 2.
        # This condition is independent of whether (u,v) is an edge or not.
        # Therefore, adding (u,v) is valid if and only if count_common_neighbors(adj, u, v) <= 2.
        
        if count_common_neighbors(adj, u, v) <= 2:
            adj[u].add(v)
            adj[v].add(u)
            edges_set.add((u, v))
            return True
        return False
    
    def remove_edge(adj, edges_set, u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        edges_set.discard((u, v))
    
    def random_greedy(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges_set = set()
        
        # Order vertices randomly? Or just iterate all pairs randomly.
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if count_common_neighbors(adj, u, v) <= 2:
                adj[u].add(v)
                adj[v].add(u)
                edges_set.add((u, v))
        
        return adj, edges_set
    
    def local_search(adj, edges_set, time_limit):
        """Local search: try to add edges, if stuck, remove an edge and try again."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add any possible edge
            added = False
            # Shuffle pairs for randomness
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if count_common_neighbors(adj, u, v) <= 2:
                    try_add_edge(adj, edges_set, u, v)
                    added = True
                    improved = True
                    break  # Restart loop after one addition? Or continue?
            
            if not added:
                # Try to remove an edge and add two others? Or just remove one to unlock others.
                # Strategy: Remove an edge that is "blocking" many potential additions.
                # Simpler: Remove a random edge and see if we can add more.
                if len(edges_set) > 0:
                    # Pick an edge to remove. Prefer edges that are "critical" for some K2,3 constraint?
                    # Actually, removing an edge never violates K2,3-freeness.
                    # We want to remove an edge that allows adding >= 2 edges.
                    # Let's just try removing a random edge and then greedily adding.
                    # To be efficient, we can check for each edge e, if removing it allows adding more edges.
                    # This is expensive. Let's do a simple hill climb.
                    
                    # Pick a random edge to remove
                    edges_list = list(edges_set)
                    random.shuffle(edges_list)
                    # Try a few removals
                    for u, v in edges_list[:5]:
                        # Remove
                        remove_edge(adj, edges_set, u, v)
                        old_count = len(edges_set)
                        
                        # Greedily add
                        pairs2 = [(x, y) for x in range(n) for y in range(x + 1, n) if y not in adj[x]]
                        random.shuffle(pairs2)
                        for x, y in pairs2:
                            if count_common_neighbors(adj, x, y) <= 2:
                                try_add_edge(adj, edges_set, x, y)
                        
                        if len(edges_set) > old_count:
                            improved = True
                            break
                        else:
                            # Restore
                            adj[u].add(v)
                            adj[v].add(u)
                            edges_set.add((u, v))
                    
                    if not improved:
                        break
        
        return adj, edges_set
    
    best_edges_count = 0
    best_edges = []
    
    # Run multiple random restarts
    num_restarts = 0
    while time.time() < deadline:
        seed = int(time.time() * 1000000) + num_restarts
        adj, edges_set = random_greedy(seed)
        
        # Local search
        time_limit = min(deadline - 5.0, time.time() + 10.0)
        adj, edges_set = local_search(adj, edges_set, time_limit)
        
        if len(edges_set) > best_edges_count:
            best_edges_count = len(edges_set)
            best_edges = list(edges_set)
        
        num_restarts += 1
        if num_restarts > 100: # Cap restarts to avoid infinite loop if search is too fast
            break
            
    # Convert to list of lists [u, v]
    result = [[u, v] for u, v in best_edges]
    return result
# EVOLVE-BLOCK-END
