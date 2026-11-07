import random
import time

def construct():
    n = 15
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def get_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid(edges):
        if not edges:
            return True
        adj = get_adj(edges)
        deg = get_degrees(edges)
        for u, v in edges:
            # Check if there exists s in N(u)\{v} and t in N(v)\{u} with deg[s] == deg[t]
            # Optimization: Precompute set of degrees of neighbors
            degs_u = set(deg[x] for x in adj[u] if x != v)
            degs_v = set(deg[y] for y in adj[v] if y != u)
            if degs_u & degs_v:
                return False
        return True

    def try_bipartite():
        """
        Deterministically construct the best complete bipartite graph K(a,b)
        where a+b=n and a!=b.
        """
        best_bip_edges = []
        for a in range(1, n):
            b = n - a
            if a == b:
                continue
            # Construct K(a,b)
            # Part A: 0..a-1, Part B: a..n-1
            edges = []
            for i in range(a):
                for j in range(a, n):
                    edges.append((i, j))
            if len(edges) > len(best_bip_edges):
                best_bip_edges = edges
        return best_bip_edges

    def repair(edges):
        """
        Takes a list of edges (possibly invalid) and removes edges until valid.
        """
        edges = list(edges)
        while True:
            if is_valid(edges):
                break
            
            # Find all invalid edges
            adj = get_adj(edges)
            deg = get_degrees(edges)
            invalid_edges = []
            for u, v in edges:
                degs_u = set(deg[x] for x in adj[u] if x != v)
                degs_v = set(deg[y] for y in adj[v] if y != u)
                if degs_u & degs_v:
                    invalid_edges.append((u, v))
            
            if not invalid_edges:
                break # Should not happen if is_valid returned False

            # Strategy: Remove the edge with the highest sum of degrees of its endpoints.
            # This tends to reduce density of high-degree nodes, which might help.
            # Sort by degree sum descending and remove the first one.
            def degree_sum(e):
                u, v = e
                return deg[u] + deg[v]
            
            invalid_edges.sort(key=degree_sum, reverse=True)
            edge_to_remove = invalid_edges[0]
            edges.remove(edge_to_remove)
            
        return edges

    def local_search_add(edges):
        """
        Greedily add any edge that keeps the graph valid.
        """
        edges = set(edges)
        improved = True
        while improved:
            improved = False
            # Try to add each missing edge
            # Shuffle to avoid bias
            candidates = [e for e in all_edges if e not in edges]
            random.shuffle(candidates)
            for e in candidates:
                # Check validity efficiently
                # We can reuse adj/deg from the current state, but adding an edge changes degrees.
                # is_valid does a full check.
                if is_valid(list(edges) + [e]):
                    edges.add(e)
                    improved = True
        return list(edges)

    # Main Loop
    start_time = time.time()
    max_time = 50 # Leave 10s buffer
    
    # 1. Deterministic Bipartite Initialization
    bip_edges = try_bipartite()
    if bip_edges:
        # Verify it's valid (it should be)
        if is_valid(bip_edges):
            best_edges = bip_edges[:]
            best_count = len(best_edges)
    
    # 2. Randomized Local Search to try to beat the bipartite solution
    for i in range(1000): # Max iterations
        if time.time() - start_time > max_time:
            break
        
        # Initialize
        # Sometimes start from the best found so far to improve it
        if i % 10 == 0 and best_edges:
            current_edges = best_edges[:]
        else:
            current_edges = all_edges[:]
            random.shuffle(current_edges)
        
        # Repair
        current_edges = repair(current_edges)
        
        # Local Search
        current_edges = local_search_add(current_edges)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    return [list(e) for e in best_edges]
