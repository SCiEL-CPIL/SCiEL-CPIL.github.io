import random
import time
import numpy as np

def construct():
    """
    Constructs a C4-free graph with no independent set of size 11.
    Uses a construction based on the incidence structure of a projective plane or
    a carefully designed algebraic graph to ensure C4-freeness and bounded independence.
    
    The best known lower bound for R(C4, K11) is 39.
    We will attempt to construct a graph on 39 vertices.
    If verification fails, we fall back to a smaller valid graph.
    """
    start_time = time.time()
    
    # Strategy:
    # 1. Try to construct a specific known or algebraic graph on 39 vertices.
    #    One common approach for C4-free graphs is using polarity graphs of finite geometries,
    #    but those are often directed or have specific symmetries.
    #    Another approach: The complement of a strongly regular graph?
    #    Let's try a greedy randomized approach with strong pruning,
    #    but seeded with a structure that is likely to work.
    
    # A known construction for R(C4, K11) > 38 is not trivial.
    # However, there is a known graph on 39 vertices?
    # Actually, the bound R(C4, K11) >= 39 is established.
    # We need to find *a* graph on 39 vertices that is C4-free and has alpha <= 10.
    
    # Let's use a randomized local search / greedy addition.
    # We start with a base structure and add vertices/edges carefully.
    
    n_target = 39
    
    # We will try multiple random restarts of a greedy construction.
    # The greedy construction adds vertices one by one, choosing neighbors
    # to maintain C4-freeness and maximize density (to keep alpha low).
    
    for attempt in range(100):
        if time.time() - start_time > 110:
            break
            
        graph = greedy_construct(n_target)
        if graph is not None:
            # Verify
            if is_c4_free(graph['adj'], n_target) and not has_independent_set_11(graph['adj'], n_target):
                return {'n': n_target, 'edges': graph['edges']}
    
    # Fallback: Return a smaller valid graph if we can't find 39 quickly.
    # We can return the seed graph (30 vertices) which is valid.
    return seed_graph()

def seed_graph():
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def greedy_construct(n):
    """
    Attempts to build a C4-free graph on n vertices.
    Returns dict with 'adj' and 'edges' if successful, else None.
    """
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Order of adding vertices: random permutation
    order = list(range(n))
    random.shuffle(order)
    
    # We'll add vertices in this order.
    # For each new vertex, we choose a set of neighbors from previously added vertices.
    # To be C4-free, for any new vertex v, its neighbors must form a "C4-free" set
    # in the sense that no two neighbors of v have a common neighbor.
    # i.e., the neighborhood of v must be an independent set in the "line graph" sense?
    # No, simply: if u, w are neighbors of v, then N(u) \cap N(w) = empty.
    
    # This is a strong constraint.
    # Let's try a different strategy: Start with a random graph and remove edges to break C4s?
    # That's hard.
    
    # Let's stick to the incremental addition.
    # When adding vertex v, we pick a subset S of existing vertices to be neighbors.
    # Condition: For all u, w in S, N(u) \cap N(w) = empty.
    # This means S must be an independent set in the graph H where H has edges between
    # pairs of existing vertices that share a common neighbor.
    # Let's maintain this "conflict graph" dynamically? Or just check on the fly.
    
    # To maximize density (minimize alpha), we want large neighborhoods.
    # So for each new vertex, we want to find a large independent set in the conflict graph.
    # This is still hard.
    
    # Alternative: Use a known algebraic construction.
    # The polarity graph of the projective plane PG(2, q) is C4-free.
    # For q=5, n=31. For q=7, n=57.
    # We need n=39.
    
    # Let's try a hybrid:
    # Use the polarity graph of PG(2,5) (31 vertices) as a core, and add 8 more?
    # Or just use a random regular graph and repair?
    
    # Given the time limit, let's try a simpler heuristic:
    # Randomly generate a sparse graph, then try to add edges to reduce alpha
    # while maintaining C4-freeness.
    
    # Actually, the previous seed approach was too weak.
    # Let's try to construct a graph from a (v, k, lambda) design if possible.
    
    # For now, let's return None to signal failure of this specific greedy attempt
    # and let the main loop try other strategies or fall back.
    # But we need a better greedy.
    
    # Let's implement the "conflict graph" independent set selection.
    # It might be slow but n=39 is small.
    
    for i, v in enumerate(order):
        if i == 0:
            continue
        
        # Find all potential neighbors among order[0:i]
        # A vertex u is a valid neighbor of v if:
        # 1. It is not already connected? No, we are building it.
        # 2. For any other neighbor w of v, u and w must not share a common neighbor.
        
        # This is tricky to do sequentially.
        # Let's pick neighbors one by one.
        # Start with empty set of neighbors for v.
        # Try to add vertices from the previous ones in random order.
        
        candidates = [u for u in order[:i]]
        random.shuffle(candidates)
        
        neighbors_of_v = []
        for u in candidates:
            # Check if u can be added to neighbors_of_v
            # Condition: For all w in neighbors_of_v, N(u) \cap N(w) must be empty.
            # Note: N(u) and N(w) are in the current graph (which includes v? No, v is being added).
            # So we check common neighbors in the graph *before* adding v.
            
            ok = True
            for w in neighbors_of_v:
                # Check common neighbors of u and w
                # They are in order[0:i]
                for x in order[:i]:
                    if adj[u][x] and adj[w][x]:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                neighbors_of_v.append(u)
                # Connect v and u
                adj[v][u] = 1
                adj[u][v] = 1
                edges.append([v, u])
                
                # Optional: Stop if we have enough neighbors?
                # To keep alpha low, we want high degree.
                # But we must be C4-free.
        
        # We don't check if the graph is "good" yet, just build it.
        
    # After building, we have a C4-free graph by construction.
    # Now we need to check if alpha <= 10.
    # If not, we fail.
    
    return {'adj': adj, 'edges': edges}

def is_c4_free(adj, n):
    for u in range(n):
        for v in range(u+1, n):
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
                    if common > 1:
                        return False
    return True

def has_independent_set_11(adj, n):
    # Use a branch and bound or simple DFS with pruning
    # For n=39, this might be feasible if the graph is dense.
    # If the graph is sparse, it's likely to have large independent sets.
    
    # Heuristic: Greedy coloring of the complement?
    # If we can color the complement with 10 colors, alpha <= 10.
    # But coloring is hard.
    
    # Let's use a simple recursive search with pruning.
    # To speed up, we order vertices by degree in the original graph (high degree first? 
    # Low degree in complement = high degree in original).
    # Vertices with high degree in G are less likely to be in an independent set?
    # No, independent set in G is clique in complement.
    # High degree in G means low degree in complement.
    # So we should pick low-degree vertices in G first for the independent set?
    # Actually, to *find* an independent set, picking low-degree vertices (in G) is good
    # because they have fewer conflicts.
    
    # Let's sort vertices by degree in G (ascending).
    degrees = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    def dfs(idx, current_set, remaining):
        if len(current_set) == 11:
            return True
        if idx >= n:
            return False
        if n - idx + len(current_set) < 11:
            return False
        
        v = order[idx]
        
        # Try including v
        can_include = True
        for u in current_set:
            if adj[v][u]:
                can_include = False
                break
        
        if can_include:
            if dfs(idx+1, current_set + [v], remaining-1):
                return True
        
        # Try excluding v
        if dfs(idx+1, current_set, remaining):
            return True
            
        return False
    
    return dfs(0, [], 11)
