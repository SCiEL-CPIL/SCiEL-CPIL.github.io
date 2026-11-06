import itertools
import random
import sys
import time

def construct():
    """
    Constructs a C4-free graph on up to 43 vertices with independence number <= 10.
    Starts with the polarity graph of PG(2,5) and greedily adds vertices.
    """
    q = 5
    
    # 1. Generate base graph (Polarity graph of PG(2,5))
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_base = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    edges = []
    # Adjacency list
    adj = [set() for _ in range(n_base)]
    
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u = points[i]
            v = points[j]
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    current_n = n_base
    current_adj = [set(s) for s in adj] # Copy
    
    # Helper: Check if a set S is C4-compatible in the current graph
    # S is C4-compatible if for all u != v in S, N(u) and N(v) are disjoint.
    def is_c4_compatible(S):
        S_list = list(S)
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i+1, len(S_list)):
                v = S_list[j]
                if current_adj[u].intersection(current_adj[v]):
                    return False
        return True

    # Helper: Find Maximum Independent Set size in induced subgraph on vertices V_set
    # Optimized Branch and Bound (Tomita-style)
    def max_independent_set_size(V_set):
        if not V_set:
            return 0
        
        # Convert to list for indexing
        V_list = list(V_set)
        n_v = len(V_list)
        if n_v == 0:
            return 0
        
        # Map original indices to 0..n_v-1 for local adjacency
        # local_adj[i] is set of indices j in V_list such that V_list[i] and V_list[j] are adjacent
        local_adj = [set() for _ in range(n_v)]
        for i in range(n_v):
            u = V_list[i]
            for j in range(i+1, n_v):
                v = V_list[j]
                if v in current_adj[u]:
                    local_adj[i].add(j)
                    local_adj[j].add(i)
        
        # Greedy Coloring for Upper Bound
        # We color the graph such that vertices in the same color class are independent?
        # No, for MIS, we want to color the *complement*? 
        # Standard MIS B&B: 
        # We order vertices. 
        # We use a coloring of the remaining graph to bound the size of the MIS.
        # The greedy coloring of the *induced subgraph* gives an upper bound on the size of the *clique* in the complement, which is the MIS in the original.
        # Wait, MIS in G is Clique in G_bar.
        # So we run a Max Clique algorithm on G_bar[V_set].
        
        # Let's implement a standard Max Clique solver on the complement graph.
        # Vertices: 0..n_v-1
        # Edge in complement if NOT edge in original.
        
        # Precompute non-adjacency (complement edges)
        comp_adj = [set() for _ in range(n_v)]
        for i in range(n_v):
            for j in range(n_v):
                if i != j and j not in local_adj[i]:
                    comp_adj[i].add(j)
        
        # Tomita's MCS algorithm for Max Clique
        # Order vertices by degree in complement?
        # Let's sort vertices by degree in the complement graph to improve branching.
        # Actually, standard MCS uses a specific ordering.
        
        # For simplicity and speed, we'll use a recursive B&B with a greedy coloring bound.
        # The bound is: color the remaining vertices greedily. The color number is an upper bound on the clique size (MIS size).
        
        best_size = 0
        
        # Order vertices: Higher degree in complement first?
        # Let's try sorting by degree in complement descending.
        order = sorted(range(n_v), key=lambda x: len(comp_adj[x]), reverse=True)
        
        # Map old index to new position? 
        # It's easier to just work with the sets directly or use the original indices but restricted to V_set.
        # Let's stick to local indices 0..n_v-1.
        
        # Greedy coloring function for a set of vertices P
        # Returns a list of colors for each vertex in P, and the max color used.
        # Vertices with the same color are independent in the complement (i.e., adjacent in original? No.)
        # In the complement graph, we are finding a Clique.
        # Greedy coloring of the complement graph: 
        # If two vertices have the same color, they are NOT adjacent in the complement?
        # No, coloring assigns colors such that adjacent vertices have different colors.
        # So if we color the complement graph, vertices with the same color form an Independent Set in the complement, which is a Clique in the original.
        # We want a Clique in the complement.
        # The greedy coloring of the complement graph provides an upper bound on the size of the maximum clique?
        # Yes, because any clique in the complement must have distinct colors.
        
        def greedy_coloring(P):
            # P is a set of local indices
            colors = {}
            color_set = set()
            # Sort P by degree in complement restricted to P? 
            # Simpler: just iterate
            for v in P:
                # Find the smallest color not used by neighbors of v in P
                neighbor_colors = {colors[u] for u in P if u in comp_adj[v] and u in colors}
                c = 0
                while c in neighbor_colors:
                    c += 1
                colors[v] = c
                color_set.add(c)
            return colors, len(color_set)

        def dfs(P, C):
            nonlocal best_size
            if not P:
                if len(C) > best_size:
                    best_size = len(C)
                return
            
            # Bound: Color P
            # We only need the number of colors, but we can also use the specific coloring to order vertices.
            # Tomita's algorithm colors P and then branches on vertices in decreasing color order.
            colors, max_color = greedy_coloring(P)
            
            if len(C) + max_color <= best_size:
                return
            
            # Order vertices in P by color (descending)
            P_list = list(P)
            P_list.sort(key=lambda v: colors[v], reverse=True)
            
            for v in P_list:
                # If color of v + current size <= best_size, break (since colors are decreasing)
                if len(C) + colors[v] <= best_size:
                    break
                
                # Include v
                new_C = C | {v}
                # New P: vertices in P that are adjacent to v in complement
                new_P = P.intersection(comp_adj[v])
                
                dfs(new_P, new_C)
                
                # Exclude v
                P.remove(v)
                
        dfs(set(range(n_v)), set())
        return best_size

    # Helper: Try to add a new vertex with neighbor set S
    def try_add_vertex(S):
        nonlocal current_n, current_adj, edges
        if not is_c4_compatible(S):
            return False
        
        # Check independence constraint
        # The new graph has vertices 0..current_n
        # New vertex is current_n
        # Independent sets containing new vertex: {current_n} U I, where I is independent in G[V \ S]
        # So we need alpha(G[V \ S]) <= 9
        V_set = set(range(current_n)) - set(S)
        
        # Optimization: If |V_set| <= 9, alpha is automatically <= 9.
        if len(V_set) <= 9:
            pass
        else:
            # Check if alpha(G[V_set]) > 9
            # We can use a quick check: if we can find an independent set of size 10, it fails.
            # max_independent_set_size might be slow. 
            # Let's try to find an independent set of size 10 specifically.
            # If found, return False.
            # If not found (and we are confident), return True.
            # Given the time limit, running full MIS B&B on ~30 vertices is feasible.
            alpha_val = max_independent_set_size(V_set)
            if alpha_val > 9:
                return False
            
        # Valid! Add the vertex
        new_v = current_n
        current_adj.append(set(S))
        for u in S:
            current_adj[u].add(new_v)
            edges.append([u, new_v])
        current_n += 1
        return True

    # Greedy addition loop
    # We want to add vertices up to 43
    # We will try multiple times to find a valid S
    
    for _ in range(43 - current_n):
        added = False
        # Try to find a valid S
        
        found = False
        # Increase attempts and use different strategies
        for attempt in range(100): 
            random.seed(1000 + attempt)
            verts = list(range(current_n))
            random.shuffle(verts)
            
            S = set()
            union_N = set()
            
            # Greedily build S
            for v in verts:
                # Check if N(v) is disjoint from union_N
                # Note: N(v) in the *current* graph.
                # The new vertex will be connected to S.
                # For C4-freeness, any two vertices in S must have disjoint neighborhoods in the *existing* graph.
                # Also, the new vertex's neighbors are S.
                # C4 condition: No two vertices in the graph share 2 neighbors.
                # If u, v in S, they share the new vertex as a neighbor.
                # So N(u) and N(v) in the *old* graph must be disjoint.
                # This is exactly what is_c4_compatible checks (intersection of old neighbors).
                
                if not current_adj[v].intersection(union_N):
                    S.add(v)
                    union_N.update(current_adj[v])
                    
            # Try adding with this S
            if try_add_vertex(S):
                added = True
                found = True
                break
            
            # If failed, try a slightly different S?
            # The greedy approach might pick a "bad" v early that blocks many others.
            # We can try to remove the last added vertex and skip it?
            # Or just rely on the next random shuffle.
            
        if not added:
            # If we couldn't add a vertex with simple greedy, 
            # try a more exhaustive search for S?
            # Given time constraints, we might just break.
            # But let's try one more thing: 
            # Sometimes a *smaller* S is valid even if the maximal one isn't?
            # No, if S_max is C4-compatible, any subset is.
            # If alpha(G \ S_max) > 9, it's possible that alpha(G \ S_subset) <= 9?
            # G \ S_subset is a SUPERSET of G \ S_max.
            # So alpha(G \ S_subset) >= alpha(G \ S_max).
            # So if the maximal S fails, subsets will also fail.
            # Thus, we need a DIFFERENT maximal S, or a non-maximal S that is "better" in terms of structure?
            # Wait. If S1 is a subset of S2, then V\S1 is a superset of V\S2.
            # So alpha(V\S1) >= alpha(V\S2).
            # We want alpha <= 9.
            # So we want V\S to be small/dense.
            # We want S to be LARGE.
            # So maximal C4-compatible sets are the best candidates.
            # If all maximal C4-compatible sets fail, we cannot add a vertex.
            
            break

    return {'n': current_n, 'edges': edges}
