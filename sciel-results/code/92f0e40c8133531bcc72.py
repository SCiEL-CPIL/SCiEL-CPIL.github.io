import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n = 16
    
    # Precompute pairs in a specific order to help backtracking
    # We process edges in a fixed order. A good order might be by vertex index.
    # Let's use the standard lexicographical order.
    pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    num_pairs = len(pairs)
    
    # Bitmask utilities for fast triangle-free check
    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True
        
    def is_3_colorable(adj):
        # Fast 3-colorability check using backtracking with bitsets
        # Order vertices by degree descending to prune early
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        # Precompute adjacency masks for faster forbidden color calculation
        # Actually, just iterating neighbors is fast enough for n=16
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Find forbidden colors
            forbidden = 0
            adj_v = adj[v]
            # Iterate through set bits in adj_v
            # Or just loop u in 0..n
            for u in range(n):
                if (adj_v >> u) & 1:
                    if colors[u] != -1:
                        forbidden |= (1 << colors[u])
            
            # Try available colors
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def try_add_edge(adj, u, v):
        """Returns True if edge (u,v) can be added while keeping triangle-free."""
        if (adj[u] >> v) & 1:
            return False
        if adj[u] & adj[v]:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Strategy:
    # 1. Use backtracking to construct the graph edge by edge.
    # 2. At each step, decide whether to include the current edge or not.
    # 3. Pruning:
    #    a. Triangle-free constraint: If adding edge (u,v) creates a triangle, skip it.
    #    b. 3-colorability constraint: If the current partial graph (plus any edges we *must* add to reach a certain density? No, we are maximizing, so we add as many as possible) is 3-colorable, we should try to add more edges.
    #       Wait, the constraint is that the FINAL graph must NOT be 3-colorable.
    #       During construction, if the current graph is 3-colorable, it's not "invalid" yet, but it means we haven't reached the 4-chromatic state.
    #       However, if we are maximizing edges, we want to add edges. Adding edges makes it HARDER to 3-color.
    #       So, if the current graph is 3-colorable, we should prefer to ADD edges.
    #       If the current graph is NOT 3-colorable, we have achieved the chromatic number >= 4. We can still add edges if they don't create triangles.
    #       
    #    The key insight for backtracking:
    #    We want to find a maximal triangle-free graph that is not 3-colorable with the maximum number of edges.
    #    
    #    Let's reformulate:
    #    We iterate through edges. For each edge, we try to include it.
    #    We maintain the current adjacency list.
    #    We also maintain a "lower bound" on the number of edges we can still add?
    #    
    #    Actually, a simpler approach:
    #    Since n=16 is small, the total number of edges is 120.
    #    We can use a recursive search that tries to include edges.
    #    To maximize edges, we should try to include an edge if it doesn't violate triangle-free.
    #    But we must ensure the final graph is not 3-colorable.
    #    
    #    This is tricky because "not 3-colorable" is a global property.
    #    However, if we build a graph that is maximal triangle-free, it is likely to be dense.
    #    The maximum number of edges in a triangle-free graph on 16 vertices is floor(16^2/4) = 64 (Turan graph T_2(16)).
    #    T_2(16) is bipartite, hence 2-colorable. So we can't just take the Turan graph.
    #    We need to "break" the bipartiteness or colorability while staying triangle-free.
    #    
    #    Let's use a different strategy:
    #    1. Start with the complete bipartite graph K_{8,8} (64 edges). It is triangle-free and 2-colorable.
    #    2. We need to make it not 3-colorable.
    #    3. We can remove some edges from K_{8,8} and add some edges within the partitions?
    #       No, adding edges within a partition creates triangles if the other partition has edges connecting to both endpoints.
    #       In K_{8,8}, every vertex in A is connected to every vertex in B.
    #       If we add an edge (u,v) within A, and there is any w in B, then (u,v,w) is a triangle.
    #       So we cannot add any edge within A or B to K_{8,8}.
    #       Therefore, any triangle-free graph on 16 vertices with > 64 edges is impossible. Max is 64.
    #       Since T_2(16) is the unique extremal graph (for n even), any triangle-free graph with 64 edges is isomorphic to K_{8,8} and thus 2-colorable.
    #       So we must have < 64 edges.
    #       
    #    So the maximum possible edge count is at most 63.
    #    We need to find a triangle-free graph with 63 edges (or less) that is not 3-colorable.
    #    
    #    How to construct such a graph?
    #    Start with K_{8,8}. Remove one edge (u,v) where u in A, v in B.
    #    Now we have 63 edges. It is still 2-colorable (bipartite).
    #    Can we add an edge within A?
    #    Suppose we add (u1, u2) in A.
    #    For this to be triangle-free, u1 and u2 must have NO common neighbors in B.
    #    In K_{8,8} minus (u,v), u1 is connected to all B \ {v if u1==u?}.
    #    If u1 != u, u1 is connected to all 8 vertices in B.
    #    If u2 != u, u2 is connected to all 8 vertices in B.
    #    They share all 8 neighbors in B. So we cannot add (u1, u2).
    #    
    #    So we must remove MORE edges from K_{8,8} to create a "gap" in the neighborhoods.
    #    Specifically, we need two vertices u1, u2 in A such that N(u1) ∩ N(u2) = ∅ in the remaining graph.
    #    Since |B|=8, if N(u1) and N(u2) are disjoint subsets of B, then |N(u1)| + |N(u2)| <= 8.
    #    This implies we must remove at least 8 - |N(u1)| edges from u1 and 8 - |N(u2)| edges from u2.
    #    To maximize total edges, we want |N(u1)| + |N(u2)| to be as large as possible, i.e., 8.
    #    So we partition B into B1 and B2. u1 connects to B1, u2 connects to B2.
    #    Then we can add edge (u1, u2).
    #    This adds 1 edge but requires removing (8-|B1|) + (8-|B2|) = 16 - 8 = 8 edges.
    #    Net change: -7 edges.
    #    So we go from 64 to 57 edges.
    #    Is this graph 3-colorable?
    #    The graph is K_{8,8} minus a complete bipartite subgraph between {u1,u2} and the "missing" neighbors, plus edge (u1,u2).
    #    Actually, let's analyze the chromatic number.
    #    The rest of the graph is still very dense bipartite.
    #    Likely still 3-colorable or even 2-colorable?
    #    If we add an edge within A, the graph is no longer bipartite. It has an odd cycle?
    #    Path u1 - b - u2 - u1 is a triangle? No, u1-u2 is edge, u1-b, u2-b. If b is neighbor of both, it's a triangle. But we ensured N(u1) ∩ N(u2) = ∅.
    #    So no triangle.
    #    But it has an odd cycle of length 5? u1 - u2 - b2 - a_other - b1 - u1?
    #    Probably not 2-colorable.
    #    Is it 3-colorable?
    #    This construction is getting complex.
    
    #    Let's go back to the "Subtractive" idea from the prompt but refined.
    #    Or better, use the **Backtracking** approach suggested in the directive.
    
    #    We will perform a depth-first search over the space of triangle-free graphs.
    #    We want to maximize edges.
    #    We will try to add edges in a specific order.
    #    To ensure we find a non-3-colorable graph, we can check 3-colorability at the end.
    #    But that's inefficient.
    #    
    #    Alternative:
    #    We know the max edges is 63.
    #    We can try to find a graph with 63, 62, ... edges that is not 3-colorable.
    #    For a fixed number of edges M, we can use random sampling + local search to find a triangle-free graph with M edges, and check if it's 3-colorable.
    #    Since the space is small, we can afford to run many restarts.
    
    #    Let's combine:
    #    1. Generate random triangle-free graphs with high edge count (near 64).
    #    2. Check if 3-colorable.
    #    3. If not, record it.
    #    4. If yes, try to modify it (remove/add edges) to break 3-colorability while keeping edge count high.
    
    #    How to generate high-edge triangle-free graphs?
    #    Start with K_{8,8}.
    #    Randomly remove edges.
    #    Then try to add edges within partitions if possible.
    
    best_edges_count = 0
    best_edge_list = []
    
    def try_candidate(adj):
        nonlocal best_edges_count, best_edge_list
        if not is_triangle_free(adj):
            return
        if is_3_colorable(adj):
            return
        ec = edge_count(adj)
        if ec > best_edges_count:
            best_edges_count = ec
            best_edge_list = get_edge_list(adj)
    
    # Initialization: K_{8,8}
    def make_k88():
        adj = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    # Local search to maximize edges while staying triangle-free and NOT 3-colorable
    def local_search_non3(adj, time_limit=5.0):
        current_adj = adj[:]
        current_edges = edge_count(current_adj)
        
        # If it's 3-colorable, we need to break it.
        # We can do this by removing edges to allow adding intra-partition edges.
        
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit:
            improved = False
            
            # Strategy A: If not 3-colorable, try to add edges.
            if not is_3_colorable(current_adj):
                # Try to add a random missing edge
                missing = [(u, v) for u in range(n) for v in range(u+1, n) if not ((current_adj[u] >> v) & 1)]
                if missing:
                    random.shuffle(missing)
                    for u, v in missing[:20]:
                        if not (current_adj[u] & current_adj[v]):
                            # Try adding
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
                            new_edges = current_edges + 1
                            # It must remain not 3-colorable (adding edges preserves non-3-colorability)
                            # So it's valid if triangle-free.
                            if new_edges > best_edges_count:
                                best_edges_count = new_edges
                                best_edge_list = get_edge_list(current_adj)
                            improved = True
                            current_edges = new_edges
                            break
                    if not improved:
                        # Rollback if we didn't break? No, we only add if we break.
                        # If we tried and failed, we didn't modify.
                        pass
            else:
                # It is 3-colorable. We need to make it not.
                # This requires a structural change.
                # Pick a random edge (u,v) to remove.
                edges_list = [(u, v) for u in range(n) for v in range(u+1, n) if (current_adj[u] >> v) & 1]
                if not edges_list:
                    break
                random.shuffle(edges_list)
                
                # Try removing one edge and then greedily adding others
                for u, v in edges_list[:5]:
                    # Remove (u,v)
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    temp_edges = current_edges - 1
                    
                    # Now try to add intra-partition edges?
                    # This is hard to target.
                    # Instead, just do a greedy fill on the resulting graph.
                    # But greedy fill might make it 3-colorable again?
                    # Actually, if we remove an edge from a 3-colorable graph, it remains 3-colorable.
                    # So removing edges from a 3-colorable graph keeps it 3-colorable.
                    # We must ADD edges to break 3-colorability.
                    # But we can only add edges if they don't create triangles.
                    # In a dense bipartite-like graph, we can't add intra-partition edges unless we remove inter-partition edges first.
                    
                    # So the move is: Remove a set of inter-partition edges to allow an intra-partition edge.
                    # This is a 2-step move: Remove (u,v), Add (u', v').
                    
                    # Let's identify if u and v are in different "parts" (bipartition).
                    # We don't know the bipartition explicitly, but we can assume the structure is close to bipartite.
                    
                    # Simpler: Just random restarts with K_{8,8} perturbations.
                    pass
                    
                    # Rollback
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges = temp_edges + 1
                    
            if not improved:
                break
                
        return current_adj, current_edges

    # Main Loop
    # We will try to construct graphs close to K_{8,8} that are not 3-colorable.
    
    # 1. Try K_{8,8} with specific perturbations.
    # 2. Random restarts.
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        # Start with K_{8,8}
        adj = make_k88()
        
        # Randomly remove some edges to create space for intra-partition edges
        # We need to remove enough edges from two vertices in A to make their neighborhoods disjoint.
        # Let's pick two random vertices in A: a1, a2.
        a1, a2 = random.sample(range(8), 2)
        # Pick a random partition of B: B1, B2
        b_perm = list(range(8, 16))
        random.shuffle(b_perm)
        split = random.randint(1, 7)
        B1 = set(b_perm[:split])
        B2 = set(b_perm[split:])
        
        # Remove edges from a1 to B2 and a2 to B1.
        # This ensures N(a1) subset B1, N(a2) subset B2.
        # N(a1) and N(a2) are disjoint.
        # We can now add edge (a1, a2).
        
        # Apply removals
        for b in B2:
            adj[a1] &= ~(1 << b)
            adj[b] &= ~(1 << a1)
        for b in B1:
            adj[a2] &= ~(1 << b)
            adj[b] &= ~(1 << a2)
            
        # Add edge (a1, a2)
        adj[a1] |= (1 << a2)
        adj[a2] |= (1 << a1)
        
        # Now we have a triangle-free graph?
        # Check:
        # Edges within A: only (a1, a2).
        # N(a1) in B is B1. N(a2) in B is B2.
        # B1 and B2 are disjoint.
        # So no triangle involving (a1, a2) and a vertex in B.
        # Other edges in A? None.
        # Other edges in B? None.
        # So it is triangle-free.
        
        # Is it 3-colorable?
        # The graph is K_{8,8} minus (A1xB2 U A2xB1) plus (a1,a2).
        # Actually, we removed a1-B2 and a2-B1.
        # a1 is connected to B1. a2 is connected to B2.
        # Other A vertices are connected to all B.
        # This graph is likely 3-colorable?
        # Let's check.
        # If it's 3-colorable, we try to add more edges or perturb further.
        
        # Let's run local search to add more edges if possible.
        # But first, check if it's a candidate.
        try_candidate(adj)
        
        # Local search to maximize edges
        # We can try to add more intra-partition edges if we remove more inter-partition edges.
        # Or add edges in B?
        # To add an edge in B, say (b1, b2), we need N(b1) ∩ N(b2) = ∅ in A.
        # N(b1) contains all A except a1 (if b1 in B2) or a2 (if b1 in B1)?
        # If b1 in B1, it is not connected to a2. It is connected to a1 and all other A.
        # If b2 in B2, it is not connected to a1. It is connected to a2 and all other A.
        # N(b1) ∩ N(b2) contains all A \ {a1, a2}.
        # So they share neighbors. Cannot add (b1, b2).
        
        # So this specific construction is "maximal" in terms of adding simple intra-partition edges?
        # Maybe we can add more by removing more.
        
        # Let's just do a generic local search on this adj.
        # We want to add edges.
        # We'll try to add any edge that doesn't create a triangle.
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1):
                        if not (adj[u] & adj[v]):
                            # Try adding
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            # Check if it became 3-colorable?
                            # Adding edges makes it HARDER to color.
                            # If it was NOT 3-colorable, it stays NOT 3-colorable.
                            # If it WAS 3-colorable, it might stay 3-colorable or become not.
                            # We only accept if it is NOT 3-colorable.
                            if not is_3_colorable(adj):
                                changed = True
                                try_candidate(adj)
                            else:
                                # Rollback
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                
    if not best_edge_list:
        # Fallback: Return a known 4-chromatic triangle-free graph (e.g. Grötzsch + isolated)
        # But we want high edges.
        # Let's just return the best found or a default.
        # If we found nothing, we return an empty list? No, must return valid.
        # Let's construct a Grötzsch graph on 11 vertices and leave 5 isolated.
        adj = [0] * n
        # Grötzsch
        # 0-4 cycle
        for i in range(5):
            u, v = i, (i+1)%5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # 5-9 copies
        for i in range(5):
            u = 5 + i
            v1 = (i-1)%5
            v2 = (i+1)%5
            adj[u] |= (1 << v1)
            adj[v1] |= (1 << u)
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
        # 10 top
        for i in range(5):
            u = 10
            v = 5 + i
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        best_edge_list = get_edge_list(adj)
        best_edges_count = edge_count(adj)

    return best_edge_list

# EVOLVE-BLOCK-END
