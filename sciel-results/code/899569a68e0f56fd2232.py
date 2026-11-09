import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute adjacency for each vertex as a bitmask
    # adj_mask[v] will be a bitmask of neighbors
    # We will maintain this dynamically.
    
    # Precompute "conflict" logic.
    # An edge (u, v) is invalid if there exists s, t such that:
    # 1. deg(s) == deg(t)
    # 2. There is a path s-u-v-t (or s-v-u-t)
    # This implies s is adjacent to u (or v), t is adjacent to v (or u).
    # And s != t, s != u, s != v, t != u, t != v.
    
    # Let's stick to the dynamic check but make it faster.
    # The previous check was:
    # for d, verts in deg_map.items():
    #   for s, t in pairs(verts):
    #     for u in adj[s]:
    #       for v in adj[u]:
    #         if t in adj[v]: return True
    #
    # We can precompute for each pair (s, t) with same degree, the set of "middle edges" (u, v) that complete a path s-u-v-t.
    # However, degrees change as we add edges. So we can't fully precompute.
    
    # Alternative: 
    # When we add edge (u, v), we only need to check if this NEW edge creates a violation.
    # A violation is created if there exists s, t with deg(s)==deg(t) such that (u, v) is part of a path s-u-v-t or s-v-u-t.
    # Actually, adding (u, v) changes degrees of u and v.
    # 1. It might make u and v have the same degree as some other pair? No, it increases deg(u) and deg(v) by 1.
    # 2. It might create NEW paths.
    #    - Paths using (u, v) as the middle edge: s-u-v-t or s-v-u-t.
    #    - Paths where (u, v) is an endpoint edge? e.g. s-u-v-w? No, the path is s-u-v-t. The middle edge is (u, v).
    #    - What if (u, v) is the first edge? s-u-v-t. If we add (u, v), it could be the edge (s, u) or (u, v) or (v, t).
    #    - If (u, v) is the middle edge: we need s ~ u, t ~ v, s != t, s, t not in {u, v}. And deg(s) == deg(t).
    #    - If (u, v) is the first edge: path u-v-w-x? No, the path is s-u-v-t. The edges are (s,u), (u,v), (v,t).
    #      So adding (u, v) could be:
    #      a) The middle edge (u, v). Then we need s ~ u and t ~ v.
    #      b) The first edge (s, u) where s=u? No, s is endpoint. If we add (u, v), it could be edge (s, u) in path s-u-v-t?
    #         If the edge added is (u, v), and it corresponds to (s, u) in the path, then s=u? No, s is a vertex.
    #         Let's clarify: The path is a sequence of 4 distinct vertices s, a, b, t. Edges (s,a), (a,b), (b,t).
    #         If we add edge (u, v), it can be:
    #         1. (s, a) -> s=u, a=v. Then path is u-v-b-t. Need v ~ b, b ~ t. And deg(u) == deg(t).
    #         2. (a, b) -> a=u, b=v. Then path is s-u-v-t. Need s ~ u, v ~ t. And deg(s) == deg(t).
    #         3. (b, t) -> b=u, t=v. Then path is s-a-u-v. Need s ~ a, a ~ u. And deg(s) == deg(v).
    #         
    #         Note: Degrees are in the FULL graph INCLUDING the new edge.
    #         So when checking validity of adding (u, v), we must use deg_new.
    #         deg_new[u] = deg[u] + 1
    #         deg_new[v] = deg[v] + 1
    #         deg_new[x] = deg[x] for x != u, v.

    # Let's implement a fast `is_valid_addition` that checks these 3 cases.
    
    def build_adj_deg(edge_set):
        adj = [0] * n  # bitmask adjacency
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def check_violation_for_add(adj, deg, u, v):
        """
        Check if adding edge (u, v) to the graph defined by adj/deg creates a violation.
        adj/deg represent the graph BEFORE adding (u, v).
        """
        # New degrees
        deg_u_new = deg[u] + 1
        deg_v_new = deg[v] + 1
        
        # We need to check if there exists s, t with deg_new(s) == deg_new(t)
        # such that a path s-a-b-t exists using the new edge (u, v).
        
        # Case 1: (u, v) is the middle edge (a, b). Path s-u-v-t.
        # s ~ u, v ~ t. s != t, s, t not in {u, v}.
        # deg_new(s) == deg_new(t).
        # s is a neighbor of u (in old graph). t is a neighbor of v (in old graph).
        # Note: s cannot be v (since s~u and v~u would imply triangle? No, s~u is edge. If s=v, then v~u is the edge we are adding? No, s~u is an existing edge. If s=v, then v~u is existing? No, we are adding (u,v). So s cannot be v because (u,v) is not in old graph. Similarly t cannot be u.)
        # So s in adj[u] (old), t in adj[v] (old).
        # s != t.
        # deg_new(s) == deg_new(t).
        # If s == u or s == v? 
        #   If s == u: path u-u-v-t? No, vertices must be distinct. s, u, v, t distinct.
        #   So s not in {u, v, t}.
        #   Since s ~ u, s != u.
        #   Since s ~ u and (u,v) is new, s != v (unless there's another edge? No, simple graph). 
        #   Wait, if s=v, then v~u must be an edge in old graph. But we are adding (u,v). So (u,v) is NOT in old graph. So v is not in adj[u]. So s != v automatically.
        #   Similarly t != u automatically.
        #   So we just need s != t.
        
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        
        # Iterate over s in neighbors_u
        # For each s, we need to find t in neighbors_v such that:
        # 1. t != s
        # 2. deg_new(s) == deg_new(t)
        
        # To speed this up, we can group neighbors of v by their new degree.
        # But degrees change for u and v.
        # deg_new(x) = deg[x] + (1 if x in {u,v} else 0)
        
        # Let's just iterate. n=15 is small.
        # Worst case: deg(u)=14, deg(v)=14. 14*14 = 196 checks. Very fast.
        
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            # s is a candidate for start
            deg_s_new = deg[s] + (1 if s == u or s == v else 0)
            # We need t in neighbors_v such that t != s and deg_new(t) == deg_s_new
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_s_new == deg_t_new:
                    # Check if s, u, v, t are distinct
                    # s != u (since s~u in simple graph)
                    # s != v (since s~u and (u,v) not in old graph, so v not in adj[u])
                    # t != u (since t~v and (u,v) not in old graph, so u not in adj[v])
                    # t != v (since t~v in simple graph)
                    # s != t (checked)
                    # So distinctness is guaranteed by simple graph properties and s!=t.
                    return True

        # Case 2: (u, v) is the first edge (s, a). Path u-v-b-t.
        # s=u, a=v.
        # Need v ~ b, b ~ t.
        # deg_new(u) == deg_new(t).
        # b in adj[v] (old). t in adj[b] (old).
        # u, v, b, t distinct.
        # u != v (given).
        # u != b? b ~ v. If b=u, then u~v in old graph. But we are adding (u,v). So u not in adj[v]. So b != u.
        # u != t? If t=u, then b~u. Path u-v-b-u? No, t is endpoint. s=u, t=u? No, s!=t.
        # So t != u.
        # v != b (simple graph).
        # v != t? If t=v, then b~v. Path u-v-b-v? No, t=v, s=u. u!=v.
        #   But t is endpoint. If t=v, then deg_new(t) = deg_new(v).
        #   Is path u-v-b-v valid? Vertices u, v, b, v. Not distinct (v repeated).
        #   So t != v.
        #   How to ensure t != v?
        #   t in adj[b]. If t=v, then v in adj[b] => b in adj[v].
        #   So we just need to check t != v.
        
        deg_u_new = deg[u] + 1
        
        for b in range(n):
            if not (neighbors_v & (1 << b)):
                continue
            # b is candidate for middle vertex
            neighbors_b = adj[b]
            for t in range(n):
                if not (neighbors_b & (1 << t)):
                    continue
                if t == u or t == v:
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_u_new == deg_t_new:
                    # Check distinctness: u, v, b, t
                    # u != v
                    # u != b (b in adj[v], u not in adj[v])
                    # u != t (checked)
                    # v != b (simple)
                    # v != t (checked)
                    # b != t (simple)
                    return True

        # Case 3: (u, v) is the last edge (b, t). Path s-a-u-v.
        # b=u, t=v.
        # Need s ~ a, a ~ u.
        # deg_new(s) == deg_new(v).
        # a in adj[u] (old). s in adj[a] (old).
        # s, a, u, v distinct.
        # s != u? s in adj[a], a in adj[u].
        #   If s=u, then u in adj[a] => a in adj[u].
        #   So we need s != u.
        # s != v? If s=v, then v in adj[a].
        # a != u (simple).
        # a != v? If a=v, then v in adj[u]. But (u,v) not in old graph. So v not in adj[u]. So a != v.
        # u != v.
        # u != s (checked).
        # v != s (checked).
        # v != a (checked).
        
        deg_v_new = deg[v] + 1
        
        for a in range(n):
            if not (neighbors_u & (1 << a)):
                continue
            neighbors_a = adj[a]
            for s in range(n):
                if not (neighbors_a & (1 << s)):
                    continue
                if s == u or s == v:
                    continue
                deg_s_new = deg[s] + (1 if s == u or s == v else 0)
                if deg_v_new == deg_s_new:
                    return True

        return False

    def is_valid(edge_set):
        adj, deg = build_adj_deg(edge_set)
        return True # We don't use this, we use incremental checks

    # Base: Complete bipartite K_{7,8}
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)
    
    # Initial adj/deg for best
    best_adj, best_deg = build_adj_deg(best_edges)
    
    # Greedy add with multiple random orderings
    # We will maintain adj/deg for the current set
    for attempt in range(100): # Reduced attempts since each is faster? No, each attempt is full greedy.
        if time.time() > deadline - 10:
            break
        current_edges = set(base_edges)
        current_adj, current_deg = build_adj_deg(current_edges)
        
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        # We need to be careful: when we add an edge, we update adj/deg.
        # But if we fail to add, we don't change.
        # So we can just iterate.
        
        for edge in shuffled:
            if edge in current_edges:
                continue
            u, v = edge
            if check_violation_for_add(current_adj, current_deg, u, v):
                continue
            # Add edge
            current_edges.add(edge)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_deg[u] += 1
            current_deg[v] += 1
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            best_adj, best_deg = current_adj, current_deg

    # Local search: remove one edge, greedily add others
    for ls_attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Pick a random edge to remove? Or iterate?
        # Let's iterate over edges in best_edges
        # To speed up, we can just try a few random edges or all if fast enough.
        # n=15, max edges ~100. 100 * (greedy add) might be slow if greedy add is O(N^2).
        # But greedy add is O(N * check). Check is O(N^2). So O(N^3). 100 * 15^3 = 337,500. Fast.
        
        improved = False
        for edge_to_remove in best_edges:
            if time.time() > deadline - 3:
                break
                
            u_rm, v_rm = edge_to_remove
            
            # Start from best_edges minus edge_to_remove
            current_edges = set(best_edges)
            current_edges.remove(edge_to_remove)
            current_adj, current_deg = build_adj_deg(current_edges)
            
            # Greedily add edges
            # We can prioritize edges that were removed or random order
            # Let's use random order
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            
            for edge in shuffled:
                if edge in current_edges:
                    continue
                u, v = edge
                if check_violation_for_add(current_adj, current_deg, u, v):
                    continue
                current_edges.add(edge)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_deg[u] += 1
                current_deg[v] += 1
                
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                best_adj, best_deg = current_adj, current_deg
                improved = True
                break # Restart local search with new best? Or continue?
                # Breaking allows us to take the improvement and maybe try more removals in next iteration
                # Or we can just continue to see if we can improve further from this new state?
                # Given time constraints, breaking and restarting the outer loop is safer.
                
        if not improved:
            # No improvement found in this pass?
            # We could try more aggressive moves or just stop if no improvement for several rounds?
            # For now, just continue to next ls_attempt with new random shuffle
            pass

    out = [[u, v] for u, v in best_edges]
    return out
