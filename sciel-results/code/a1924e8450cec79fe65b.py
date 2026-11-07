"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge list and mapping
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
            edge_map[(u, v)] = idx
            idx += 1
            
    # State: adjacency matrix (boolean) and degrees
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    current_edges = set()
    
    # Fast validity check for adding edge (u, v)
    # A path s-u-v-t exists if s is neighbor of u (s!=v) and t is neighbor of v (t!=u) and s!=t.
    # The constraint is: NO such path can have deg(s) == deg(t).
    # Note: Degrees are measured in the FULL graph *after* addition?
    # "Reject if any simple three-edge path has endpoints of equal degree in the FULL candidate graph."
    # When we add edge (u, v), the degrees of u and v increase by 1.
    # The degrees of s and t do NOT change when we add (u, v) unless s or t is u or v.
    # But s is a neighbor of u, so s != u. t is a neighbor of v, so t != v.
    # However, s could be v? No, s is neighbor of u, s!=v is required for simple path s-u-v-t?
    # Path s-u-v-t. Vertices must be distinct? "simple three-edge path". Usually implies distinct vertices.
    # Prompt: "s != t". It does not explicitly say s != v or t != u, but "simple path" implies no repeated vertices.
    # If s=v, path is v-u-v-t, which repeats v. Not simple.
    # So s is in N(u) \ {v, u}. t is in N(v) \ {u, v}.
    # Also s != t.
    
    # Crucial insight:
    # When we add edge (u, v):
    # 1. deg[u] and deg[v] increase by 1.
    # 2. New paths of length 3 involving the new edge (u, v) as the middle edge are:
    #    s - u - v - t where s in N(u)\{v} and t in N(v)\{u} and s != t.
    # 3. Are there other new paths?
    #    The edge (u, v) can be the first, second, or third edge of the path.
    #    Case A: (u, v) is middle edge. Path s-u-v-t. Checked above.
    #    Case B: (u, v) is first edge. Path u-v-t-x. 
    #            This requires edge (v, t) and (t, x).
    #            This path already existed before adding (u, v) IF (v, t) and (t, x) existed.
    #            But wait, the path is u-v-t-x. The endpoints are u and x.
    #            Before adding (u, v), the path v-t-x existed (length 2).
    #            The path u-v-t-x did NOT exist because (u, v) was missing.
    #            So we must check if deg(u) == deg(x) for any such path.
    #            Note: deg(u) has just increased. deg(x) is unchanged (unless x==v, but x!=v in simple path).
    #    Case C: (u, v) is last edge. Path x-s-u-v.
    #            Requires (x, s) and (s, u).
    #            Endpoints x and v. Check if deg(x) == deg(v).
    #            deg(v) has just increased. deg(x) unchanged.
            
    def can_add(u, v):
        # Temporarily simulate adding edge
        # We need to check all paths that *include* the new edge (u, v).
        # Paths are of length 3 (4 vertices).
        
        # 1. Path s-u-v-t (u, v is middle)
        # s in N(u) \ {v}, t in N(v) \ {u}, s != t
        # Check if deg[s] == deg[t]
        # Note: deg[s] and deg[t] are current degrees (before adding u-v, unless s==u or t==v which is impossible)
        # s is neighbor of u, so s != u. t is neighbor of v, so t != v.
        # So deg[s] and deg[t] are stable.
        for s in range(n):
            if s == u or s == v or not adj[u][s]:
                continue
            for t in range(n):
                if t == u or t == v or t == s or not adj[v][t]:
                    continue
                if deg[s] == deg[t]:
                    return False
        
        # 2. Path u-v-t-x (u, v is first edge)
        # Endpoints u and x.
        # v-t-x is a path of length 2 in current graph.
        # So t in N(v) \ {u}, x in N(t) \ {v, u}.
        # Check if deg(u) + 1 == deg(x)  (since deg(u) increases by 1)
        new_deg_u = deg[u] + 1
        for t in range(n):
            if t == u or not adj[v][t]:
                continue
            for x in range(n):
                if x == u or x == v or x == t or not adj[t][x]:
                    continue
                if new_deg_u == deg[x]:
                    return False
        
        # 3. Path x-s-u-v (u, v is last edge)
        # Endpoints x and v.
        # x-s-u is a path of length 2 in current graph.
        # So s in N(u) \ {v}, x in N(s) \ {u, v}.
        # Check if deg(x) == deg(v) + 1
        new_deg_v = deg[v] + 1
        for s in range(n):
            if s == v or not adj[u][s]:
                continue
            for x in range(n):
                if x == u or x == v or x == s or not adj[s][x]:
                    continue
                if deg[x] == new_deg_v:
                    return False
                    
        return True

    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
        current_edges.add((u, v))

    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
        current_edges.remove((u, v))

    # Strategy:
    # 1. Try to add all edges in a randomized order.
    #    If can_add fails, skip it.
    # 2. This yields a maximal valid graph (no single edge can be added).
    # 3. Try to improve by swapping? Or just stick with greedy?
    #    Greedy might get stuck in a local optimum where removing one edge allows adding three.
    #    However, with n=15, the space is small.
    #    Let's run multiple random restarts of the greedy construction.
    
    best_edge_count = 0
    best_adj = None
    best_deg = None
    best_edges_set = None
    
    # Number of restarts based on time
    # Each restart is O(E * check_cost). E=105. check_cost is small.
    # We can do many restarts.
    
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Reset state
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        current_edges = set()
        
        # Shuffle edges
        edge_list = edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if can_add(u, v):
                add_edge(u, v)
                
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            best_edges_set = current_edges.copy()
            
        # If we found a very dense graph, we might stop early?
        # Max possible is 105.
        # If best_edge_count > 80, it's very good.
        
    # If we have a best graph, let's try a local search to see if we can add more edges
    # by removing some.
    # Actually, the greedy produces a maximal graph. To add more, we must remove.
    # A simple swap: remove e1, try to add e2, e3...
    
    if best_edges_set is not None:
        # Load best state
        adj = [row[:] for row in best_adj]
        deg = best_deg[:]
        current_edges = best_edges_set.copy()
        
        # Local Search: Randomly remove an edge, then try to greedily add back missing edges.
        # If the result is better, keep it.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge in the current graph
            # To save time, we can just try a few random removals
            # Or iterate through all edges? 105 edges * 105 additions * check cost.
            # 105 * 105 * ~100 ops = ~1M ops. Very fast.
            
            # Let's try all single-edge removals
            for u, v in list(current_edges):
                if time.time() > deadline:
                    break
                    
                # Save state
                adj_backup = [row[:] for row in adj]
                deg_backup = deg[:]
                edges_backup = current_edges.copy()
                
                # Remove edge
                remove_edge(u, v)
                
                # Try to add back all missing edges greedily
                missing = [e for e in edges if e not in current_edges]
                random.shuffle(missing) # Random order for tie-breaking
                
                for a, b in missing:
                    if can_add(a, b):
                        add_edge(a, b)
                        
                # Check if we gained edges
                if len(current_edges) > best_edge_count:
                    best_edge_count = len(current_edges)
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_edges_set = current_edges.copy()
                    improved = True
                else:
                    # Restore state
                    adj = adj_backup
                    deg = deg_backup
                    current_edges = edges_backup

    # Construct final result
    result = []
    for u, v in best_edges_set:
        result.append([u, v])
        
    # Ensure types
    # result is list of lists of ints.
    
    return result
# EVOLVE-BLOCK-END
