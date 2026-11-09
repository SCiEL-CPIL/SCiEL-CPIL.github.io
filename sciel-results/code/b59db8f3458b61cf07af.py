import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin
    rng = random.Random(42)

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
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
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        
        # Case 1: (u, v) is the middle edge. Path s-u-v-t.
        # s in N(u), t in N(v), s != t, deg_new(s) == deg_new(t).
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            deg_s_new = deg[s] + (1 if s == u or s == v else 0)
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_s_new == deg_t_new:
                    return True

        # Case 2: (u, v) is the first edge. Path u-v-b-t.
        # b in N(v), t in N(b), t not in {u,v}, deg_new(u) == deg_new(t).
        deg_u_new = deg[u] + 1
        for b in range(n):
            if not (neighbors_v & (1 << b)):
                continue
            neighbors_b = adj[b]
            for t in range(n):
                if not (neighbors_b & (1 << t)):
                    continue
                if t == u or t == v:
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_u_new == deg_t_new:
                    return True

        # Case 3: (u, v) is the last edge. Path s-a-u-v.
        # a in N(u), s in N(a), s not in {u,v}, deg_new(s) == deg_new(v).
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

    def is_valid_graph(edge_set):
        """Check if a complete edge set has no violations."""
        adj, deg = build_adj_deg(edge_set)
        for s in range(n):
            for t in range(n):
                if s == t or deg[s] != deg[t]:
                    continue
                for u in range(n):
                    if u in (s, t) or not (adj[s] & (1 << u)):
                        continue
                    for v in range(n):
                        if v in (s, t, u) or not (adj[u] & (1 << v)):
                            continue
                        if adj[v] & (1 << t):
                            return False
        return True

    # Strategy: Use compatibility graph + branch and bound
    # Two edges are compatible if they can coexist in a valid graph
    # We'll use a heuristic: edges are compatible if adding them to an empty graph doesn't create a violation
    # This is a necessary but not sufficient condition, but gives us a good starting point
    
    # Build compatibility graph: edge i and edge j are compatible if they don't conflict
    # For speed, we'll use a simpler approach: greedy with multiple strategies + local search
    
    # Base: Complete bipartite K_{7,8} - provably valid
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)
    
    # Initial adj/deg for best
    best_adj, best_deg = build_adj_deg(best_edges)
    
    # Multiple greedy strategies with different orderings
    for attempt in range(100):
        if time.time() > deadline - 20:
            break
        
        current_edges = set(base_edges)
        current_adj, current_deg = build_adj_deg(current_edges)
        
        # Different ordering strategies
        if attempt % 4 == 0:
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
        elif attempt % 4 == 1:
            # Prefer edges within parts (intra-part)
            shuffled = sorted(all_edges, key=lambda e: (0 if e[0] < 7 and e[1] < 7 else 1 if e[0] >= 7 and e[1] >= 7 else 2, rng.random()))
        elif attempt % 4 == 2:
            # Prefer edges that connect low-degree vertices
            shuffled = sorted(all_edges, key=lambda e: (current_deg[e[0]] + current_deg[e[1]], rng.random()))
        else:
            # Random
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
        
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

    # Local search: swap edges to improve
    no_improve_count = 0
    max_no_improve = 15
    
    for ls_attempt in range(100):
        if time.time() > deadline - 5:
            break
        if no_improve_count >= max_no_improve:
            break
        
        improved = False
        
        # Try removing one edge and re-greedy
        edges_to_try = best_edges[:]
        rng.shuffle(edges_to_try)
        
        for edge_to_remove in edges_to_try[:30]:  # Limit to 30 removals per iteration
            if time.time() > deadline - 3:
                break
            
            current_edges = set(best_edges)
            current_edges.remove(edge_to_remove)
            current_adj, current_deg = build_adj_deg(current_edges)
            
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            
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
                no_improve_count = 0
                break
        
        # Try 2-edge swaps if no 1-edge improvement
        if not improved:
            for i in range(15):
                if time.time() > deadline - 3:
                    break
                
                if len(best_edges) < 2:
                    break
                e1 = rng.choice(best_edges)
                e2 = rng.choice(best_edges)
                if e1 == e2:
                    continue
                
                current_edges = set(best_edges)
                current_edges.remove(e1)
                current_edges.remove(e2)
                current_adj, current_deg = build_adj_deg(current_edges)
                
                shuffled = all_edges[:]
                rng.shuffle(shuffled)
                
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
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1

    # Final validation
    if not is_valid_graph(best_edges):
        # Fallback to base if something went wrong
        best_edges = base_edges[:]
    
    out = [[int(u), int(v)] for u, v in best_edges]
    return out
