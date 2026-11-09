import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    deadline = start + 118.0  # safety margin under 120s limit
    rng = random.Random(42)

    def verify(edges):
        if not edges:
            return edges
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    def build_greedy(seed_rng, strategy="random"):
        """Build K2,3-free graph using greedy edge addition."""
        adj = [[0] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        if strategy == "random":
            seed_rng.shuffle(all_pairs)
        elif strategy == "degree_order":
            # Sort by sum of degrees (prefer connecting low-degree vertices)
            def pair_key(p):
                u, v = p
                du = sum(adj[u])
                dv = sum(adj[v])
                return (du + dv, u, v)
            all_pairs.sort(key=pair_key)
            seed_rng.shuffle(all_pairs)  # add randomness to ties
        elif strategy == "complement_priority":
            # Prefer pairs that are far apart in some ordering
            all_pairs.sort(key=lambda p: abs(p[0] - p[1]), reverse=True)
            seed_rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if adj[u][v]:
                continue
            
            # Check if adding (u,v) would create K2,3
            valid = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w] and adj[v][w]:
                    # w is a common neighbor of u and v
                    # For each common neighbor w, check if there's another common neighbor
                    # Actually, we need: after adding (u,v), do u and v have 3+ common neighbors?
                    # No - the constraint is: no two vertices have 3+ common neighbors.
                    # Adding edge (u,v) affects: for any pair (x,y) where x is neighbor of u or v,
                    # and y is neighbor of u or v, the common neighbor count might change.
                    # 
                    # Simpler: after adding (u,v), for every pair (a,b) != (u,v),
                    # check if |N(a) ∩ N(b)| > 2.
                    # But that's expensive. Instead:
                    # Adding (u,v) increases common neighbors for:
                    # - (u, w) for each w in N(u): common(u,w) += 1 (v becomes common)
                    #   wait no. v is now neighbor of u. For pair (u, w) where w in N(u),
                    #   v is in N(u) and v in N(w)? Only if w was neighbor of v.
                    #   Actually: common(u, w) = |N(u) ∩ N(w)|. After adding (u,v):
                    #   N(u) gains v. So common(u,w) gains 1 iff v in N(w).
                    #   Similarly for (v, w) where w in N(v): common(v,w) gains 1 iff u in N(w).
                    #   And for pairs not involving u or v: no change.
                    #   Also for (u, v) itself: they now share all their other common neighbors.
                    pass
            
            # Efficient check: 
            # After adding (u,v), the only pairs whose common-neighbor count changes are:
            # 1. (u, v): new count = |N(u) ∩ N(v)| (before addition, minus each other since not yet connected)
            #    Actually before: common(u,v) = |N(u) ∩ N(v)| where neither includes the other.
            #    After: N(u) includes v, N(v) includes u. Common neighbors of u and v = 
            #    {w : w in N(u) and w in N(v)} = old common + possibly nothing new since u not in N(v) before.
            #    Wait, u is now in N(v) and v is now in N(u), but u is not in N(u) and v not in N(v).
            #    So common(u,v) after = |{w != u,v : w in N(u) and w in N(v)}| = old common(u,v)
            #    Hmm, that's the same! Because u is not a neighbor of itself.
            #    
            # Let me think again. The constraint is: for ALL pairs (a,b), |N(a) ∩ N(b)| <= 2.
            # Adding edge (u,v):
            # - For pair (u,v): |N(u) ∩ N(v)|. Before adding, u not in N(v) and v not in N(u).
            #   After: u in N(v), v in N(u). But u is not in N(u) ∩ N(v) since u not in N(u).
            #   v is not in N(u) ∩ N(v) since v not in N(v).
            #   So |N(u) ∩ N(v)| after = |{w: w in N(u)∪{v} and w in N(v)∪{u}}| 
            #   = |{w: w in old_N(u)∪{v} and w in old_N(v)∪{u}}|
            #   w=v: v in old_N(u)∪{v}? yes. v in old_N(v)∪{u}? v in old_N(v)? no (no self-loops). v in {u}? no. So no.
            #   w=u: u in old_N(u)∪{v}? u in old_N(u)? no. u in {v}? no. So no.
            #   w=other: w in old_N(u) and w in old_N(v). So same as before.
            #   Therefore common(u,v) doesn't change!
            #
            # - For pair (u, w) where w in old_N(u), w != v:
            #   |N(u) ∩ N(w)| after. N(u) gained v. So if v in N(w), count increases by 1.
            #   v in N(w) iff adj[v][w] == 1.
            #
            # - For pair (v, w) where w in old_N(v), w != u:
            #   Similarly, if u in N(w), count increases by 1.
            #
            # - For pair (w1, w2) where neither is u or v: no change.
            # - For pair (u, w) where w not in old_N(u) and w != v: 
            #   N(u) gained v. If v in N(w), count increases by 1. But w not in N(u) before.
            #   This is a pair where u and w weren't connected. The common neighbor count could increase.
            #   Wait, we need to check ALL pairs, not just connected ones.
            #   
            # OK so the correct check: adding (u,v) can increase common-neighbor counts for:
            # - Pairs (u, w) for all w != u,v: if v in N(w), then common(u,w) += 1
            # - Pairs (v, w) for all w != u,v: if u in N(w), then common(v,w) += 1
            # 
            # So we need: for all w != u,v with adj[v][w]: common[u][w] + 1 <= 2, i.e., common[u][w] <= 1
            # And: for all w != u,v with adj[u][w]: common[v][w] + 1 <= 2, i.e., common[v][w] <= 1
            
            valid = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w] and common[u][w] >= 2:
                    valid = False
                    break
            if valid:
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w] and common[v][w] >= 2:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add edge
            adj[u][v] = 1
            adj[v][u] = 1
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
                if adj[u][w]:
                    common[v][w] += 1
                    common[w][v] += 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def local_search(edges, max_time=2.0):
        """Local search: try single edge additions and 1-for-2 swaps."""
        if not edges:
            return edges
        
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        
        common = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    c = 0
                    for w in range(n):
                        if adj[u][w] and adj[v][w]:
                            c += 1
                    common[u][v] = c
        
        current_count = len(edges)
        ls_start = time.time()
        
        while time.time() - ls_start < max_time:
            improved = False
            
            # Try all single edge additions
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        continue
                    
                    valid = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[v][w] and common[u][w] >= 2:
                            valid = False
                            break
                    if valid:
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[u][w] and common[v][w] >= 2:
                                valid = False
                                break
                    
                    if valid:
                        adj[u][v] = 1
                        adj[v][u] = 1
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v][w]:
                                common[u][w] += 1
                                common[w][u] += 1
                            if adj[u][w]:
                                common[v][w] += 1
                                common[w][v] += 1
                        current_count += 1
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swaps
            edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
            rng.shuffle(edge_list)
            
            swap_found = False
            for rem_u, rem_v in edge_list:
                if time.time() - ls_start > max_time:
                    break
                
                # Save state
                adj_save = [row[:] for row in adj]
                common_save = [row[:] for row in common]
                
                # Remove edge (rem_u, rem_v)
                adj[rem_u][rem_v] = 0
                adj[rem_v][rem_u] = 0
                for w in range(n):
                    if w == rem_u or w == rem_v:
                        continue
                    if adj_save[rem_v][w]:
                        common[rem_u][w] -= 1
                        common[w][rem_u] -= 1
                    if adj_save[rem_u][w]:
                        common[rem_v][w] -= 1
                        common[w][rem_v] -= 1
                
                # Try to add two edges
                added = 0
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) 
                             if not adj[u][v] and not (u == rem_u and v == rem_v)]
                rng.shuffle(non_edges)
                
                for u, v in non_edges:
                    if time.time() - ls_start > max_time:
                        break
                    
                    valid = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[v][w] and common[u][w] >= 2:
                            valid = False
                            break
                    if valid:
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[u][w] and common[v][w] >= 2:
                                valid = False
                                break
                    
                    if valid:
                        adj[u][v] = 1
                        adj[v][u] = 1
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v][w]:
                                common[u][w] += 1
                                common[w][u] += 1
                            if adj[u][w]:
                                common[v][w] += 1
                                common[w][v] += 1
                        added += 1
                        if added == 2:
                            break
                
                if added == 2:
                    current_count += 1
                    swap_found = True
                    break
                else:
                    # Restore
                    adj[:] = adj_save
                    common[:] = common_save
            
            if not swap_found:
                break
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    result.append([u, v])
        return result

    def final_polish(edges):
        """Final polish: exhaustively try all single edge additions."""
        if not edges:
            return edges
        
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        
        common = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    c = 0
                    for w in range(n):
                        if adj[u][w] and adj[v][w]:
                            c += 1
                    common[u][v] = c
        
        current_count = len(edges)
        improved = True
        
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        continue
                    
                    valid = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[v][w] and common[u][w] >= 2:
                            valid = False
                            break
                    if valid:
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[u][w] and common[v][w] >= 2:
                                valid = False
                                break
                    
                    if valid:
                        adj[u][v] = 1
                        adj[v][u] = 1
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v][w]:
                                common[u][w] += 1
                                common[w][u] += 1
                            if adj[u][w]:
                                common[v][w] += 1
                                common[w][v] += 1
                        current_count += 1
                        improved = True
                        break
                if improved:
                    break
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    result.append([u, v])
        return result

    best = None
    best_e = 0

    # Run multiple greedy builds with different strategies and seeds
    strategies = ["random", "degree_order", "complement_priority"]
    for trial in range(60):
        if time.time() > deadline - 15.0:
            break
        
        strategy = strategies[trial % len(strategies)]
        trial_rng = random.Random(trial * 7919 + 104729)
        edges = build_greedy(trial_rng, strategy)
        edges = verify(edges)
        
        if edges is not None:
            # Local search
            ls_time = min(2.0, max(0.5, (deadline - time.time()) / 20.0))
            improved = local_search(edges, max_time=ls_time)
            improved = verify(improved)
            
            if improved is not None:
                # Final polish
                polished = final_polish(improved)
                polished = verify(polished)
                if polished is not None and len(polished) > best_e:
                    best_e = len(polished)
                    best = polished
            elif len(edges) > best_e:
                best_e = len(edges)
                best = edges

    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
