import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n = 31
    m = 91
    
    def count_c4_from_adj(adj):
        A2 = adj @ adj
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = A2[u, v] - adj[u, v]
                if cn >= 2:
                    total += cn * (cn - 1)
        return total // 4
    
    def delta_c4_add_edge(adj, u, v):
        """Compute change in C4 count when adding edge (u,v) to graph."""
        # Adding edge (u,v) changes λ for pairs involving u or v
        # New edge means u and v are now adjacent
        # λ_xy changes for x,y where x or y is u or v
        # Specifically:
        # - λ_uv increases by 0 (they're now adjacent, but λ counts common neighbors, not including each other)
        #   Actually λ_uv = |N(u) ∩ N(v)|. Adding edge (u,v) doesn't change N(u) or N(v) in terms of common neighbors.
        #   Wait, it does: u is now in N(v) and v is now in N(u). But λ_uv counts common neighbors, 
        #   i.e., vertices w ≠ u,v such that w~u and w~v. Adding (u,v) doesn't change this.
        # - For x ≠ u,v: λ_ux changes if v ~ x (then v becomes a common neighbor of u and x... no)
        #   λ_ux = |N(u) ∩ N(x)|. Adding edge (u,v) adds v to N(u). So if v ~ x, then v becomes a common neighbor.
        #   Similarly λ_vx changes if u ~ x.
        # - λ_uw for w: if v~w, λ_uw increases by 1 (v is new common neighbor of u and w)
        # - λ_vw for w: if u~w, λ_vw increases by 1
        # - λ_uv: unchanged (common neighbors don't include u or v themselves)
        
        # Let me compute the delta directly
        # Before: λ values as they are
        # After adding (u,v):
        #   For each w ≠ u,v:
        #     if adj[v,w] == 1: λ_uw increases by 1
        #     if adj[u,w] == 1: λ_vw increases by 1
        #   λ_uv unchanged
        #   Other λ unchanged
        #
        # The C4 count is (1/4) * Σ_{a<b} λ_ab*(λ_ab - 1)
        # Delta = (1/4) * [sum over affected pairs of (new_λ*(new_λ-1) - old_λ*(old_λ-1))]
        
        delta = 0
        # Compute old A2
        # Actually, let's just compute the affected pairs
        
        # Pairs involving u: (u, w) for w ≠ u, w ≠ v
        # Pairs involving v: (v, w) for w ≠ u, w ≠ v
        # Pair (u,v) itself: λ unchanged
        
        # For w in range(n), w != u, w != v:
        for w in range(n):
            if w == u or w == v:
                continue
            # λ_uw: if adj[v,w] then increases by 1
            if adj[v, w]:
                old_lam = 0  # need to compute
                # This is getting complicated. Let me just recompute.
                pass
        
        # This is too complex to do incrementally correctly. Let me just recompute.
        pass
    
    # Given the complexity of incremental updates, let me use a different strategy:
    # Greedy construction: add edges one at a time, picking the edge that minimizes C4 count.
    
    def greedy_construct():
        adj = np.zeros((n, n), dtype=np.int32)
        edges = set()
        
        for k in range(m):
            best_edge = None
            best_c4 = float('inf')
            
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs)
            
            # Try a subset of candidate edges for speed
            candidates = all_pairs[:500] if len(all_pairs) > 500 else all_pairs
            
            for u, v in candidates:
                if (u, v) in edges:
                    continue
                adj[u, v] = 1
                adj[v, u] = 1
                c4 = count_c4_from_adj(adj)
                adj[u, v] = 0
                adj[v, u] = 0
                
                if c4 < best_c4:
                    best_c4 = c4
                    best_edge = (u, v)
            
            if best_edge is None:
                # Fallback: pick random
                for u, v in all_pairs:
                    if (u, v) not in edges:
                        best_edge = (u, v)
                        break
            
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edges.add(best_edge)
        
        return edges, adj
    
    random.seed(42)
    edges, adj = greedy_construct()
    best_c4 = count_c4_from_adj(adj)
    best_edges = set(edges)
    best_adj = adj.copy()
    
    # Local search to improve
    current_edges = set(edges)
    current_adj = adj.copy()
    current_c4 = best_c4
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        edge_list = list(current_edges)
        a, b = random.choice(edge_list)
        
        non_edge = None
        for _ in range(30):
            c = random.randint(0, n-1)
            d = random.randint(0, n-1)
            if c != d:
                ce = (min(c,d), max(c,d))
                if ce not in current_edges:
                    non_edge = ce
                    break
        
        if non_edge is None:
            continue
        
        c, d = non_edge
        
        current_edges.discard((a,b))
        current_edges.add(non_edge)
        current_adj[a, b] = 0
        current_adj[b, a] = 0
        current_adj[c, d] = 1
        current_adj[d, c] = 1
        
        new_c4 = count_c4_from_adj(current_adj)
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_edges = set(current_edges)
                best_adj = current_adj
        else:
            current_edges.discard(non_edge)
            current_edges.add((a,b))
            current_adj[a, b] = 1
            current_adj[b, a] = 1
            current_adj[c, d] = 0
            current_adj[d, c] = 0
    
    return [[u, v] for u, v in sorted(best_edges)]
