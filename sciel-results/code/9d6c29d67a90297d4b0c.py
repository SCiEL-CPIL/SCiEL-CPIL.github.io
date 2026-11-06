import time
import random
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = 0
    best_graph = None
    
    # Precompute all triples
    triples = list(itertools.combinations(range(n), 3))
    
    def is_valid(adj):
        # adj is list of int bitmasks
        for u, v, w in triples:
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                return False
        return True

    def is_add_valid(adj, u, v):
        """Check if adding edge (u,v) would create a K3,3 violation."""
        # A K3,3 violation involving the new edge (u,v) must have
        # u and v as two of the three vertices on one side, OR
        # u and v on the same side with a third vertex w, OR
        # u on one side and v on the other side
        #
        # Case 1: u and v are on the same side of K3,3
        #   Need a third vertex w such that {u,v,w} share >=3 common neighbors
        #   The common neighbors of u,v,w must include vertices that are
        #   neighbors of both u and v (since w is added)
        #
        # More precisely: after adding (u,v), check if there exists w such that
        # |N(u) ∩ N(v) ∩ N(w)| >= 3
        #
        # Before adding (u,v), N(u) and N(v) don't include each other.
        # After adding, u ∈ N(v) and v ∈ N(u).
        #
        # For a violation with u,v on same side: need w such that
        # (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) ∩ N(w) has size >= 3
        # = (N(u) ∩ N(v) ∩ N(w)) ∪ ({v} ∩ N(w)) ∪ ({u} ∩ N(w)) minus overlaps
        #
        # This is getting complex. Let's use a simpler approach:
        # Only check triples that include both u and v, or one of them.
        
        # Fast path: check all triples involving u or v
        for u2, v2, w2 in triples:
            # Skip if neither u nor v is in this triple
            if u not in (u2, v2, w2) and v not in (u2, v2, w2):
                continue
            
            common = adj[u2] & adj[v2] & adj[w2]
            # Include the new edge contributions
            if u2 == u and v2 == v:
                pass  # u and v are in the triple
            elif u2 == u and w2 == v:
                pass
            elif v2 == u and w2 == v:
                pass
            else:
                # Only one of u,v is in the triple, so the new edge doesn't affect
                # the common neighbor count for this triple (since the edge connects
                # two vertices, and if only one is in the triple, the other's
                # adjacency to the triple members is unchanged)
                # Actually, this is wrong. Let me think again.
                #
                # If only u is in the triple (say u2=u, v2 and w2 are others),
                # then adding edge (u,v) adds v to N(u). So N(u) changes.
                # The common neighbors of {u, v2, w2} now include v if v is
                # adjacent to both v2 and w2.
                # So we DO need to check this.
                pass
            
            # Simple approach: just check all triples, but this is slow.
            # Let's use a targeted approach.
        
        # Better approach: enumerate all possible K3,3 violations that could
        # be created by adding edge (u,v).
        #
        # A K3,3 has two sides of size 3. The new edge (u,v) can be:
        # 1. Within one side: u and v are both on the left side
        #    Then we need a third vertex w on the left, and 3 vertices on the right
        #    that are all adjacent to u, v, and w.
        #
        # 2. Across sides: u on left, v on right (or vice versa)
        #    Then we need 2 more vertices on left (adjacent to all 3 on right including v)
        #    and 2 more on right (adjacent to all 3 on left including u).
        
        # Case 1: u and v on same side
        # Find w such that u, v, w have >=3 common neighbors
        # Common neighbors of u and v (before adding edge) are N(u) ∩ N(v)
        # After adding edge, u is in N(v) and v is in N(u), but u and v are not
        # neighbors of themselves, so this doesn't change N(u) ∩ N(v) for other vertices.
        # Actually, N(u) gains v, N(v) gains u. So N(u) ∩ N(v) gains {u,v} if
        # u is adjacent to v (which it now is) and v is adjacent to u. But u is not
        # in N(u) and v is not in N(v), so u ∉ N(u) ∩ N(v) and v ∉ N(u) ∩ N(v).
        # So N(u) ∩ N(v) is unchanged by adding edge (u,v).
        #
        # So for case 1: need w such that |N(u) ∩ N(v) ∩ N(w)| >= 3
        # (using the old N(u) and N(v), since they don't change for this purpose)
        
        # But wait, we're checking AFTER adding the edge. So let's just do the full check
        # for triples involving u or v.
        
        for w in range(n):
            if w == u or w == v:
                continue
            # Check triple (u, v, w): common neighbors
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                return False
        
        # Case 2: u and v on opposite sides
        # u on left, v on right. Need 2 more on left (a,b) adjacent to v, w1, w2
        # and 2 more on right (w1,w2) adjacent to u, a, b.
        #
        # This is more complex. Let's enumerate:
        # For each pair (a,b) of vertices (a != b, a != u, b != u, a != v, b != v):
        #   Check if a and b are both adjacent to v (they need to be on left, adjacent to all right)
        #   And find w1,w2 such that w1,w2 are adjacent to u,a,b and also to v
        #   Actually, w1,w2 must be adjacent to u,a,b (left side) and one of them is v.
        #   So we need 2 vertices w1,w2 (w1 != w2, w1 != v, w2 != v) such that
        #   w1 and w2 are both adjacent to u, a, b, and v is adjacent to a, b.
        #   And v must be adjacent to a and b (since v is on right, adjacent to all left).
        
        # This is getting very expensive. Let's use a simpler but still fast approach:
        # Just check all triples that include u or v.
        
        for u2, v2, w2 in triples:
            if u in (u2, v2, w2) or v in (u2, v2, w2):
                common = adj[u2] & adj[v2] & adj[w2]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def find_violation(adj):
        for u, v, w in triples:
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                neighbors = [i for i in range(n) if (common >> i) & 1]
                v_set = neighbors[:3]
                u_set = [u, v, w]
                return u_set, v_set
        return None, None

    def repair(adj):
        while True:
            if time.time() > deadline:
                return False
            u_set, v_set = find_violation(adj)
            if u_set is None:
                return True
            candidates = [(u, v) for u in u_set for v in v_set]
            
            best_edge = None
            max_score = -1
            for u, v in candidates:
                score = adj[u].bit_count() + adj[v].bit_count()
                if score > max_score:
                    max_score = score
                    best_edge = (u, v)
            
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
    def improve(adj):
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            edges_to_check = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
            random.shuffle(edges_to_check)
            
            for u, v in edges_to_check:
                if time.time() > deadline:
                    return False
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not is_add_valid(adj, u, v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    improved = True
                    break
        return True

    for seed in range(500):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        p = 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if repair(adj):
            improve(adj)
            
            if is_valid(adj):
                edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_graph = [adj[i] for i in range(n)]
        
        if best_edges > 85:
            break

    if best_graph is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_graph[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
