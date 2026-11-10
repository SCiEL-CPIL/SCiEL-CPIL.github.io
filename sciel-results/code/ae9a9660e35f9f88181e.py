import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135
    n = 20
    vertices = list(range(n))
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Precompute triple membership for faster checks
    # triple_members[t] = (a, b, c)
    # For each vertex v, precompute which triples contain v
    vertex_to_triples = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)
    
    # Convert triples to array for numpy operations
    triples_arr = np.array(triples)  # shape (num_triples, 3)
    
    def popcount(x):
        return x.bit_count()
    
    def is_k33_free(adj_mask):
        # Check all triples
        # adj_mask is list of ints
        # For efficiency, use numpy if possible, but python loop is fast enough for 1140 triples
        # Actually, let's try a vectorized approach for the full check
        # adj_matrix = np.zeros((n, n), dtype=bool)
        # for i in range(n):
        #     for j in range(n):
        #         if adj_mask[i] & (1 << j):
        #             adj_matrix[i, j] = True
        # # Common neighbors of triple (u,v,w) is sum of adj_matrix[u] & adj_matrix[v] & adj_matrix[w]
        # This is too slow to build matrix each time.
        
        # Python loop is fast enough:
        for t in triples:
            u, v, w = t
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return False
        return True
    
    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2
    
    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges
    
    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for t_idx in vertex_to_triples[u]:
            a, b, c = triples[t_idx]
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        
        # Check triples containing v (but not u, to avoid double check, though u!=v so disjoint sets? No, triples can contain both u and v)
        # Actually, if a triple contains both u and v, it's in vertex_to_triples[u] AND vertex_to_triples[v].
        # So we should only check unique triples.
        # Let's just iterate over union of indices.
        # Or simpler: just check all triples that contain u OR v.
        # The previous loop checked all triples with u.
        # Now we need to check triples with v that did NOT contain u.
        for t_idx in vertex_to_triples[v]:
            a, b, c = triples[t_idx]
            if u in (a, b, c):
                continue # Already checked
            
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == v: ma = new_v
            if b == v: mb = new_v
            if c == v: mc = new_v
            
            # u is not in this triple, so no change for u
            if popcount(ma & mb & mc) > 2:
                return False
        
        return True
    
    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            changed = True
        return adj_mask
    
    def find_violation(adj_mask):
        """Find a violating triple (u, v, w) and a common neighbor c."""
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                # Find a common neighbor
                c_list = [x for x in range(n) if common & (1 << x)]
                if c_list:
                    return (u, v, w, c_list[0], common)
        return None
    
    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(100):
            if time.time() > deadline:
                break
            if is_k33_free(adj_mask):
                break
            viol = find_violation(adj_mask)
            if viol is None:
                break
            u, v, w, c, common = viol
            # Remove edge (u, c) to fix
            adj_mask[u] &= ~(1 << c)
            adj_mask[c] &= ~(1 << u)
    
    def local_search(adj_mask, max_iters=500):
        # Phase 1: Repair
        repair(adj_mask)
        
        # Phase 2: Greedy Add
        adj_mask = greedy_add(adj_mask)
        
        # Phase 3: Local Search (Hill Climbing with Swaps)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no direct addition works, try a swap
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & (1 << j)]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            # Try a few swaps
            for i in range(min(20, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove r_u-r_v
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                # Add a_u-a_v
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    # Check if this swap allows adding more edges
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= (1 << r_v)
                    adj_mask[r_v] |= (1 << r_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if not improved:
                # Try removing a random edge to see if it helps (escape local optimum)
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~(1 << r_v)
                    adj_mask[r_v] &= ~(1 << r_u)
                    adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Random initialization
        density = 0.4 + rng.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair and Optimize
        adj = local_search(adj, max_iters=200)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
    if best_adj is None:
        # Fallback: Empty graph
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
