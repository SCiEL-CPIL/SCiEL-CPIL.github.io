import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples (u, v, w) for K3,3 check
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Precompute for each vertex, which triples contain it
    # triples_containing[v] = list of indices of triples that include v
    triples_containing = [[] for _ in range(n)]
    for idx, (u, v, w) in enumerate(triples):
        triples_containing[u].append(idx)
        triples_containing[v].append(idx)
        triples_containing[w].append(idx)

    def bit(i):
        return 1 << i

    def popcount(x):
        return x.bit_count()

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    edges.append([i, j])
        return edges

    def is_k33_free(adj_mask):
        """Check if graph is K3,3-free."""
        for u, v, w in triples:
            if popcount(adj_mask[u] & adj_mask[v] & adj_mask[w]) > 2:
                return False
        return True

    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = bit(v)
        bit_u = bit(u)
        
        # We need to check if adding (u,v) creates a K3,3.
        # A K3,3 is formed by {u,v,w} on one side and {x,y,z} on the other.
        # If we add (u,v), new common neighbors of any triple containing u or v might increase.
        # Actually, adding an edge (u,v) increases the common neighbor count for triples that contain u or v?
        # No. Adding (u,v) means u and v are now connected.
        # Consider a triple T = {a,b,c}. The common neighbors are N(a) \cap N(b) \cap N(c).
        # If we add (u,v):
        # - If u,v are not in T, N(a), N(b), N(c) don't change directly? 
        #   Wait, N(x) is the set of neighbors of x.
        #   Adding (u,v) adds v to N(u) and u to N(v).
        #   So for any triple T, if T contains u, then N(u) changes.
        #   If T contains v, then N(v) changes.
        #   If T contains neither, N(a), N(b), N(c) are unchanged, so intersection is unchanged.
        
        # So we only need to check triples containing u or v.
        
        # Optimization: 
        # new_N_u = N(u) \cup {v}
        # new_N_v = N(v) \cup {u}
        
        # For a triple (a,b,c):
        # If u in (a,b,c):
        #   Let pos be index of u.
        #   New intersection = new_N_u & N(b) & N(c) (if u is a)
        #   We need popcount(new_intersection) <= 2.
        
        new_N_u = adj_mask[u] | bit_v
        new_N_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for tidx in triples_containing[u]:
            a, b, c = triples[tidx]
            if a == u:
                common = new_N_u & adj_mask[b] & adj_mask[c]
            elif b == u:
                common = adj_mask[a] & new_N_u & adj_mask[c]
            else: # c == u
                common = adj_mask[a] & adj_mask[b] & new_N_u
            
            if popcount(common) > 2:
                return False
        
        # Check triples containing v but not u
        for tidx in triples_containing[v]:
            a, b, c = triples[tidx]
            if a == u or b == u or c == u:
                continue  # Already checked in previous loop
            
            if a == v:
                common = new_N_v & adj_mask[b] & adj_mask[c]
            elif b == v:
                common = adj_mask[a] & new_N_v & adj_mask[c]
            else: # c == v
                common = adj_mask[a] & adj_mask[b] & new_N_v
            
            if popcount(common) > 2:
                return False
                
        return True

    def greedy_fill(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & bit(j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= bit(j)
                            adj_mask[j] |= bit(i)
                            changed = True
        return adj_mask

    def local_search(adj_mask, rng, max_time=None):
        """
        Local search with 1-for-1 and 2-for-2 swaps.
        """
        if max_time is None:
            max_time = deadline - 5
        
        adj_mask = list(adj_mask)
        adj_mask = greedy_fill(adj_mask)
        best_cnt = edge_count(adj_mask)
        best_adj = list(adj_mask)
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Cache missing edges for faster access? 
        # We'll recompute missing edges when needed or maintain a set.
        
        while time.time() < max_time:
            current_cnt = edge_count(adj_mask)
            if current_cnt > best_cnt:
                best_cnt = current_cnt
                best_adj = list(adj_mask)
            
            # Identify missing edges
            missing = [(i, j) for i, j in all_edges if not (adj_mask[i] & bit(j))]
            if not missing:
                break # Complete graph, impossible for K3,3 free but good to check
            
            # Strategy 1: Random 1-for-1 swap
            # Pick a random existing edge to remove, then try to add a random missing edge
            # This is slow. Better: Pick a random missing edge (x,y). Try to remove a random existing edge (u,v) such that adding (x,y) is valid?
            # No, removing (u,v) might make (x,y) valid.
            
            # Let's try: Pick random missing edge (x,y).
            # It is currently invalid. 
            # Find a set of edges to remove such that (x,y) becomes valid.
            # This is complex.
            
            # Simpler: 
            # 1. Remove a random edge (u,v).
            # 2. Try to add (x,y) (the one that was blocking us? No, just any missing edge).
            # 3. Greedy fill.
            
            # Let's try a "kick" move: Remove k random edges, then greedy fill.
            # If we get stuck, we try swaps.
            
            # Try 1-for-1 swap:
            # Pick random missing edge (x,y).
            # Pick random existing edge (u,v).
            # Remove (u,v). Check if (x,y) can be added.
            # If yes, add it. Then greedy fill.
            
            if rng.random() < 0.5:
                # 1-for-1 swap
                if not missing:
                    continue
                x, y = rng.choice(missing)
                # Find existing edges
                existing = [(i, j) for i, j in all_edges if adj_mask[i] & bit(j)]
                if not existing:
                    continue
                u, v = rng.choice(existing)
                
                # Temporarily remove (u,v)
                adj_mask[u] &= ~bit(v)
                adj_mask[v] &= ~bit(u)
                
                # Try to add (x,y)
                if can_add(x, y, adj_mask):
                    adj_mask[x] |= bit(y)
                    adj_mask[y] |= bit(x)
                    adj_mask = greedy_fill(adj_mask)
                else:
                    # Restore (u,v) if we didn't improve? 
                    # Actually, if we can't add (x,y), we might still benefit from removing (u,v) and adding something else via greedy fill.
                    # But greedy fill might not add (x,y) if it's still blocked.
                    # Let's just greedy fill anyway.
                    adj_mask = greedy_fill(adj_mask)
            else:
                # Random removal kick
                existing = [(i, j) for i, j in all_edges if adj_mask[i] & bit(j)]
                if not existing:
                    break
                u, v = rng.choice(existing)
                adj_mask[u] &= ~bit(v)
                adj_mask[v] &= ~bit(u)
                adj_mask = greedy_fill(adj_mask)
                
        return best_adj, best_cnt

    def try_2_for_2_swap(adj_mask, rng):
        """
        Try to remove 2 edges and add 2 different edges.
        """
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        existing = [(i, j) for i, j in all_edges if adj_mask[i] & bit(j)]
        missing = [(i, j) for i, j in all_edges if not (adj_mask[i] & bit(j))]
        
        if len(existing) < 2 or len(missing) < 2:
            return adj_mask, False
        
        # Pick 2 existing edges to remove
        e1, e2 = rng.sample(existing, 2)
        
        # Pick 2 missing edges to add
        # We want to pick missing edges that are "likely" to be valid after removal.
        # Just pick random ones for now.
        m1, m2 = rng.sample(missing, 2)
        
        # Remove e1, e2
        adj2 = list(adj_mask)
        u1, v1 = e1
        u2, v2 = e2
        adj2[u1] &= ~bit(v1)
        adj2[v1] &= ~bit(u1)
        adj2[u2] &= ~bit(v2)
        adj2[v2] &= ~bit(u2)
        
        # Try to add m1, m2
        success = True
        for mx, my in [m1, m2]:
            if not (adj2[mx] & bit(my)):
                if can_add(mx, my, adj2):
                    adj2[mx] |= bit(my)
                    adj2[my] |= bit(mx)
                else:
                    success = False
                    break
        
        if success:
            adj2 = greedy_fill(adj2)
            return adj2, True
        return adj_mask, False

    def local_search_advanced(adj_mask, rng, max_time=None):
        """
        More aggressive local search with 2-for-2 swaps.
        """
        if max_time is None:
            max_time = deadline - 5
            
        adj_mask = list(adj_mask)
        adj_mask = greedy_fill(adj_mask)
        best_cnt = edge_count(adj_mask)
        best_adj = list(adj_mask)
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < max_time:
            current_cnt = edge_count(adj_mask)
            if current_cnt > best_cnt:
                best_cnt = current_cnt
                best_adj = list(adj_mask)
            
            missing = [(i, j) for i, j in all_edges if not (adj_mask[i] & bit(j))]
            if not missing:
                break
            
            # Try 2-for-2 swap
            if rng.random() < 0.3:
                new_adj, success = try_2_for_2_swap(adj_mask, rng)
                if success:
                    adj_mask = new_adj
                    continue
            
            # Standard 1-for-1 or removal
            existing = [(i, j) for i, j in all_edges if adj_mask[i] & bit(j)]
            if not existing:
                break
            
            if rng.random() < 0.5:
                # 1-for-1
                x, y = rng.choice(missing)
                u, v = rng.choice(existing)
                adj_mask[u] &= ~bit(v)
                adj_mask[v] &= ~bit(u)
                if can_add(x, y, adj_mask):
                    adj_mask[x] |= bit(y)
                    adj_mask[y] |= bit(x)
                adj_mask = greedy_fill(adj_mask)
            else:
                # Removal kick
                u, v = rng.choice(existing)
                adj_mask[u] &= ~bit(v)
                adj_mask[v] &= ~bit(u)
                adj_mask = greedy_fill(adj_mask)
                
        return best_adj, best_cnt

    # Initializations
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # 1. Circulant base
    def construct_circulant(diffs):
        adj = [0] * n
        for d in diffs:
            for i in range(n):
                j = (i + d) % n
                if i < j:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        return adj

    # Try various circulant sets
    circulant_sets = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [1, 3, 4, 5],
        [2, 3, 4, 5],
        [1, 2, 3, 6],
        [1, 2, 3, 4, 6],
    ]
    
    for diffs in circulant_sets:
        if time.time() > deadline - 10:
            break
        adj = construct_circulant(diffs)
        adj = greedy_fill(adj)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # Local search on this base
        adj, cnt = local_search_advanced(adj, rng, max_time=deadline - 20)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)

    # 2. Random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        density = 0.4 + rng.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        adj = greedy_fill(adj)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # Local search
        adj, cnt = local_search_advanced(adj, rng, max_time=deadline - 10)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # If we find a very good solution, break early?
        if best_edges > 140:
            break

    # Final polish
    if best_adj is not None and time.time() < deadline - 2:
        adj = list(best_adj)
        adj = greedy_fill(adj)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
    if best_adj is None:
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges

