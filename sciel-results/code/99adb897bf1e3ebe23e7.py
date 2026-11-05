"""Construct a K2,3-free graph on 20 vertices maximizing edges."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_total = len(all_pairs)  # 190
    
    # Precompute bitmasks for all pairs
    pair_masks = {}
    for u, v in all_pairs:
        mask = 0
        for w in range(n):
            if w != u and w != v:
                mask |= (1 << w)
        pair_masks[(u, v)] = mask
    
    # Also precompute for single vertex exclusion
    single_masks = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                mask = 0
                for w in range(n):
                    if w != u and w != v:
                        mask |= (1 << w)
                single_masks[(u, v)] = mask
    
    def common_count(adj_mask_u, adj_mask_v, exclude=None):
        """Count common neighbors of u and v given their adjacency masks."""
        # adj_mask_u has bits set for neighbors of u
        # adj_mask_v has bits set for neighbors of v
        common = adj_mask_u & adj_mask_v
        if exclude is not None:
            common &= ~(1 << exclude)
        return bin(common).count('1')
    
    def can_add_edge(u, v, adj_masks):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Adding edge (u,v): u gains neighbor v, v gains neighbor u
        # For any w != u,v:
        #   Pair (u,w): common neighbors increase by 1 if w is adjacent to v
        #   Pair (v,w): common neighbors increase by 1 if w is adjacent to u
        #   We need all pairs to have <= 2 common neighbors after addition
        
        # For pair (u, w) where w != u, v:
        #   Current common neighbors of u and w: count of x where adj[u][x] and adj[w][x]
        #   After adding (u,v): if adj[v][w] is true, v becomes a common neighbor of u and w
        #   So new count = old_count + (1 if adj[v][w] else 0)
        #   Need new_count <= 2
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            # Common neighbors of u and w currently
            cnt_uw = common_count(adj_masks[u], adj_masks[w])
            if adj_masks[v] & (1 << w):  # v is adjacent to w, so v becomes common neighbor of u and w
                if cnt_uw > 1:
                    return False
            
            # Check pair (v, w)
            cnt_vw = common_count(adj_masks[v], adj_masks[w])
            if adj_masks[u] & (1 << w):  # u is adjacent to w, so u becomes common neighbor of v and w
                if cnt_vw > 1:
                    return False
        
        # Also need to check that u and v themselves don't have too many common neighbors
        # But adding edge (u,v) doesn't change |N(u) ∩ N(v)| since u∉N(u) and v∉N(v)
        # However, we should verify current common neighbors of u and v is <= 2
        cnt_uv = common_count(adj_masks[u], adj_masks[v])
        if cnt_uv > 2:
            return False
        
        return True
    
    def greedy_construct(random_order=None):
        """Greedy construction with given edge order."""
        if random_order is None:
            random_order = list(all_pairs)
        
        adj_masks = [0] * n
        edges = []
        
        for u, v in random_order:
            if time.time() > deadline - 5.0:
                break
            if (adj_masks[u] >> v) & 1:
                continue
            if can_add_edge(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
        
        return adj_masks, edges
    
    def local_search(adj_masks, edges, max_time=30.0):
        """Improve via local search: try adding edges, or swap operations."""
        start = time.time()
        best_adj = list(adj_masks)
        best_edges = [e[:] for e in edges]
        best_score = len(best_edges)
        
        # First: greedy add all possible edges
        improved = True
        while improved and time.time() - start < max_time:
            improved = False
            for u, v in all_pairs:
                if (best_adj[u] >> v) & 1:
                    continue
                if can_add_edge(u, v, best_adj):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    best_edges.append([u, v])
                    improved = True
                    if len(best_edges) > best_score:
                        best_score = len(best_edges)
        
        # Local search: try removing one edge and adding two
        for iteration in range(200):
            if time.time() - start > max_time:
                break
            
            # Try random edge removal + re-addition
            if best_edges:
                idx = random.randint(0, len(best_edges) - 1)
                u, v = best_edges[idx]
                
                # Temporarily remove
                temp_adj = list(best_adj)
                temp_adj[u] &= ~(1 << v)
                temp_adj[v] &= ~(1 << u)
                temp_edges = [e for i, e in enumerate(best_edges) if i != idx]
                
                # Try to add back plus one more
                found_better = False
                for u2, v2 in all_pairs:
                    if (temp_adj[u2] >> v2) & 1:
                        continue
                    if can_add_edge(u2, v2, temp_adj):
                        temp_adj[u2] |= (1 << v2)
                        temp_adj[v2] |= (1 << u2)
                        temp_edges.append([u2, v2])
                        
                        # Try to add one more
                        for u3, v3 in all_pairs:
                            if (temp_adj[u3] >> v3) & 1:
                                continue
                            if can_add_edge(u3, v3, temp_adj):
                                temp_adj[u3] |= (1 << v3)
                                temp_adj[v3] |= (1 << u3)
                                temp_edges.append([u3, v3])
                                if len(temp_edges) > best_score:
                                    best_adj = list(temp_adj)
                                    best_edges = [e[:] for e in temp_edges]
                                    best_score = len(best_edges)
                                found_better = True
                                break
                        
                        if not found_better and len(temp_edges) > best_score:
                            best_adj = list(temp_adj)
                            best_edges = [e[:] for e in temp_edges]
                            best_score = len(temp_edges)
                        break
                else:
                    # Just try to add back the same edge plus one more
                    if can_add_edge(u, v, temp_adj):
                        temp_adj[u] |= (1 << v)
                        temp_adj[v] |= (1 << u)
                        temp_edges.append([u, v])
                        for u2, v2 in all_pairs:
                            if (temp_adj[u2] >> v2) & 1:
                                continue
                            if can_add_edge(u2, v2, temp_adj):
                                temp_adj[u2] |= (1 << v2)
                                temp_adj[v2] |= (1 << u2)
                                temp_edges.append([u2, v2])
                                if len(temp_edges) > best_score:
                                    best_adj = list(temp_adj)
                                    best_edges = [e[:] for e in temp_edges]
                                    best_score = len(temp_edges)
                                break
        
        return best_adj, best_edges, best_score
    
    # Multiple random restarts with greedy + local search
    best_overall = 0
    best_edges_overall = []
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different random orderings
        random_order = list(all_pairs)
        random.shuffle(random_order)
        
        adj_masks, edges = greedy_construct(random_order)
        
        # Local search
        adj_masks, edges, score = local_search(adj_masks, edges, max_time=6.0)
        
        if score > best_overall:
            best_overall = score
            best_edges_overall = [e[:] for e in edges]
        
        # Also try with different initial structures
        if r % 3 == 0:
            # Start with a random sparse graph, then fill
            sparse_adj = [0] * n
            for u, v in random_order[:20]:
                sparse_adj[u] |= (1 << v)
                sparse_adj[v] |= (1 << u)
            
            adj_masks, edges = greedy_construct(random_order[20:])
            # Merge
            merged_adj = list(adj_masks)
            for u in range(n):
                merged_adj[u] |= sparse_adj[u]
                merged_adj[u] &= ~sparse_adj[u]  # Actually just OR
                merged_adj[u] = adj_masks[u] | sparse_adj[u]
            
            # Rebuild edges from merged_adj
            merged_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (merged_adj[u] >> v) & 1:
                        merged_edges.append([u, v])
            
            # Validate and fix
            valid_adj = [0] * n
            valid_edges = []
            for u, v in sorted(merged_edges, key=lambda e: random.random()):
                if can_add_edge(u, v, valid_adj):
                    valid_adj[u] |= (1 << v)
                    valid_adj[v] |= (1 << u)
                    valid_edges.append([u, v])
            
            if len(valid_edges) > best_overall:
                adj_masks, edges, score = local_search(valid_adj, valid_edges, max_time=4.0)
                if score > best_overall:
                    best_overall = score
                    best_edges_overall = [e[:] for e in edges]
    
    # Final polish
    if best_edges_overall:
        final_adj = [0] * n
        for u, v in best_edges_overall:
            final_adj[u] |= (1 << v)
            final_adj[v] |= (1 << u)
        
        final_adj, final_edges, final_score = local_search(final_adj, best_edges_overall, max_time=10.0)
        
        if final_score > best_overall:
            best_overall = final_score
            best_edges_overall = [e[:] for e in final_edges]
    
    return best_edges_overall
