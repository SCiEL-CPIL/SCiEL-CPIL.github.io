import time
import random

def construct():
    n = 20
    # Deadline: 120s limit, keep 5s margin
    deadline = time.time() + 115
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to get neighbors from mask
    def get_neighbors(mask):
        return [i for i in range(n) if mask & (1 << i)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency list adj (list of masks)
        results in a K2,3-free graph.
        """
        if adj[u] & (1 << v):
            return True # Already exists
            
        # Condition 1: |N(u) cap N(v)| <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # Condition 2: For all w in N(u) cap N(v), 
        # |N(u) cap N(w)| <= 1 and |N(v) cap N(w)| <= 1
        # Note: N(u) and N(v) in the NEW graph include each other, but 
        # the intersection logic derived above used G (old graph).
        # Let's re-verify.
        # New N(u) = Old N(u) U {v}
        # New N(v) = Old N(v) U {u}
        # New N(w) = Old N(w)
        
        # Pair (u, v): |New N(u) cap New N(v)| = |Old N(u) cap Old N(v)|. Checked.
        
        # Pair (u, w) for w in Old N(u):
        # |New N(u) cap New N(w)| = |Old N(u) cap Old N(w)| + (1 if v in Old N(w) else 0)
        # We need this <= 2.
        # Since G was valid, |Old N(u) cap Old N(w)| <= 2.
        # If v not in Old N(w), sum <= 2. OK.
        # If v in Old N(w), sum = |Old N(u) cap Old N(w)| + 1.
        # We need |Old N(u) cap Old N(w)| + 1 <= 2 => |Old N(u) cap Old N(w)| <= 1.
        # So for all w in Old N(u) such that v in Old N(w) (i.e., w in Old N(u) cap Old N(v)),
        # we need |Old N(u) cap Old N(w)| <= 1.
        
        # Pair (v, w) for w in Old N(v):
        # Symmetric: for all w in Old N(v) cap Old N(u), we need |Old N(v) cap Old N(w)| <= 1.
        
        # So we only need to check w in common_uv.
        mask_w = common_uv
        while mask_w:
            # Extract lowest set bit
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            
            # Check |N(u) cap N(w)| <= 1
            if popcount(adj[u] & adj[w]) > 1:
                return False
            # Check |N(v) cap N(w)| <= 1
            if popcount(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def greedy_construct():
        adj = [0] * n
        edges = []
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges

    def local_search(adj, edges):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing edge
            for u, v in all_edges:
                if adj[u] & (1 << v):
                    continue
                if is_valid_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
            
            if time.time() >= deadline:
                break
                
            # 2. Try 1-out 2-in swaps
            # To make this efficient, we can sample or iterate carefully.
            # Iterating all edges (up to ~50) * all pairs of missing edges (huge) is too slow.
            # Strategy: For each edge e, remove it. Then try to add edges greedily.
            # If we add >= 2 edges, keep the swap.
            
            current_edges = edges[:]
            for i, (u1, v1) in enumerate(current_edges):
                if time.time() >= deadline:
                    break
                    
                # Remove edge (u1, v1)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                edges.remove((u1, v1))
                
                added = 0
                new_edges_to_add = []
                
                # Try to add edges back, prioritizing those that might help
                # We can just shuffle all edges and try to add them.
                # This is O(E^2) roughly if we do it naively for every removal.
                # E is small (~40). E^2 is 1600. 40 * 1600 = 64000 checks.
                # Each check is O(N) or O(1) with popcount.
                # This is fast enough in Python? Maybe.
                
                candidate_edges = all_edges[:]
                random.shuffle(candidate_edges)
                
                for u, v in candidate_edges:
                    if (u, v) == (u1, v1) or (u, v) == (v1, u1):
                        continue
                    if adj[u] & (1 << v):
                        continue
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append((u, v))
                        added += 1
                        new_edges_to_add.append((u, v))
                        if added >= 2:
                            break # We found a beneficial swap (net +1)
                            
                if added < 1:
                    # Revert
                    for u, v in new_edges_to_add:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edges.remove((u, v))
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    edges.append((u1, v1))
                else:
                    improved = True
        
        return adj, edges

    best_adj = None
    best_edges = []
    best_count = 0
    
    # Run multiple iterations
    while time.time() < deadline:
        adj, edges = greedy_construct()
        adj, edges = local_search(adj, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
            
    # Final cleanup: ensure best_edges matches best_adj
    # Re-verify validity of best_edges just in case
    final_adj = [0] * n
    for u, v in best_edges:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
        
    # Verify
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            if popcount(final_adj[u] & final_adj[v]) > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a simple valid graph if something went wrong (shouldn't happen)
        # Just return empty or a known small graph
        return []
        
    return [[u, v] for u, v in best_edges]
