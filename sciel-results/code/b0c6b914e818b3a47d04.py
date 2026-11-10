import time
import random
from itertools import combinations

def construct():
    n = 16
    # Precompute subsets of size 8
    all_subsets = list(combinations(range(n), 8))
    
    # Map pair (u, v) to list of subsets containing them
    subsets_by_pair = {}
    for S in all_subsets:
        for u, v in combinations(S, 2):
            subsets_by_pair.setdefault((u, v), []).append(S)

    # Precompute bitmasks for each subset for fast degree counting
    subset_masks = {}
    for S in all_subsets:
        m = 0
        for v in S:
            m |= (1 << v)
        subset_masks[S] = m

    def is_q3_candidate(adj, S, u, v):
        """
        Check if adding (u, v) to adj makes G[S] a Q3.
        u, v are in S.
        Assumes (u, v) is NOT yet in adj.
        """
        mask_S = subset_masks[S]
        
        # Degree check:
        # For w in S \ {u, v}, deg(w) in S must be 3.
        # For u, v, deg(w) in S must be 2 (so that deg becomes 3 after adding edge).
        
        deg_u = (adj[u] & mask_S).bit_count()
        if deg_u != 2:
            return False
        deg_v = (adj[v] & mask_S).bit_count()
        if deg_v != 2:
            return False
            
        for w in S:
            if w == u or w == v:
                continue
            deg_w = (adj[w] & mask_S).bit_count()
            if deg_w != 3:
                return False
                
        # If degrees match, check bipartiteness of H = G[S] + (u, v)
        colors = {}
        colors[u] = 0
        queue = [u]
        
        while queue:
            curr = queue.pop(0)
            curr_color = colors[curr]
            
            # Neighbors in H
            neigh_mask = adj[curr] & mask_S
            
            # Add the new edge (u, v) if applicable
            if curr == u:
                neigh_mask |= (1 << v)
            elif curr == v:
                neigh_mask |= (1 << u)
                
            while neigh_mask:
                lsb = neigh_mask & -neigh_mask
                nxt = lsb.bit_length() - 1
                neigh_mask ^= lsb
                
                if nxt not in colors:
                    colors[nxt] = 1 - curr_color
                    queue.append(nxt)
                else:
                    if colors[nxt] == curr_color:
                        return False # Odd cycle found, not bipartite
        
        # Check if all 8 vertices are colored (connected)
        if len(colors) != 8:
            return False
            
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    best_adj = [0] * n
    best_count = 0
    random.seed(42)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        if restart == 0:
            adj = [0] * n
        else:
            # Perturb best
            adj = best_adj[:]
            edges_to_remove = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        edges_to_remove.append((i, j))
            
            if edges_to_remove:
                # Remove 1-3 random edges to escape local optimum
                k = random.randint(1, 3)
                for _ in range(k):
                    if edges_to_remove:
                        u, v = random.choice(edges_to_remove)
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        edges_to_remove.remove((u, v))
        
        added_any = True
        while added_any:
            if time.time() > deadline:
                break
            added_any = False
            # Shuffle edges to ensure we don't always try the same order
            # Shuffling 120 items is cheap
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                
                key = (min(u, v), max(u, v))
                subs = subsets_by_pair.get(key, [])
                
                creates_q3 = False
                for S in subs:
                    if is_q3_candidate(adj, S, u, v):
                        creates_q3 = True
                        break
                
                if not creates_q3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added_any = True
                    # Break out of edge loop to re-shuffle and try others?
                    # No, just continue. The 'added_any' loop handles the "keep trying" logic.
                    # But we should probably break to re-shuffle?
                    # If we don't break, we continue scanning the same list.
                    # This is fine.
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result
