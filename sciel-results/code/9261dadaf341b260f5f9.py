# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 100
    n = 16
    
    # Q3 edges in canonical form (8 vertices, 12 edges)
    q3e = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Q3 adjacency
    q3_adj = {v: [] for v in range(8)}
    for a, b in q3e:
        q3_adj[a].append(b)
        q3_adj[b].append(a)
    
    # Helper to get edge index
    def eidx(u, v):
        if u > v: u, v = v, u
        return u * 16 + v
    
    # Start with K_{3,13}
    edges = set()
    edge_mask = 0
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
            edge_mask |= (1 << eidx(a, b))
            
    def creates_q3(mask, new_edge):
        """Check if adding new_edge to mask creates a Q3 subgraph."""
        u, v = new_edge
        # For each Q3 edge, try both orientations
        for a, b in q3e:
            for perm in [(a, b), (b, a)]:
                pa, pb = perm
                assign = {pa: u, pb: v}
                used = {u, v}
                remaining = [i for i in range(8) if i not in (pa, pb)]
                
                def bt(idx):
                    if time.time() > deadline:
                        return True # Assume conflict to be safe
                    if idx == len(remaining):
                        for ea, eb in q3e:
                            x, y = assign[ea], assign[eb]
                            if (x, y) == new_edge or (y, x) == new_edge:
                                continue
                            if not (mask >> eidx(x, y)) & 1:
                                return False
                        return True
                    qv = remaining[idx]
                    for cand in range(n):
                        if cand in used:
                            continue
                        ok = True
                        for nb in q3_adj[qv]:
                            if nb in assign:
                                x = assign[nb]
                                if (cand, x) == new_edge or (x, cand) == new_edge:
                                    continue
                                if not (mask >> eidx(cand, x)) & 1:
                                    ok = False
                                    break
                        if ok:
                            assign[qv] = cand
                            used.add(cand)
                            res = bt(idx + 1)
                            if res:
                                return True
                            del assign[qv]
                            used.remove(cand)
                    return False
                
                if bt(0):
                    return True
        return False
    
    # Try to add missing edges
    all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    missing = list(all_edges - edges)
    random.seed(42)
    random.shuffle(missing)
    
    for e in missing:
        if time.time() > deadline - 10:
            break
        if creates_q3(edge_mask, e):
            continue
        edges.add(e)
        edge_mask |= (1 << eidx(*e))
        
    # Local Search: 1-for-k swaps
    # We try to remove one edge and add multiple others
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # List of edges to try removing
        current_edges = list(edges)
        random.shuffle(current_edges)
        
        for e in current_edges:
            if time.time() > deadline - 5:
                break
            
            # Temporarily remove e
            mask_without_e = edge_mask & ~(1 << eidx(*e))
            
            # Try to add missing edges
            # We need to consider edges that are currently missing
            # But the set of missing edges changes if we add some.
            # We iterate through a snapshot of missing edges?
            # Or just iterate through all possible edges and check if they are in mask_without_e?
            # If not in mask_without_e, it's a candidate.
            
            temp_mask = mask_without_e
            temp_added = []
            
            # To be efficient, we should only try edges that are "close" to e?
            # Or just all missing edges.
            # There are 120 edges total.
            # Iterating 120 edges for each of ~50 edges is 6000 checks.
            # Each check is fast?
            
            for m in all_edges:
                if m == e:
                    continue
                # Check if m is already in temp_mask
                if (temp_mask >> eidx(*m)) & 1:
                    continue
                    
                if creates_q3(temp_mask, m):
                    continue
                
                # Safe to add
                temp_mask |= (1 << eidx(*m))
                temp_added.append(m)
                
            # If we added more than 1 edge, it's an improvement
            if len(temp_added) > 1:
                # Accept the change
                edges.discard(e)
                for m in temp_added:
                    edges.add(m)
                edge_mask = temp_mask
                improved = True
                break # Restart the scan
        
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
