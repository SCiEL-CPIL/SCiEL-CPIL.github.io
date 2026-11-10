import time
import numpy as np

def construct():
    n = 20
    deadline = time.time() + 145.0
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
    m_total = len(all_edges)
    
    # Fast K3,3 check using bitmasks
    # adj[i] is an integer bitmask of neighbors of i
    # We need to ensure no 3 vertices have 3 common neighbors.
    # Condition: For all u < v < w, popcount(adj[u] & adj[v] & adj[w]) <= 2
    
    def is_valid(adj):
        # adj is a list of ints
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = adj[u] & adj[v]
                if common_uv == 0:
                    continue
                # Iterate w > v
                # We can optimize by checking if common_uv has enough bits
                # But n=20 is small, so just loop w
                for w in range(v + 1, n):
                    if (common_uv & adj[w]).bit_count() > 2:
                        return False
        return True
        
    def can_add(adj, u, v):
        # Check if adding edge (u,v) violates K3,3-free property
        # New edges: (u,v)
        # This affects triples containing u or v.
        # Specifically, for any triple {a,b,c}, the common neighbors change only if 
        # the triple contains u or v? No.
        # Adding (u,v) means u becomes neighbor of v and vice versa.
        # This changes N(u) and N(v).
        # It does NOT change N(x) for x != u, v.
        
        # A violation occurs if there is a triple {x,y,z} such that 
        # |N(x) ∩ N(y) ∩ N(z)| >= 3.
        
        # Since only N(u) and N(v) change, any violating triple must contain u or v?
        # Let's analyze:
        # If triple {x,y,z} does not contain u or v, then N(x), N(y), N(z) are unchanged.
        # So no new violations from triples not containing u or v.
        
        # Case 1: Triple contains u but not v. {u, x, y} where x,y != v.
        # New N(u) = Old N(u) ∪ {v}.
        # Intersection: (Old N(u) ∪ {v}) ∩ N(x) ∩ N(y)
        # = (Old N(u) ∩ N(x) ∩ N(y)) ∪ ({v} ∩ N(x) ∩ N(y))
        # Size = |Old N(u) ∩ N(x) ∩ N(y)| + (1 if v ∈ N(x) ∩ N(y) else 0)
        # We need this <= 2.
        # Note: Old graph was valid, so |Old N(u) ∩ N(x) ∩ N(y)| <= 2.
        # If |Old ...| == 2, we need v NOT in N(x) ∩ N(y).
        # If |Old ...| == 1, we need v NOT in N(x) ∩ N(y) OR if it is, 1+1=2 which is ok.
        # If |Old ...| == 0, 0+1=1 ok.
        # So we only fail if |Old N(u) ∩ N(x) ∩ N(y)| == 2 AND v ∈ N(x) ∩ N(y).
        
        # Case 2: Triple contains v but not u. Symmetric.
        # Fails if |Old N(v) ∩ N(x) ∩ N(y)| == 2 AND u ∈ N(x) ∩ N(y).
        
        # Case 3: Triple contains both u and v. {u, v, x}.
        # New N(u) = Old N(u) ∪ {v}
        # New N(v) = Old N(v) ∪ {u}
        # Intersection: (Old N(u) ∪ {v}) ∩ (Old N(v) ∪ {u}) ∩ N(x)
        # = (Old N(u) ∩ Old N(v) ∩ N(x)) 
        #   ∪ ({v} ∩ Old N(v) ∩ N(x))  -> v not in N(v), so empty
        #   ∪ (Old N(u) ∩ {u} ∩ N(x))  -> u not in N(u), so empty
        #   ∪ ({v} ∩ {u} ∩ N(x))       -> empty
        # So it's just Old N(u) ∩ Old N(v) ∩ N(x).
        # This is unchanged from the old graph. Since old graph was valid, this is <= 2.
        
        # So we only need to check Cases 1 and 2.
        
        # Optimization: 
        # For Case 1: We need to check all pairs (x, y) from V \ {u, v}.
        # If v is adjacent to both x and y, we must ensure |N(u) ∩ N(x) ∩ N(y)| < 2.
        # Wait, if |N(u) ∩ N(x) ∩ N(y)| == 2, adding v to N(u) makes it 3 if v is in N(x) ∩ N(y).
        # So: For all x < y in V \ {u, v}, if (adj[x] & adj[y] & (1 << v)) != 0:
        #       if (adj[u] & adj[x] & adj[y]).bit_count() == 2: return False
        
        # For Case 2: For all x < y in V \ {u, v}, if (adj[x] & adj[y] & (1 << u)) != 0:
        #       if (adj[v] & adj[x] & adj[y]).bit_count() == 2: return False
        
        # Let's implement this efficiently.
        # Precompute common neighbors for pairs? No, just compute on fly.
        
        # Note: adj[u] does NOT contain v yet.
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                
                # Check if v is neighbor of both x and y
                # adj[x] & adj[y] & (1 << v)
                if (adj[x] & adj[y]) & (1 << v):
                    # Then we need |N(u) ∩ N(x) ∩ N(y)| <= 1
                    if (adj[u] & adj[x] & adj[y]).bit_count() >= 2:
                        return False
                
                # Check if u is neighbor of both x and y
                if (adj[x] & adj[y]) & (1 << u):
                    # Then we need |N(v) ∩ N(x) ∩ N(y)| <= 1
                    if (adj[v] & adj[x] & adj[y]).bit_count() >= 2:
                        return False
                        
        return True

    # Local Search with Fast Bitmasks
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random graph
        # Start with empty or random
        # Random with p=0.5 is good for n=20
        adj = [0] * n
        edges = set()
        
        # Random initialization
        for i, j in all_edges:
            if rng.random() < 0.4:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.add((i, j))
        
        # Repair: Remove edges causing violations
        # Iterate until valid
        changed = True
        while changed:
            changed = False
            # Find a violation
            # We can just check all triples, but that's slow.
            # Better: pick a random edge, check if removing it fixes a violation?
            # Or: scan for violations and remove one edge from the violating K3,3.
            
            # Scan for violation
            found_violation = False
            for u in range(n):
                if found_violation: break
                for v in range(u + 1, n):
                    if found_violation: break
                    common_uv = adj[u] & adj[v]
                    if common_uv == 0: continue
                    for w in range(v + 1, n):
                        common_uvw = common_uv & adj[w]
                        if common_uvw.bit_count() >= 3:
                            # Found K3,3: {u,v,w} and 3 neighbors in common_uvw
                            # Remove one edge. Pick random edge from the K3,3.
                            # The K3,3 consists of edges between {u,v,w} and the 3 neighbors.
                            # Let's pick a random neighbor z from common_uvw and remove (u,z)
                            # Or just remove (u,v)? No, (u,v) is not part of the bipartite K3,3 definition 
                            # in the sense that u,v are on one side. The edges are (u,z), (v,z), (w,z) etc.
                            # Actually, any edge in the subgraph induced by {u,v,w} ∪ C where C is the 3 neighbors 
                            # can be removed. The edges are (a,b) for a in {u,v,w}, b in C.
                            # Let's remove a random one.
                            c_list = [i for i in range(n) if (common_uvw >> i) & 1]
                            z = c_list[rng.integers(3)]
                            a = [u, v, w][rng.integers(3)]
                            # Remove (a, z)
                            if a > z: a, z = z, a
                            if (a, z) in edges:
                                edges.remove((a, z))
                                adj[a] &= ~(1 << z)
                                adj[z] &= ~(1 << a)
                                changed = True
                                found_violation = True
                            break
            if not changed:
                break
                
        if time.time() > deadline - 5:
            break
            
        # Now try to add edges greedily
        # Shuffle missing edges
        missing = [(i, j) for i, j in all_edges if (i, j) not in edges]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
                
        # Local Search: Try to swap edges to increase count? 
        # Since we started with random and added greedily, we are likely at a local maximum for edge addition.
        # To improve, we can try 1-for-1 swaps (remove one, add one) which doesn't increase count, 
        # but might allow further additions.
        # Or 1-for-2 swaps (remove one, add two) which increases count.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_edges = list(edges)
            rng.shuffle(current_edges)
            
            for e in current_edges:
                if time.time() > deadline - 5:
                    break
                u, v = e
                # Remove (u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add 2 edges
                added = 0
                # Candidates are missing edges
                # To be fast, just iterate all missing edges in random order
                # We need to re-generate missing list or just check all pairs
                # Checking all pairs is O(n^2) = 400 checks. can_add is fast.
                candidates = [(i, j) for i, j in all_edges if (i, j) not in edges]
                rng.shuffle(candidates)
                
                for a, b in candidates:
                    if added >= 2:
                        break
                    if can_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        edges.add((a, b))
                        added += 1
                        
                if added >= 2:
                    improved = True
                    # Net gain of 1 edge
                elif added == 1:
                    # Net 0, keep it? Might help later. 
                    # But we removed 1 and added 1. Count is same.
                    # Let's keep it to explore different local optima.
                    pass
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add(e)
                    # No added edges to remove
                    
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
        if best_count >= 55:
            break
            
    # Final verification
    if best_edges:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        if is_valid(adj_final):
            return best_edges
            
    # Fallback if something went wrong
    return []
