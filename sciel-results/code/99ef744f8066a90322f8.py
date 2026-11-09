import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    target_n = 20
    
    # Precompute subsets of size 6 and 7
    subsets_6_7 = []
    for k in [6, 7]:
        for comb in itertools.combinations(range(target_n), k):
            mask = 0
            for v in comb:
                mask |= 1 << v
            subsets_6_7.append(mask)
    
    # Map each subset to an index
    subset_to_idx = {mask: i for i, mask in enumerate(subsets_6_7)}
    
    # Map each edge (u, v) to list of subset indices containing both
    # Edge represented as tuple (min(u,v), max(u,v))
    edge_to_subsets = defaultdict(list)
    for i, mask in enumerate(subsets_6_7):
        verts = [v for v in range(target_n) if mask & (1 << v)]
        for u_idx in range(len(verts)):
            for v_idx in range(u_idx + 1, len(verts)):
                u = verts[u_idx]
                v = verts[v_idx]
                edge = (u, v) if u < v else (v, u)
                edge_to_subsets[edge].append(i)
                
    def is_regular(mask, adj):
        """Check if induced subgraph on mask is regular."""
        verts = [v for v in range(target_n) if mask & (1 << v)]
        if len(verts) < 6:
            return False
        degs = []
        for v in verts:
            degs.append((adj[v] & mask).bit_count())
        return len(set(degs)) == 1

    def full_verify(adj):
        """Check if graph has no regular induced subgraph on >= 6 vertices."""
        # Check sizes 6 to target_n
        for k in range(6, target_n + 1):
            for comb in itertools.combinations(range(target_n), k):
                mask = 0
                for v in comb:
                    mask |= 1 << v
                if is_regular(mask, adj):
                    return False
        return True

    def run_search(n, time_limit):
        adj = [0] * n
        # Initialize random graph
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        
        # Calculate initial bad status for subsets 6,7
        is_bad = [False] * len(subsets_6_7)
        bad_count = 0
        for i, mask in enumerate(subsets_6_7):
            if is_regular(mask, adj):
                is_bad[i] = True
                bad_count += 1
                
        best_result = None
        
        while time.time() < time_limit:
            if bad_count == 0:
                if full_verify(adj):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            
            # Pick a random bad subset
            # Find indices of bad subsets
            bad_indices = [i for i in range(len(is_bad)) if is_bad[i]]
            if not bad_indices:
                # Should not happen if bad_count > 0, but safety
                continue
                
            idx = random.choice(bad_indices)
            mask = subsets_6_7[idx]
            
            # Get vertices in this subset
            verts = [v for v in range(n) if mask & (1 << v)]
            
            # Pick a random edge in this subset to flip
            # Ensure there is at least one edge? 
            # If the subset is regular, it has some degree d.
            # If d=0, no edges. We must add an edge.
            # If d>0, we can flip an existing edge or add a new one?
            # Flipping an edge means if present, remove; if not, add.
            # To break regularity, we should change the degree sequence.
            
            # Let's pick a random pair in the subset
            u, v = random.sample(verts, 2)
            
            # Flip edge (u, v)
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            
            # Update bad status for all subsets containing this edge
            edge_key = (u, v) if u < v else (v, u)
            affected_indices = edge_to_subsets.get(edge_key, [])
            
            for i in affected_indices:
                new_bad = is_regular(subsets_6_7[i], adj)
                old_bad = is_bad[i]
                if new_bad != old_bad:
                    is_bad[i] = new_bad
                    if new_bad:
                        bad_count += 1
                    else:
                        bad_count -= 1
                        
        return None

    # Try to find graph on 20 vertices
    result = run_search(20, deadline)
    if result:
        return result
        
    # If failed, try 19
    # Re-compute subsets for 19? 
    # For simplicity and time, if 20 fails, return seed or try 19 quickly.
    # Given 20 is known, 20 should work.
    
    return {'n': 5, 'edges': []}
