import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def try_find_graph(n, time_limit):
        # Bitmask representation: adj[i] is an integer with bits set for neighbors of i
        adj = [0] * n
        
        # Pre-compute all 6-subsets
        subsets = list(combinations(range(n), 6))
        
        # Map from subset (tuple) to index? Or just list of tuples.
        # We need to know which subsets contain a specific pair (u, v).
        # Pre-compute: pair_to_subsets = {(u,v): [idx1, idx2, ...]}
        pair_to_subsets = {}
        for idx, s in enumerate(subsets):
            for u, v in combinations(s, 2):
                if u > v: u, v = v, u
                if (u, v) not in pair_to_subsets:
                    pair_to_subsets[(u, v)] = []
                pair_to_subsets[(u, v)].append(idx)
        
        # Initialize random graph
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Function to check if a subset is regular
        # Returns True if regular
        def is_regular(idx):
            s = subsets[idx]
            degs = []
            for u in s:
                # Count neighbors of u within s
                # mask for s
                # We can precompute mask for each subset? 
                # mask = sum(1<<v for v in s)
                # deg = popcount(adj[u] & mask)
                # Python 3.8+ has int.bit_count()
                deg = (adj[u] & subset_masks[idx]).bit_count()
                degs.append(deg)
            return all(d == degs[0] for d in degs)
        
        # Precompute masks for subsets
        subset_masks = []
        for s in subsets:
            m = 0
            for v in s:
                m |= (1 << v)
            subset_masks.append(m)
            
        # Initialize violation list
        violations = []
        for idx in range(len(subsets)):
            if is_regular(idx):
                violations.append(idx)
        
        # Main loop
        while time.time() < time_limit:
            if not violations:
                # Valid
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return n, edges
            
            # Pick a random violation
            v_idx = random.choice(violations)
            s = subsets[v_idx]
            
            # Pick a random internal edge (u, v)
            u, v = random.choice(list(combinations(s, 2)))
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Update violations
            # Subsets affected are those containing both u and v
            affected_indices = pair_to_subsets.get((min(u,v), max(u,v)), [])
            
            # Remove affected indices from violations (if present)
            # And re-check them
            # To make this efficient, we can just rebuild the violations list?
            # No, that's expensive.
            # Let's maintain a set of violations? 
            # Or just a list and filter?
            # Since number of violations is likely small, list is fine.
            
            new_violations = []
            # First, keep violations that are NOT affected
            # But we need to re-check the affected ones.
            # It's easier to just re-check the affected ones and update the list.
            
            # Let's use a set for fast lookup of current violations?
            # No, let's just iterate through current violations and keep those not in affected_indices.
            # Then add back any affected indices that are still regular.
            
            affected_set = set(affected_indices)
            
            # Filter existing violations
            filtered_violations = [idx for idx in violations if idx not in affected_set]
            
            # Re-check affected indices
            for idx in affected_indices:
                if is_regular(idx):
                    filtered_violations.append(idx)
                    
            violations = filtered_violations
            
        return None, None

    # Try n=20
    res_n, res_edges = try_find_graph(20, deadline)
    
    if res_n is not None:
        return {'n': res_n, 'edges': res_edges}
        
    # Fallback n=19
    if time.time() < deadline:
        res_n, res_edges = try_find_graph(19, deadline)
        if res_n is not None:
            return {'n': res_n, 'edges': res_edges}
            
    # Fallback n=5
    return {'n': 5, 'edges': []}
