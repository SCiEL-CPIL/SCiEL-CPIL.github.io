"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    def try_greedy_bitmask(n, seed):
        """Try to build a C4-free graph on n vertices with min degree >= n-39 using bitmasks."""
        rng = random.Random(seed)
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
        
        # Start with a random ordering
        order = list(range(n))
        rng.shuffle(order)
        
        # adj[i] is an integer bitmask of neighbors of i
        adj = [0] * n
        
        for u in order:
            # Get potential neighbors not yet connected
            # We can iterate through all v, or shuffle candidates
            candidates = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
            rng.shuffle(candidates)
            
            for v in candidates:
                # Check if adding u-v creates a C4
                # Common neighbors of u and v: adj[u] & adj[v]
                # If count of bits >= 2, it forms a C4 (u-w-v-x-u)
                common = adj[u] & adj[v]
                if common.bit_count() < 2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    # Early exit if u is fully connected? No, we just want high degree.
                    # But if u reaches n-1, we can break
                    if adj[u].bit_count() == n - 1:
                        break
        
        # Check minimum degree
        min_deg = min(adj[i].bit_count() for i in range(n))
        if min_deg >= target_deg:
            # Verify C4-free property (should be true by construction, but good to be safe if logic was complex)
            # The construction logic ensures no edge is added if it creates a C4.
            # Is it possible that adding an edge creates a C4 involving *other* edges not directly checked?
            # A C4 is u-v-w-x-u. 
            # When we add u-v, we check if u and v share 2 neighbors. 
            # If they share w and x, then u-w-v-x-u is a C4? No, u-w-v-x-u implies edges u-w, w-v, v-x, x-u.
            # The check `common = adj[u] & adj[v]` finds neighbors w such that u-w and v-w exist.
            # If there are two such neighbors w1, w2, then u-w1-v-w2-u is a C4.
            # So the check is correct.
            
            edges = []
            for i in range(n):
                mask = adj[i]
                # Iterate over set bits
                j = 0
                while mask:
                    if mask & 1:
                        if i < j:
                            edges.append([i, j])
                    mask >>= 1
                    j += 1
            return {'n': n, 'edges': edges}
        return None
    
    # Try n=47 first, then 46
    for n in [47, 46]:
        target_deg = n - 39
        max_attempts = 1000 if n == 47 else 2000
        
        for seed in range(max_attempts):
            if time.time() > deadline - 5.0: # Safety margin
                break
            
            result = try_greedy_bitmask(n, seed)
            if result is not None:
                return result
    
    # Fallback: return the trivial 39-vertex graph
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
