"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_independent_set_10(adj_mask, n):
    """
    Returns True if there is an independent set of size 10.
    Uses branch and bound.
    """
    target = 10
    
    def dfs(candidates, size):
        if size == target:
            return True
        # If we can't reach target even if we pick all remaining
        if size + candidates.bit_count() < target:
            return False
        
        if not candidates:
            return False
            
        # Heuristic: pick a vertex with high degree in the induced subgraph?
        # For independent set, we want to branch on a vertex.
        # Picking a vertex with high degree in the *complement* (i.e., low degree in original) 
        # is often good for pruning? 
        # Actually, standard MIS heuristics: pick vertex with max degree in the current graph?
        # Here, `candidates` are vertices we can still pick.
        # The "degree" in the independent set context is the number of neighbors in `candidates`.
        # If we pick `v`, we remove `v` and its neighbors.
        # To prune effectively, we want to remove many candidates.
        # So pick `v` with max `adj_mask[v] & candidates` count.
        
        best_v = -1
        max_deg = -1
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v_idx = lsb.bit_length() - 1
            deg = (adj_mask[v_idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v_idx
            temp_cand ^= lsb
            
        if best_v == -1:
            return False
            
        # Option 1: Include best_v
        # Remove best_v and its neighbors from candidates
        new_cand = candidates & ~(adj_mask[best_v] | (1 << best_v))
        if dfs(new_cand, size + 1):
            return True
            
        # Option 2: Exclude best_v
        if dfs(candidates & ~(1 << best_v), size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _is_triangle_free(adj_mask, n):
    for u in range(n):
        # Check all pairs of neighbors
        # If u is connected to v and w, and v is connected to w, it's a triangle.
        # Iterate neighbors of u
        neigh_u = adj_mask[u]
        while neigh_u:
            lsb_v = neigh_u & -neigh_u
            v = lsb_v.bit_length() - 1
            # Check if any neighbor of v is also a neighbor of u
            # (adj_mask[v] & adj_mask[u]) should be 0
            if adj_mask[v] & adj_mask[u]:
                return False
            neigh_u ^= lsb_v
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target n=39 first, then 40
    for target_n in [39, 40]:
        if time.time() > deadline - 20.0:
            break
            
        found = False
        # Try multiple random seeds
        for seed in range(1000):
            if time.time() > deadline - 5.0:
                break
                
            random.seed(seed)
            n = target_n
            adj_mask = [0] * n
            
            # Greedy construction:
            # Add vertices 0 to n-1.
            # For vertex i, connect to a large independent set of previous vertices.
            for i in range(n):
                if i == 0:
                    continue
                # Find a large independent set in G[0..i-1]
                # We can use a greedy approximation or exact for small i
                # For speed, let's use a greedy random selection of an independent set
                # Start with all previous vertices as candidates
                candidates = (1 << i) - 1
                S = 0
                # Greedy: repeatedly pick a random vertex, add to S, remove its neighbors
                # To make it "large", we might want to be smarter, but random is fast
                # Let's try to pick a maximal independent set
                temp_cand = candidates
                while temp_cand:
                    # Pick a random vertex from temp_cand
                    # To bias towards "large" sets, we might pick vertices with fewer connections?
                    # For now, random
                    # Get list of bits
                    bits = []
                    t = temp_cand
                    while t:
                        lsb = t & -t
                        bits.append(lsb.bit_length() - 1)
                        t ^= lsb
                    if not bits: break
                    v = random.choice(bits)
                    S |= (1 << v)
                    # Remove v and its neighbors from temp_cand
                    temp_cand &= ~(adj_mask[v] | (1 << v))
                
                # Connect i to all in S
                adj_mask[i] = S
                for v_bit in S:
                    v = v_bit.bit_length() - 1
                    adj_mask[v] |= (1 << i)
            
            # Verify triangle-free
            if not _is_triangle_free(adj_mask, n):
                continue
                
            # Check independent set
            if _has_independent_set_10(adj_mask, n):
                continue
            
            # Valid graph found!
            edges = []
            for u in range(n):
                m = adj_mask[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    m ^= lsb
            return {'n': n, 'edges': edges}
            
        # If not found for this target_n, try next (or fallback)
        
    # Fallback: Return a known valid small graph (e.g., n=26 Andrásfai)
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
