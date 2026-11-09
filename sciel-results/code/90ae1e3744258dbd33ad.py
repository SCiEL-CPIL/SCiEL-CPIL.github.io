import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_n = 17
    best_edges = None
    
    # Seed: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17

    def is_valid(n, edges):
        # Build adjacency matrix for speed
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        
        # Check K4
        # Iterate over triples, check if they form a triangle, then check 4th vertex
        # Or just iterate all 4-combinations? n<=40, C(40,4) ~ 91k. Fast enough.
        # But we do this in the loop, so it needs to be fast.
        # Using bitmasks is faster.
        
        # Let's use a simpler check for the search loop
        # Count K4s and Ind 6s
        
        # K4 Count
        k4_count = 0
        # Precompute neighborhoods as bitmasks?
        # For n up to 40, Python ints are fine.
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        
        # K4: u<v<w<x, all connected
        # This is slow in Python loops.
        # Optimization: For each edge (u,v), count common neighbors.
        # If a common neighbor set has size >= 2, it might form K4s.
        # Actually, just count K4s directly using bit operations.
        
        # Independent Set 6:
        # Check if there is a set of 6 vertices with no edges between them.
        # This is equivalent to checking if the complement graph has a K6.
        comp_masks = [0] * n
        for u in range(n):
            # Complement of adj (excluding self)
            # All bits set except u and neighbors
            full_mask = (1 << n) - 1
            # Remove u
            full_mask ^= (1 << u)
            # Remove neighbors
            full_mask &= ~masks[u]
            comp_masks[u] = full_mask
            
        # Check for K6 in complement (Ind 6 in original)
        # And K4 in original.
        
        # This check is O(n^4) or similar.
        # For n=20, it's fast.
        
        # Fast K4 check:
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    # Iterate through common neighbors w > v
                    # To avoid duplicates, we can just check if any w, x in common are connected.
                    # Simpler: just count K4s by iterating 4-combinations?
                    # Let's just return False if we find ONE K4 or ONE Ind6.
                    
        # We only need to know if it's VALID (0 K4, 0 Ind6).
        # So we can stop early.
        
        # Check K4
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]: continue
                common = masks[u] & masks[v]
                # Check if any pair in common is connected
                # Iterate w in common
                w = common
                while w:
                    lsb = w & -w
                    wi = lsb.bit_length() - 1
                    if wi <= v: # only consider w > v to avoid dup? No, just check connection
                        pass
                    # Check if w is connected to any other in common
                    # This is getting complicated.
                    
        # Let's use a simpler, slightly slower but correct check for n <= 30
        # For n > 30, we might skip or use approximation? No, must be exact.
        
        # Actually, for the search, we need a cost function.
        # Cost = #K4 + #Ind6.
        # We minimize this.
        
        pass # Placeholder, will implement cost function

    def cost(n, adj):
        # Returns number of K4s + number of Ind 6s
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        
        k4_count = 0
        # Count K4s
        # For each pair (u,v) with edge, look at common neighbors.
        # A K4 is a triangle (u,v,w) plus x connected to all.
        # Easier: Iterate all 4-combinations?
        # For n=20, C(20,4)=4845. Very fast.
        # For n=30, C(30,4)=27405. Fast.
        # For n=40, C(40,4)=91390. Fast.
        
        import itertools
        for comb in itertools.combinations(range(n), 4):
            u, v, w, x = comb
            if adj[u,v] and adj[u,w] and adj[u,x] and adj[v,w] and adj[v,x] and adj[w,x]:
                k4_count += 1
        
        # Count Ind 6s
        # Complement graph adjacency
        comp_adj = np.zeros((n, n), dtype=bool)
        for u in range(n):
            for v in range(n):
                if not adj[u,v] and u != v:
                    comp_adj[u,v] = True
        
        ind6_count = 0
        for comb in itertools.combinations(range(n), 6):
            is_ind = True
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = comb[i], comb[j]
                    if adj[u, v]: # Edge exists in original -> not independent
                        is_ind = False
                        break
                if not is_ind:
                    break
            if is_ind:
                ind6_count += 1
                
        return k4_count + ind6_count

    def try_find_graph(n, time_budget):
        end_time = time.time() + time_budget
        # Initial graph: Random regular-ish
        adj = np.zeros((n, n), dtype=bool)
        # Try Paley-like if n is prime and 1 mod 4?
        # For now, random.
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i, j] = True
                    adj[j, i] = True
        
        current_cost = cost(n, adj)
        
        while time.time() < end_time:
            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            old_state = adj[u, v]
            adj[u, v] = not old_state
            adj[v, u] = not old_state
            
            new_cost = cost(n, adj)
            
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                # Accept with probability exp(-delta/T)
                delta = new_cost - current_cost
                T = 1.0 # Cooling
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u, v] = old_state
                    adj[v, u] = old_state
            
            if current_cost == 0:
                # Valid found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                return edges
                
        return None

    # Search from 18 to 35
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        # Give more time to higher n? No, higher n is harder.
        # Give fixed small budget per n.
        budget = 2.0
        if n > 25:
            budget = 1.0
            
        edges = try_find_graph(n, budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            # If we found a high n, we can stop or continue to try for higher?
            # The task says "Maximize n". So we should keep trying.
            # But if we found 35, we can stop.
            if n == 35:
                break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
