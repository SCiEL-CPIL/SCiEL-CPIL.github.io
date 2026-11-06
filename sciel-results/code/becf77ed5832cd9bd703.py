import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # We aim for n around 30-35. 
    # Known: R(3,10) <= 41. Largest known triangle-free graph with alpha < 10 is likely around 30-35.
    # The Andrasfai graph A_9 has n=26, alpha=9.
    # We will try to find a graph with n > 26 and alpha <= 9.
    
    best_n = 26
    best_edges = None
    best_score = 0.0 # Higher is better: n - alpha
    
    # Bitset representation for speed
    # n vertices -> n bits. Use Python integers as bitsets.
    
    def get_neighbors(adj, v):
        return adj[v]
    
    def count_triangles(adj, n):
        # Fast triangle count using bitsets
        # For each edge (u, v), count common neighbors
        # Sum over u < v of popcount(adj[u] & adj[v])
        # This counts each triangle 3 times? No, each triangle (u,v,w) is counted in:
        # edge (u,v): w is common
        # edge (u,w): v is common
        # edge (v,w): u is common
        # So total / 3.
        # Actually, let's just sum popcount(adj[u] & adj[v]) for all u < v.
        # This counts each triangle exactly 3 times.
        t = 0
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                t += (au & adj[v]).bit_count()
        return t // 3

    def max_independent_set(adj, n):
        # Exact branch and bound for alpha
        # Use a simple recursive search with pruning
        # State: available vertices bitmask
        # This might be slow for n=35 if not optimized.
        # Let's use a greedy lower bound and a simple upper bound (clique cover or similar)
        # Or just a fast heuristic for alpha? 
        # The task says "checked by exact branch and bound". 
        # My program must return a valid graph. I can use a heuristic to guide search, 
        # but I must ensure the final graph is valid.
        # I will use a fast exact alpha check for small n, or a bounded search.
        
        # For n <= 35, exact alpha is feasible if we use good pruning.
        # Let's implement a standard Bron-Kerbosch-like or simple branch and bound for alpha.
        # Actually, alpha(G) = clique number of complement.
        # Let's find max clique in complement.
        
        # Complement adjacency:
        # comp_adj[v] = ~adj[v] & mask (excluding v)
        
        # We'll use a recursive function to find max clique in complement.
        # To speed up, we can use a greedy coloring bound on the complement graph (which is G).
        
        # Let's stick to a simpler heuristic for the search phase:
        # We want to MINIMIZE alpha.
        # We can estimate alpha using a greedy maximal independent set repeatedly?
        # No, let's just use a fast exact solver for the final candidates.
        # During search, we can use a heuristic alpha estimate.
        
        # Heuristic: Greedy Max Independent Set
        # This gives a lower bound on alpha.
        # If lower bound > 9, it's invalid.
        # But we need to be sure.
        
        # Let's use a bitset-based exact max independent set search with time limit.
        # If it times out, we assume it's large (bad).
        
        # For the search loop, we will use a fast heuristic:
        # 1. Check if triangle-free.
        # 2. Estimate alpha.
        
        pass # Placeholder, will implement inline in search loop for speed
    
    # Search Strategy:
    # Start with random graphs of size n_target.
    # Objective: Minimize (number of triangles) + (lambda * max_independent_set_size)
    # But we want triangle-free AND alpha < 10.
    
    # Let's try specific n values: 28, 30, 32, 34.
    # If we find a valid one, return it.
    
    def is_valid(adj, n):
        # Check triangle-free
        # Check alpha < 10
        # Triangle check:
        for u in range(n):
            au = adj[u]
            # For each v > u in adj[u], check if adj[v] intersects adj[u]
            # If yes, triangle.
            # To be faster:
            # for v in range(u+1, n):
            #    if au & (1 << v):
            #       if au & adj[v]: return False
            # This is O(E * avg_degree) roughly.
            
            # Faster: 
            # For each u, let N(u) be neighbors.
            # For each v in N(u) with v > u, check if N(u) & N(v) is non-empty.
            # We can iterate over set bits.
            
            # Let's just do the bitset check:
            # For each u:
            #   for each v in N(u):
            #      if v > u and (adj[u] & adj[v]):
            #          return False (triangle)
            
            # This is O(sum of deg(u)^2) which is fine for n=30.
            
            # Independent set check:
            # We need to ensure alpha < 10.
            # We can run a quick exact alpha check.
            # For n=30, it's fast.
            
            return True # Placeholder

    # Let's implement a proper fast validity checker.
    
    def check_triangle_free(adj, n):
        for u in range(n):
            au = adj[u]
            # Iterate through neighbors v > u
            v = (au >> (u+1)) & 1
            # Better: 
            # remaining = au & (~((1 << (u+1)) - 1)) # neighbors > u
            # while remaining:
            #    lsb = remaining & -remaining
            #    v = lsb.bit_length() - 1
            #    if au & adj[v]:
            #        return False
            #    remaining ^= lsb
            
            # Simpler loop:
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    if au & adj[v]:
                        return False
        return True

    def exact_alpha(adj, n):
        # Max independent set in G = Max clique in complement of G.
        # Let's build complement adj list (bitset)
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for u in range(n):
            # comp_adj[u] = full_mask ^ adj[u] ^ (1 << u)
            comp_adj[u] = (full_mask ^ adj[u]) & ~(1 << u)
        
        # Bron-Kerbosch with pivot for max clique
        max_clique_size = 0
        
        def bronk(R, P, X):
            nonlocal max_clique_size
            if not P and not X:
                if len(R) > max_clique_size:
                    max_clique_size = len(R)
                return
            
            # Choose pivot u from P U X
            # We want to minimize the number of recursive calls
            # Standard: pick u that maximizes |P & N(u)|
            # For simplicity and speed in Python, let's just pick the first one in P or X
            # Or use a greedy coloring bound? 
            # Given n <= 35, simple BK might be okay if we prune well.
            
            # Optimization: If |R| + |P| <= max_clique_size, return
            if len(R) + P.bit_count() <= max_clique_size:
                return
            
            # Pivot
            # Let's pick u from P
            if P:
                # Pick u with max degree in P?
                # Just pick one
                u_lsb = P & -P
                u = u_lsb.bit_length() - 1
            else:
                u_lsb = X & -X
                u = u_lsb.bit_length() - 1
            
            # Iterate over P \ N(u)
            # P_new = P & ~comp_adj[u]
            # Note: comp_adj[u] is neighbors in complement.
            # We want vertices in P that are NOT adjacent to u in complement.
            # So P & ~comp_adj[u]
            
            P_iter = P & ~comp_adj[u]
            
            while P_iter:
                v_lsb = P_iter & -P_iter
                v = v_lsb.bit_length() - 1
                
                # Recurse
                # New R = R U {v}
                # New P = P & N(v) in complement = P & comp_adj[v]
                # New X = X & N(v) in complement = X & comp_adj[v]
                
                bronk(R | (1 << v), P & comp_adj[v], X & comp_adj[v])
                
                P ^= v_lsb
                X |= v_lsb
                P_iter ^= v_lsb

        # Initialize
        # P = all vertices
        # R = 0, X = 0
        bronk(0, (1 << n) - 1, 0)
        
        return max_clique_size

    # Search
    # Try n from 35 down to 27
    # If we find a valid graph, return it.
    
    for n_target in range(35, 26, -1):
        if time.time() > deadline - 10:
            break
            
        # Random restarts
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            # Probability of edge p
            # For triangle-free, we want sparse-ish but not too sparse.
            # Andrasfai is regular of degree 2*(n-1)/3? No, degree is 2*(n-1)/3 for A_3?
            # A_9: n=26, degree = 2*(25)/3? No.
            # A_k: Z_{3k} with i~j if diff is 1 mod 3.
            # Degree is 2*(k-1)?
            # For k=9, n=27? No, A_9 is on 26 vertices?
            # The seed says n=26.
            # Let's just use p=0.5 or so.
            
            adj = [0] * n_target
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            # Local Search / Hill Climbing
            # We want to eliminate triangles and reduce alpha.
            # Objective: Score = -100 * num_triangles - alpha
            # We want to maximize score (minimize triangles and alpha).
            
            current_adj = adj[:]
            
            # Check initial validity
            if check_triangle_free(current_adj, n_target):
                alpha = exact_alpha(current_adj, n_target)
                if alpha < 10:
                    # Found valid graph!
                    edges = []
                    for u in range(n_target):
                        for v in range(u + 1, n_target):
                            if (current_adj[u] >> v) & 1:
                                edges.append([u, v])
                    return {'n': n_target, 'edges': edges}
            
            # Try to improve
            # Randomly flip edges
            for step in range(200):
                if time.time() > deadline - 2:
                    break
                
                u = random.randint(0, n_target - 1)
                v = random.randint(0, n_target - 1)
                if u == v:
                    continue
                    
                # Flip edge (u, v)
                if (current_adj[u] >> v) & 1:
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
                # Check if improved
                # We only care if it becomes triangle-free and alpha < 10
                # To save time, only check alpha if triangle-free
                if check_triangle_free(current_adj, n_target):
                    alpha = exact_alpha(current_adj, n_target)
                    if alpha < 10:
                        edges = []
                        for i in range(n_target):
                            for j in range(i + 1, n_target):
                                if (current_adj[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n_target, 'edges': edges}
                        
                # If not triangle-free, we could try to fix triangles, but random walk is simpler.
                # To bias towards triangle-free, we can revert if triangles increase significantly?
                # For now, simple random walk.
                
        # If no graph found for this n, try smaller n
        pass

    # Fallback: Return Andrasfai graph (n=26)
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
