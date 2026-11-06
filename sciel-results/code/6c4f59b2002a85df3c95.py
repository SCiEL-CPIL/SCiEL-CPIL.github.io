import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strategy: Use Paley Graphs as a base.
    # Paley graph P_p exists for p prime, p = 1 mod 4.
    # Vertices are 0..p-1. Edge (u,v) if (v-u) is a quadratic residue mod p.
    # These graphs are strongly regular and have specific induced subgraph properties.
    
    # Known small primes p = 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97...
    # We want to maximize n.
    # The task states graphs with 20 vertices are known. 
    # Paley graphs are defined for prime n. 
    # P_13 has 13 vertices. 
    # P_17 has 17 vertices.
    # P_29 has 29 vertices (too big, n<=22).
    # So candidates from Paley: 5, 13, 17.
    # We can also try to combine or modify.
    
    # First, let's define a fast validator for a given graph.
    # A graph is invalid if it has an induced regular subgraph of size >= 6.
    # An induced subgraph on S is regular if all vertices in S have the same degree within S.
    
    def is_valid(n, edges):
        # Build adjacency matrix for speed
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        # Check all subsets of size 6, 7, ..., n
        # For efficiency, we can check size 6 first. If any size 6 is regular, it's invalid.
        # Actually, if a subset of size k>=6 is regular, it might not contain a regular subset of size 6.
        # But usually, small regular subgraphs are the bottleneck.
        # The condition is "No set of 6 or more vertices may induce a regular subgraph".
        # So we must check ALL subsets S with |S| >= 6.
        
        # Optimization: 
        # If a subset S is regular, all vertices in S have degree d_S in G[S].
        # For |S|=6, possible degrees are 0..5.
        
        # Let's iterate through subsets.
        # n is small (<=22). 
        # Number of subsets of size 6 in 22 is 74613. 
        # Number of subsets of size 7 is 170544.
        # Total subsets of size >=6 for n=22 is sum(C(22,k) for k=6..22) which is ~2.2 million.
        # Checking each subset involves computing degrees, which is O(k^2) or O(k) with precomputation.
        # 2.2M * 22 ops is ~50M ops. Python might be slow for this in 140s if done naively for many candidates.
        # We need a faster check or fewer candidates.
        
        # Precompute degrees for all subsets? No, too much memory.
        # We can use bitmasks.
        
        # Let's represent the graph as a list of bitmasks for adjacency.
        adj_mask = [0]*n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        def check_subset_mask(mask, size):
            # Compute degree of each vertex in the subset
            # Vertices in subset are bits set in mask
            # deg[v] = popcount(adj_mask[v] & mask)
            # Check if all deg[v] are equal
            first_deg = -1
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                d = (adj_mask[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    return False
                m ^= lsb
            return True

        # Check subsets of size 6 to n
        # It is faster to check smaller sizes first? 
        # If a large subset is regular, it doesn't imply a smaller one is.
        # However, regular subgraphs of size 6 are the most common "bad" pattern in random-ish graphs.
        # We must check all sizes.
        
        # To speed up, we can prune.
        # But for n=13, 17, the number of subsets is much smaller.
        # n=13: sum(C(13,k) for k=6..13) = 1716 + 1716 + 715 + 286 + 78 + 13 + 1 = 4525. Very fast.
        # n=17: sum(C(17,k) for k=6..17) = 12376 + 24310 + 38760 + 50388 + 53130 + 43758 + 28394 + 14950 + 6188 + 2040 + 510 + 102 + 17 + 1 = ~262k. Fast.
        # n=22: ~2.2M. Might take a few seconds in Python.
        
        # We will generate candidates and validate.
        
        candidates = []
        
        # Generate Paley Graphs
        def is_prime(x):
            if x < 2: return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0: return False
            return True
            
        def quadratic_residues(p):
            return [(i*i) % p for i in range(1, p)]
            
        def make_paley(p):
            if p % 4 != 1 or not is_prime(p):
                return None
            qrs = set(quadratic_residues(p))
            edges = []
            for u in range(p):
                for v in range(u+1, p):
                    if (v - u) % p in qrs:
                        edges.append([u, v])
            return {'n': p, 'edges': edges}
            
        # Try known Paley primes <= 22
        for p in [5, 13, 17]:
            g = make_paley(p)
            if g:
                candidates.append(g)
                
        # Also try to construct n=20 or 21 by modifying Paley 17 or 13?
        # Or use a known graph from literature? 
        # The prompt says "Graphs with 20 vertices are known".
        # Can we find one? 
        # A common strategy is to take a valid graph on n-1 and add a vertex.
        # Or take a Paley graph and add vertices.
        
        # Let's try a local search starting from Paley 17.
        # Add vertices one by one, choosing edges to maintain validity.
        
        def add_vertex(g):
            n = g['n']
            edges = g['edges'][:]
            new_v = n
            # Try all possible edge sets for new vertex? 2^n possibilities.
            # For n=17, 2^17 = 131k possibilities.
            # For each, check validity of the new graph (n+1).
            # This is expensive if done for all subsets.
            # But we can optimize: only check subsets involving the new vertex?
            # No, adding a vertex doesn't change induced subgraphs not containing it.
            # So we only need to check subsets S containing new_v.
            # |S| >= 6. S = {new_v} U T, where T is subset of old vertices.
            # T can be size 5 to n.
            # This reduces the check significantly.
            
            # However, implementing this incremental check is complex.
            # Given the 140s budget, and that n=17 is a solid improvement over 5,
            # and n=20 is "known" but we don't have the specific edges,
            # let's stick to validating Paley graphs and maybe a simple greedy add if time permits.
            
            pass # Placeholder for complex logic
            
        # Let's just validate the candidates we have.
        # Sort by n descending
        candidates.sort(key=lambda x: x['n'], reverse=True)
        
        best_graph = None
        for g in candidates:
            if time.time() > deadline:
                break
            if is_valid(g['n'], g['edges']):
                best_graph = g
                break # We want max n, so first valid one in sorted order is best
                
        if best_graph is None:
            # Fallback to empty graph on 5 if all fail (shouldn't happen)
            return {'n': 5, 'edges': []}
            
        return best_graph

# We can also try to generate a random graph on n=20 and check?
# Random graphs likely have regular induced subgraphs.
# Paley graphs are structured to avoid them.

# Let's add a simple greedy attempt to extend the best found graph if it's < 20.
# But the code above returns immediately upon finding the best Paley.
# Let's modify to allow a quick search for n=20 if Paley 17 is valid.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0: return False
            return True
            
    def quadratic_residues(p):
        return [(i*i) % p for i in range(1, p)]
        
    def make_paley(p):
        if p % 4 != 1 or not is_prime(p):
            return None
        qrs = set(quadratic_residues(p))
        edges = []
        for u in range(p):
            for v in range(u+1, p):
                if (v - u) % p in qrs:
                    edges.append([u, v])
        return {'n': p, 'edges': edges}
        
    def is_valid(n, edges):
        adj_mask = [0]*n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        def check_subset_mask(mask, size):
            first_deg = -1
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                d = (adj_mask[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    return False
                m ^= lsb
            return True

        # Check subsets of size 6 to n
        # Use itertools.combinations for clarity, but bitmasks are faster for checking.
        # For n <= 17, combinations are fine.
        for k in range(6, n+1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if check_subset_mask(mask, k):
                    return False
        return True

    # Candidates
    candidates = []
    for p in [17, 13, 5]:
        g = make_paley(p)
        if g:
            candidates.append(g)
            
    # Sort by n desc
    candidates.sort(key=lambda x: x['n'], reverse=True)
    
    best_graph = None
    for g in candidates:
        if time.time() > deadline:
            break
        if is_valid(g['n'], g['edges']):
            best_graph = g
            break
            
    if best_graph is None:
        return {'n': 5, 'edges': []}
        
    # If we found n=17, can we try to extend to n=20?
    # This is complex. Let's see if we have time.
    if best_graph['n'] < 20 and time.time() < deadline - 20:
        # Try a simple random greedy extension from the best graph
        current_n = best_graph['n']
        current_edges = best_graph['edges'][:]
        
        # We want to add vertices up to 20 or 21.
        # This is a search problem.
        # Given time constraints, we might skip this and return the Paley graph.
        # Paley 17 is a score of 17/5 = 3.4.
        # The inspiration said 3.4 was achieved by randomized greedy.
        # So matching 3.4 is a good goal.
        # If we can hit 20, score is 4.0.
        
        # Let's try a very limited random search for n=20.
        # Start with empty graph on 20? No, start with Paley 17 and add 3 vertices.
        # This is hard to implement correctly in the time left.
        
        pass
        
    return best_graph
