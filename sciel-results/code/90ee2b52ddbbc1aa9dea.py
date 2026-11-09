"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    deadline = time.time() + 140
    
    # Start with the known best: 8 disjoint K4's
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    # Try to improve by random search for C5-free graphs with alpha < 9
    # This is a heuristic search; may or find something better
    rng = random.Random(42)
    
    # We'll try to construct graphs with more vertices by using
    # structured approaches: e.g., union of cliques and other C5-free components
    
    # First, try different clique combinations
    # A C5-free graph on n vertices with alpha < 9
    # Max n if we use only K4's: 8 * 4 = 32
    # Can we do better with mixed components?
    
    # Let's try: some K4's and some other C5-free graphs
    # For a component H, we need alpha(H) and it should be C5-free
    # The ratio n/alpha matters for maximizing total n with sum(alpha) < 9
    
    # K4: 4/1 = 4
    # K3: 3/1 = 3  
    # K2: 2/1 = 2
    # K1: 1/1 = 1
    # So K4 is best. 8 K4's gives n=32, alpha=8.
    
    # Can we find a connected C5-free graph with better n/alpha ratio?
    # For connected C5-free graphs:
    # - Complete bipartite K_{a,b} is C5-free, alpha = max(a,b)
    #   Ratio = (a+b)/max(a,b) <= 2
    # - Complete graphs K_k: ratio = k/k = 1... wait no, alpha(K_k)=1, ratio=k
    #   But K_k is C5-free for k < 5. K5 contains C5? No, K5 has many 5-cycles.
    #   Wait, does K5 contain a C5 as a subgraph? Yes! K5 contains all 5-cycles.
    #   So K5 is NOT C5-free.
    #   K4 is C5-free (no 5 vertices to form a cycle).
    
    # So among complete graphs, K4 is the largest C5-free one with alpha=1.
    
    # What about other C5-free graphs?
    # - Cycles C_k for k != 5: C3, C4, C6, etc.
    #   C4: alpha=2, n=4, ratio=2
    #   C6: alpha=3, n=6, ratio=2
    #   C7: alpha=3, n=7, ratio=7/3 ≈ 2.33
    #   C8: alpha=4, n=8, ratio=2
    #   So cycles have ratio at most 7/3 < 4.
    
    # - Trees: alpha can be large. Star K_{1,n}: alpha=n, ratio=1.
    #   Path P_n: alpha = ceil(n/2), ratio ≈ 2.
    
    # So disjoint union of K4's seems optimal among simple constructions.
    
    # Let's try a more systematic search: can we find ANY C5-free graph
    # with n > 32 and alpha < 9?
    
    # This is equivalent to asking if R(C5, K9) > 33.
    # Known: R(C5, K9) is between 33 and 36 (best known lower bound is 32 or 33).
    
    # Let's do a limited random search to see if we can find one.
    
    def has_c5(n, edges):
        """Check if graph has a 5-cycle."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Look for cycles of length 5
        # Use a simple approach: for each pair of non-adjacent vertices,
        # check if there are two distinct paths of length 2 between them
        # through different intermediate vertices
        
        seen_cycles = set()
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    continue  # skip adjacent pairs
                # Find common neighbors
                common = adj[i] & adj[j]
                if len(common) < 2:
                    continue
                # Check if there's a path i-a-b-c-j where a,c in common, b is neighbor of both a and c
                # Actually, for a 5-cycle i-a-b-c-j-i, we need:
                # a in adj[i] ∩ adj[j]? No, that's for 4-cycles.
                # For 5-cycle: i-a-b-c-j-i means:
                # a adjacent to i and b, b adjacent to a and c, c adjacent to b and j
                # So we need a in adj[i], c in adj[j], b in adj[a] ∩ adj[c]
                
                neighbors_i = adj[i]
                neighbors_j = adj[j]
                
                for a in neighbors_i:
                    if a == j:
                        continue
                    for c in neighbors_j:
                        if c == i or c == a:
                            continue
                        if a in adj[c]:
                            # Direct edge a-c, that would make a 4-cycle i-a-c-j-i
                            continue
                        common_ac = adj[a] & adj[c]
                        if common_ac:
                            # Found a 5-cycle: i-a-b-c-j-i
                            cycle = frozenset([i, a, min(common_ac), c, j])
                            # Normalize
                            verts = sorted([i, a, min(common_ac), c, j])
                            if tuple(verts) not in seen_cycles:
                                seen_cycles.add(tuple(verts))
                                return True
        return False
    
    def max_independent_set_greedy(n, edges):
        """Greedy upper bound on independence number (actually finds a lower bound).
        We need an upper bound to verify alpha < 9."""
        # This is hard. Let's use a different approach.
        # For small graphs, we can use a simple branch and bound.
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Use a simple greedy to find a large independent set
        # Then use a quick check for whether alpha >= 9
        
        # Greedy: repeatedly pick vertex with min degree among remaining
        remaining = set(range(n))
        indep = set()
        while remaining:
            # Pick vertex with fewest neighbors in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            indep.add(best_v)
            remaining.discard(best_v)
            remaining -= adj[best_v]
        
        return len(indep)
    
    # The greedy above finds a LOWER bound on alpha.
    # To verify alpha < 9, we need to show NO independent set of size 9 exists.
    # This is co-NP-hard in general, but for n <= 35 we can try a bounded search.
    
    def has_independent_set_of_size_k(n, edges, k):
        """Check if there's an independent set of size k using backtracking."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def backtrack(candidates, chosen):
            if len(chosen) == k:
                return True
            if len(candidates) < k - len(chosen):
                return False
            
            v = min(candidates)
            # Option 1: include v
            new_candidates = candidates - {v} - adj[v]
            if backtrack(new_candidates, chosen | {v}):
                return True
            
            # Option 2: exclude v
            candidates.discard(v)
            if backtrack(candidates, chosen):
                return True
            candidates.add(v)  # restore
            
            return False
        
        return backtrack(set(range(n)), set())
    
    # Now let's try random constructions
    
    # Strategy: start from 8 K4's and try to add edges or vertices
    
    # Actually, let's just try random graphs with various densities
    
    attempts = 0
    while time.time() < deadline - 5 and attempts < 5000:
        attempts += 1
        
        # Try random graphs with n = 33 or 34
        for n_try in [33, 34]:
            if time.time() >= deadline - 5:
                break
            
            # Random graph with appropriate edge probability
            # To be C5-free, we need relatively sparse graph
            # C5-free graphs can have at most ~n^{3/2} edges (by Erdos-Simonovits)
            # For n=33, max edges ~ 33^1.5 ≈ 189
            # But we also need alpha < 9
            
            # Try a random graph with p chosen to be C5-free with some probability
            # This is hard. Let's try structured approaches instead.
            
            # Approach: random union of cliques and other C5-free components
            # But this won't exceed 32 since K4 is optimal.
            
            # Let's try: take a C5-free graph and check properties
            # Generate random sparse graph
            p = rng.uniform(0.1, 0.3)
            edges_try = []
            for i in range(n_try):
                for j in range(i+1, n_try):
                    if rng.random() < p:
                        edges_try.append([i, j])
            
            # Quick check: does it have a C5?
            if has_c5(n_try, edges_try):
                continue
            
            # Check alpha
            if has_independent_set_of_size_k(n_try, edges_try, 9):
                continue
            
            # Valid!
            if n_try > best_n:
                best_n = n_try
                best_edges = edges_try
        
        if time.time() >= deadline - 5:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
