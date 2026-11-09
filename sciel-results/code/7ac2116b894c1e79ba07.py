import time
import sys
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        # Clique part 0..k-1
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Independent part k..n-1
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # K is 0..k-1, I is k..n-1
        K = list(range(k))
        I = list(range(k, n))
        
        # Track covered edges in K
        # Use a set of frozensets or just a set of tuples
        covered_K_edges = set()
        
        cliques = []
        
        # Helper to get uncovered edges in K
        def get_uncovered_K_edges():
            # We need to find cliques in the graph of uncovered edges in K
            # Graph G_uncov has vertices K, edges are (u,v) in K if (u,v) not in covered_K_edges
            pass

        # Sort I by degree in K (size of neighborhood in K)
        # Smaller neighborhoods first
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(K)))
        
        for v in I_sorted:
            N_v_K = adj[v] & set(K)
            if not N_v_K:
                continue
            
            # Find largest clique S in N_v_K such that all edges in S are uncovered in K
            # G_uncov restricted to N_v_K
            # Vertices: N_v_K
            # Edges: (u, w) in N_v_K if (u, w) not in covered_K_edges
            
            # Greedy clique finding in G_uncov[N_v_K]
            # Start with all vertices in N_v_K
            # Remove vertices that don't form a clique?
            # Simple greedy:
            # 1. Sort vertices in N_v_K by degree in G_uncov[N_v_K] descending
            # 2. Pick first, intersect neighbors, repeat.
            
            # Calculate degrees in G_uncov[N_v_K]
            # For u in N_v_K, deg(u) = count of w in N_v_K such that (u,w) not in covered_K_edges
            
            # To speed up, if N_v_K is small, we can just check subsets.
            # k is at most 30. N_v_K is at most 30.
            
            # Let's try to find a maximal clique greedily
            candidates = list(N_v_K)
            # Sort by degree in G_uncov
            def deg_in_uncov(u):
                count = 0
                for w in candidates:
                    if u != w and (u, w) not in covered_K_edges and (w, u) not in covered_K_edges:
                         # Note: covered_K_edges stores (min, max)
                         a, b = sorted((u, w))
                         if (a, b) not in covered_K_edges:
                             count += 1
                return count
            
            # We need to be careful with the definition of covered_K_edges.
            # It contains tuples (u, v) with u < v.
            
            # Let's re-implement the check
            def is_edge_uncovered(u, w):
                a, b = sorted((u, w))
                return (a, b) not in covered_K_edges

            # Greedy Max Clique on G_uncov[N_v_K]
            # This is a heuristic.
            S = []
            remaining = set(N_v_K)
            # Sort remaining by degree in G_uncov[remaining]
            # We'll do a simple pass
            
            # To improve, we can try multiple starts?
            # Given time limit, let's stick to one good heuristic.
            
            # Heuristic:
            # 1. Find vertex with max degree in G_uncov[N_v_K]
            # 2. Add to S
            # 3. Remove all non-neighbors from remaining
            # 4. Repeat
            
            current_remaining = set(N_v_K)
            while current_remaining:
                # Find max degree vertex in current_remaining
                max_deg_v = None
                max_deg = -1
                for u in current_remaining:
                    deg = 0
                    for w in current_remaining:
                        if u != w and is_edge_uncovered(u, w):
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        max_deg_v = u
                
                if max_deg_v is None:
                    break
                    
                S.append(max_deg_v)
                # Keep only neighbors of max_deg_v in current_remaining
                current_remaining = {u for u in current_remaining if is_edge_uncovered(u, max_deg_v)}
                
            if S:
                # Check if S is a valid clique in G_uncov
                # It should be by construction
                # Add clique {v} + S
                clique = [v] + S
                cliques.append(clique)
                
                # Mark edges in S as covered in K
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = sorted((S[i], S[j]))
                        covered_K_edges.add((a, b))
            else:
                # No clique of size > 1 found?
                # This means N_v_K is an independent set in G_uncov.
                # So we must cover cross edges individually.
                for u in N_v_K:
                    cliques.append([v, u])
                    # Note: This does NOT cover any edge in K.
                    # So covered_K_edges is unchanged.

        # Cover remaining edges in K
        # Find all uncovered edges in K
        # Greedily form cliques
        
        # Get list of uncovered edges
        all_K_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        
        while True:
            # Find an uncovered edge
            next_edge = None
            for e in all_K_edges:
                if e not in covered_K_edges:
                    next_edge = e
                    break
            
            if next_edge is None:
                break
                
            u, v = next_edge
            # Expand to a clique containing (u, v)
            # Start with S = {u, v}
            S = {u, v}
            
            # Try to add more vertices to S
            # Candidates are vertices w in K such that (w, s) is uncovered for all s in S
            # and (w, s) is an edge in original graph (which is always true for K)
            
            # Greedy expansion
            # Find vertex w with max degree to S in G_uncov
            while True:
                best_w = None
                best_score = -1
                for w in range(k):
                    if w in S:
                        continue
                    # Check if w can be added
                    can_add = True
                    score = 0
                    for s in S:
                        if not is_edge_uncovered(w, s):
                            can_add = False
                            break
                        score += 1
                    if can_add and score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w is None:
                    break
                S.add(best_w)
                
            # Add clique S
            cliques.append(list(S))
            
            # Mark edges in S as covered
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a, b = sorted((S_list[i], S_list[j]))
                    covered_K_edges.add((a, b))
                    
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        # Ensure types are correct
        # cliques is list of lists of ints
        results.append(cliques)
        
    return results

# Add deadline guard
import time
start_time = time.time()
# The function construct is called by evaluator.
# We need to ensure it finishes in time.
# The above algorithm is O(N * K^3) roughly, which is fast for K=30.
