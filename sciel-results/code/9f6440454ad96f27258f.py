"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(edges + cross_edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        k_edges = [(u, v) for u, v in edges if u < k and v < k]
        cross_edges = [(u, v) for u, v in edges if u < k and v >= k]
        
        # Identify full cross vertices (connected to all K)
        full_cross = set()
        for v in range(k, n):
            if len(adj[v] & set(range(k))) == k:
                full_cross.add(v)
        
        best_solution = None
        best_count = float('inf')
        
        def validate(cliques):
            """Check if cliques are valid and return count"""
            # Check each clique is actually a clique
            for cl in cliques:
                if len(cl) < 2:
                    return float('inf')
                for i in range(len(cl)):
                    for j in range(i+1, len(cl)):
                        if cl[j] not in adj[cl[i]]:
                            return float('inf')
            
            # Check edge coverage: each edge must be in exactly one clique
            edge_count = {}
            for cl in cliques:
                for i in range(len(cl)):
                    for j in range(i+1, len(cl)):
                        u, v = cl[i], cl[j]
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        edge_count[key] = edge_count.get(key, 0) + 1
            
            # Check all edges are covered exactly once
            for u, v in edges:
                key = (min(u,v), max(u,v))
                if edge_count.get(key, 0) != 1:
                    return float('inf')
            
            # Check no non-edges are covered
            for key, cnt in edge_count.items():
                u, v = key
                if (u, v) not in [(min(a,b), max(a,b)) for a,b in edges]:
                    return float('inf')
            
            return len(cliques)
        
        # Strategy 1: Greedy set cover on cross edges
        # Each "super-clique" is K ∪ {subset of full_cross}
        # We need to cover all cross edges
        
        # For vertices in I (not full cross), each edge (u, v) must be in a clique
        # A clique can contain at most one vertex from I (since I is independent)
        # So for each v in I, we need cliques that cover all edges from v to K
        
        # Key insight: A clique containing v (in I) and a subset S of K
        # must have S such that all of S is connected to v, AND S forms a clique in K
        # Since K is a clique, any subset of K is a clique.
        # So a valid clique with v is: {v} ∪ S where S ⊆ N_K(v) and |S| ≥ 1 (plus v)
        # The clique covers all edges (u, v) for u in S, AND all edges within S (which are K-edges)
        
        # So the problem decomposes:
        # - For each v in I, partition N_K(v) into subsets, each becoming a clique with v
        # - Remaining K-edges must be covered by cliques entirely in K
        # - Full cross vertices can be added to any K-clique or K∪I-clique
        
        # Optimal strategy for each v in I:
        # Use one clique: {v} ∪ N_K(v) - this covers ALL edges from v to K
        # But it also covers all K-edges within N_K(v)
        # This is optimal for v: 1 clique per non-full vertex in I
        
        # For full cross vertices:
        # They can be added to any existing clique that contains all of K
        # Or we make a separate clique {v} ∪ K
        
        # Let's think about the structure:
        # If we use one clique per v in I (non-full): {v} ∪ N_K(v)
        # And one clique for K itself (or K ∪ full_cross)
        # But K-edges within N_K(v) are now covered by v's clique
        # K-edges NOT in any N_K(v) need separate cliques
        
        # Actually, the K-edges covered by v's cliques are those within N_K(v)
        # If multiple v's have overlapping N_K(v), those K-edges might be covered multiple times!
        
        # This is the key issue. Let's use a greedy approach:
        
        # Approach: 
        # 1. Start with all K-edges uncovered
        # 2. For each v in I (in some order), create cliques to cover v's cross edges
        #    Each clique also covers some K-edges
        # 3. Cover remaining K-edges
        
        # Better approach: This is essentially a set cover problem
        # Let's try a simple greedy:
        
        # For each v in I:
        #   We must cover all edges (u, v) for u in N_K(v)
        #   A clique {v} ∪ S where S ⊆ N_K(v) covers all (u,v) for u in S
        #   To minimize cliques for v, use ONE clique: {v} ∪ N_K(v)
        #   This covers all cross edges for v AND all K-edges within N_K(v)
        
        # So the natural solution is:
        # - For each v in I: clique {v} ∪ N_K(v) [if N_K(v) non-empty]
        # - For remaining K-edges: cover them with cliques in K
        # - Full cross vertices: add to a clique containing K
        
        # Let's implement this:
        
        # Step 1: Create cliques for each v in I
        cliques = []
        k_edges_covered = set()
        
        # Process non-full I vertices
        for v in range(k, n):
            nbrs = sorted([u for u in adj[v] if u < k])
            if nbrs:
                clique = nbrs + [v]
                cliques.append(clique)
                # Mark K-edges within nbrs as covered
                for i in range(len(nbrs)):
                    for j in range(i+1, len(nbrs)):
                        k_edges_covered.add((min(nbrs[i], nbrs[j]), max(nbrs[i], nbrs[j])))
        
        # Step 2: Handle full cross vertices
        # They can be added to any clique, or we make a dedicated clique
        # Best: add them to a clique that covers K
        # Let's see what K-edges remain uncovered
        remaining_k_edges = set((min(u,v), max(u,v)) for u,v in k_edges) - k_edges_covered
        
        # Step 3: Cover remaining K-edges
        # Use greedy: find largest clique in K that covers uncovered edges
        # Since K is a clique, any subset works
        # We want to cover all remaining K-edges with minimum cliques
        # Each clique in K covers all edges within it
        
        # Simple approach: just use {u, w} for each remaining edge
        # But we can do better by grouping
        
        # For now, let's just add pairs for remaining edges
        # (This can be improved with a set cover)
        
        # Actually, let's try to be smarter:
        # Group remaining K-edges into cliques
        # Since all K vertices form a clique, we can group any subset
        
        # Greedy set cover for K-edges:
        uncovered = list(remaining_k_edges)
        while uncovered:
            # Find vertex in K with most uncovered incident edges
            best_vertex = -1
            best_count = -1
            for u in range(k):
                count = sum(1 for (a,b) in uncovered if a == u or b == u)
                if count > best_count:
                    best_count = count
                    best_vertex = u
            
            if best_vertex == -1:
                break
            
            # Find all vertices connected to best_vertex via uncovered edges
            connected = set()
            remaining_uncovered = []
            for (a,b) in uncovered:
                if a == best_vertex or b == best_vertex:
                    if a == best_vertex:
                        connected.add(b)
                    else:
                        connected.add(a)
                else:
                    remaining_uncovered.append((a,b))
            
            # The clique is {best_vertex} ∪ connected
            clique = [best_vertex] + sorted(connected)
            if len(clique) >= 2:
                cliques.append(clique)
            
            # Add full cross vertices to this clique if beneficial
            # Actually, let's handle full cross separately
            
            uncovered = remaining_uncovered
        
        # Step 4: Handle full cross vertices
        # They must be in cliques. Each full cross vertex v is connected to all K.
        # We can add them to any existing clique that contains all of K, 
        # or make a dedicated clique.
        
        # Check if any existing clique contains all of K
        has_full_k_clique = False
        for cl in cliques:
            if set(range(k)).issubset(set(cl)):
                has_full_k_clique = True
                break
        
        if not has_full_k_clique and full_cross:
            # Create a clique with K and all full cross vertices
            clique = list(range(k)) + sorted(full_cross)
            cliques.append(clique)
        
        # Validate and return
        count = validate(cliques)
        if count < best_count:
            best_count = count
            best_solution = cliques
        
        # Strategy 2: Try different orderings or groupings
        # For now, return the best we have
        if best_solution is None:
            # Fallback: one clique per edge
            best_solution = [[u, v] for u, v in edges]
        
        return best_solution
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        # Ensure types are correct
        cliques = [[int(v) for v in cl] for cl in cliques]
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
