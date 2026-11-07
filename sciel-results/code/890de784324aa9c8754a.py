"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(edges + cross_edges)

    def solve_split_graph(n, k, p, seed):
        # Regenerate graph structure to get neighborhoods
        rng = random.Random(seed)
        # C is 0..k-1, I is k..n-1
        # N[i] for i in I is set of neighbors in C
        N = {}
        for v in range(k, n):
            neighbors = []
            for u in range(k):
                if rng.random() < p:
                    neighbors.append(u)
            N[v] = neighbors
        
        # E_C is all pairs in C
        all_c_edges = list(combinations(range(k), 2))
        
        # Map each edge (u, v) in C to the list of I-vertices that connect to both
        # I_uv = [i for i in I if u in N[i] and v in N[i]]
        # To save time/memory, we can just iterate edges and check.
        
        # Initialize assignments
        # E_i is set of edges assigned to i
        E_assign = {i: set() for i in range(k, n)}
        E_residual = set() # Edges in C not assigned to any I-vertex
        
        for u, v in all_c_edges:
            candidates = []
            for i in range(k, n):
                if u in N[i] and v in N[i]:
                    candidates.append(i)
            
            if candidates:
                # Greedy: assign to first candidate
                # Better: assign to candidate with fewest current edges? 
                # Or just random to spread load?
                # Let's try assigning to the one with smallest current degree in E_assign
                min_i = min(candidates, key=lambda x: len(E_assign[x]))
                E_assign[min_i].add((u, v))
            else:
                E_residual.add((u, v))
        
        cliques = []
        
        # Form cliques for I-vertices based on E_assign
        for i in range(k, n):
            if not N[i]:
                continue # No edges incident to i, no clique needed? 
                # Wait, if N[i] is empty, there are no edges to cover. 
                # But the problem says "every edge must lie in exactly one listed clique".
                # If there are no edges, we don't need a clique.
                # However, if N[i] is not empty, we must cover edges (i, u) for u in N[i].
                # My construction: Cliques are {i} U S where S is a component of (N[i], E_assign[i]).
                # Does this cover all edges (i, u)?
                # Yes, u is in N[i], so u is in some component S. So (i, u) is in {i} U S.
                # What if u is in N[i] but no edge (u, v) in C is assigned to i?
                # Then u is an isolated vertex in the graph (N[i], E_assign[i]).
                # It forms a component {u}. Clique {i, u}.
                # This covers edge (i, u).
                # So we just need to find connected components of (N[i], E_assign[i]).
            
            # Build graph on N[i] with edges E_assign[i]
            # Find connected components
            nodes = N[i]
            if not nodes:
                continue
                
            # Union-Find or BFS for components
            parent = {u: u for u in nodes}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
                    
            for u, v in E_assign[i]:
                if u in parent and v in parent:
                    union(u, v)
            
            components = {}
            for u in nodes:
                root = find(u)
                if root not in components:
                    components[root] = []
                components[root].append(u)
                
            for comp in components.values():
                clique = [i] + comp
                cliques.append(clique)
        
        # Cover residual edges in C
        # E_residual is a set of edges on vertices 0..k-1
        # We need to partition these edges into cliques.
        # Greedy: repeatedly find a maximal clique in the remaining graph.
        remaining_edges = set(E_residual)
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: treat each edge as a clique
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break
            
            # Find a clique that covers the most edges
            # Since k is small, we can try all subsets? No.
            # Greedy heuristic: Pick a vertex with max degree, extend to clique.
            # Or just pick a random edge, try to extend it to a maximal clique.
            
            # Let's pick an edge (u, v)
            u, v = next(iter(remaining_edges))
            
            # Start with clique {u, v}
            clique = {u, v}
            
            # Try to add more vertices to make it a clique in the graph of remaining_edges?
            # No, the clique must be a clique in the ORIGINAL graph G (which is complete on C).
            # So ANY subset of C is a clique.
            # We want to cover the most edges in remaining_edges using this clique.
            # The edges covered by a clique K are all pairs in K that are in remaining_edges.
            
            # Greedy expansion:
            # Try adding vertices w to clique such that all edges (w, x) for x in clique are in remaining_edges?
            # No, if (w, x) is NOT in remaining_edges, it's already covered.
            # We only care about covering remaining_edges.
            # So we can add w if it doesn't "waste" the clique?
            # Actually, adding w covers edges (w, x) for all x in clique IF those edges are in remaining_edges.
            # If (w, x) is not in remaining_edges, adding w doesn't help cover it (already done) but doesn't hurt.
            # However, we must ensure the clique is valid in G. It always is.
            
            # Strategy: Find a clique K that maximizes |E(K) \cap remaining_edges|.
            # Simple heuristic: Start with u, v. 
            # Check all other vertices w. 
            # If we add w, we gain edges (w, x) for x in clique that are in remaining_edges.
            # We should add w if it gains at least 1 edge.
            
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w in clique:
                        continue
                    # Check if adding w increases coverage
                    # Edges gained: (w, x) for x in clique if (w, x) in remaining_edges
                    gain = 0
                    for x in clique:
                        e = (min(w, x), max(w, x))
                        if e in remaining_edges:
                            gain += 1
                    if gain > 0:
                        clique.add(w)
                        changed = True
            
            # Remove covered edges
            for u2, v2 in combinations(clique, 2):
                e = (min(u2, v2), max(u2, v2))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            if len(clique) >= 2:
                cliques.append(list(clique))
            else:
                # Should not happen as we start with an edge
                break

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed-like behavior if time runs out
            # Just return edges as cliques
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split_graph(*spec)
            # Ensure types are correct
            cleaned_res = []
            for c in res:
                cleaned_res.append([int(x) for x in c])
            results.append(cleaned_res)
            
    return results
# EVOLVE-BLOCK-END
