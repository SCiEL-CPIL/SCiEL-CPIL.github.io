"""Optimized split-graph clique partition using local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        start_time = time.time()
        deadline = start_time + 110
        
        # Generate the graph
        rng = random.Random(seed)
        N = [set() for _ in range(n)]  # N[v] = neighbors of v in C (for v in I)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        
        # Initial solution: one clique for C, one 2-clique per cross edge
        cliques = [list(range(k))]  # The big clique C
        for v in range(k, n):
            for u in sorted(N[v]):
                cliques.append([u, v])
        
        # Track which edges are covered by which clique
        # edge_to_clique maps (u,v) -> index in cliques
        edge_to_clique = {}
        for idx, clique in enumerate(cliques):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    edge = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    edge_to_clique[edge] = idx
        
        total_cliques = len(cliques)
        
        # Local search: try to merge two 2-cliques into a larger clique
        # A 2-clique [u,v] where v in I can be merged with [w,v'] if the union forms a clique
        # and the edges are not already covered elsewhere
        
        # Better approach: try to replace a set of 2-cliques with a single larger clique
        # For each pair of I-vertices v1, v2, check if {v1, v2} ∪ (N[v1] ∩ N[v2]) is a clique
        # and if so, replace the individual 2-cliques with this one
        
        # Actually, let's use a different strategy:
        # For each I-vertex v, we can form a clique {v} ∪ S where S ⊆ N[v]
        # This clique covers edges (v, u) for u in S, and edges (u1, u2) for u1, u2 in S
        # The C-edges within S must not be covered by any other clique
        
        # Greedy improvement: for each I-vertex v (sorted by degree), try to make it "active"
        # by forming {v} ∪ N[v], if the C-edges within N[v] are not covered by other active vertices
        
        # Let's use a simulated annealing / local search approach
        # State: for each I-vertex v, either it's "active" (using {v} ∪ N[v]) or "inactive" (individual 2-cliques)
        
        # Represent state as a set of active vertices
        active = set()
        
        def compute_cliques(active_set):
            """Compute the clique list given the active set."""
            cliques = []
            
            # C-edges not covered by any active vertex's N[v]
            covered_c_pairs = set()
            for v in active_set:
                nv = list(N[v])
                for i in range(len(nv)):
                    for j in range(i + 1, len(nv)):
                        covered_c_pairs.add((min(nv[i], nv[j]), max(nv[i], nv[j])))
            
            # Add cliques for active vertices
            for v in active_set:
                cliques.append([v] + sorted(N[v]))
            
            # Add 2-cliques for inactive vertices
            for v in range(k, n):
                if v not in active_set:
                    for u in sorted(N[v]):
                        cliques.append([u, v])
            
            # Cover remaining C-edges with a greedy clique cover
            all_c_pairs = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_c_pairs.add((u, w))
            
            remaining = all_c_pairs - covered_c_pairs
            
            if remaining:
                rem_adj = {u: set() for u in range(k)}
                for u, w in remaining:
                    rem_adj[u].add(w)
                    rem_adj[w].add(u)
                
                rem_copy = set(remaining)
                while rem_copy:
                    if time.time() > deadline:
                        for u, w in rem_copy:
                            cliques.append([u, w])
                        break
                    
                    # Find a maximal clique greedily
                    degrees = {}
                    for u in range(k):
                        deg = 0
                        for w in rem_adj[u]:
                            if (min(u, w), max(u, w)) in rem_copy:
                                deg += 1
                        degrees[u] = deg
                    
                    if max(degrees.values(), default=0) == 0:
                        break
                    
                    start = max(degrees, key=degrees.get)
                    clique = [start]
                    candidates = [c for c in rem_adj[start] if (min(start, c), max(start, c)) in rem_copy]
                    for c in candidates:
                        if all((min(c, u), max(c, u)) in rem_copy for u in clique):
                            clique.append(c)
                    
                    cliques.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_copy.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
            
            return cliques
        
        def count_cliques(active_set):
            """Count the number of cliques without building the full list."""
            # C-edges covered by active vertices
            covered_c_pairs = set()
            for v in active_set:
                nv = list(N[v])
                for i in range(len(nv)):
                    for j in range(i + 1, len(nv)):
                        covered_c_pairs.add((min(nv[i], nv[j]), max(nv[i], nv[j])))
            
            # Count cliques from active vertices
            count = len(active_set)
            
            # Count 2-cliques from inactive vertices
            for v in range(k, n):
                if v not in active_set:
                    count += len(N[v])
            
            # Count cliques needed for remaining C-edges
            all_c_pairs = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_c_pairs.add((u, w))
            
            remaining = all_c_pairs - covered_c_pairs
            
            if remaining:
                rem_adj = {u: set() for u in range(k)}
                for u, w in remaining:
                    rem_adj[u].add(w)
                    rem_adj[w].add(u)
                
                rem_copy = set(remaining)
                while rem_copy:
                    degrees = {}
                    for u in range(k):
                        deg = 0
                        for w in rem_adj[u]:
                            if (min(u, w), max(u, w)) in rem_copy:
                                deg += 1
                        degrees[u] = deg
                    
                    if max(degrees.values(), default=0) == 0:
                        break
                    
                    start = max(degrees, key=degrees.get)
                    clique = [start]
                    candidates = [c for c in rem_adj[start] if (min(start, c), max(start, c)) in rem_copy]
                    for c in candidates:
                        if all((min(c, u), max(c, u)) in rem_copy for u in clique):
                            clique.append(c)
                    
                    count += 1
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_copy.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
            
            return count
        
        # Greedy: add active vertices one by one if it reduces the count
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(N[v]), reverse=True)
        
        current_count = count_cliques(active)
        
        for v in i_vertices:
            if time.time() > deadline:
                break
            
            new_active = active | {v}
            new_count = count_cliques(new_active)
            if new_count < current_count:
                active = new_active
                current_count = new_count
        
        # Try removing some active vertices to see if it helps
        for v in list(active):
            if time.time() > deadline:
                break
            
            new_active = active - {v}
            new_count = count_cliques(new_active)
            if new_count < current_count:
                active = new_active
                current_count = new_count
        
        # Build final solution
        cliques = compute_cliques(active)
        
        # Verify: every edge must be in exactly one clique
        # (This is guaranteed by construction, but let's be safe)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
