import time
import random
from collections import defaultdict
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def get_clique_cover_of_k_edges(remaining_edges, k):
        """
        Greedy clique cover for the remaining edges in K.
        K is a clique, so any subset of vertices forms a clique.
        We need to cover the graph induced by remaining_edges with minimum cliques.
        This is equivalent to coloring the complement graph of the remaining edges.
        Since k is small (<=30), we can use a simple greedy coloring on the complement.
        """
        if not remaining_edges:
            return []
        
        # Build adjacency matrix of the "conflict" graph (complement of remaining edges)
        # If (u,v) is NOT in remaining_edges, they can be in the same clique.
        # If (u,v) IS in remaining_edges, they MUST be in different cliques.
        # So we are coloring the graph G_rem where edges are remaining_edges.
        # Wait. A clique in the original K covers all edges between its vertices.
        # If we pick a clique C, it covers all edges (u,v) with u,v in C.
        # We want to cover all edges in remaining_edges.
        # This is the Clique Cover problem on the graph H=(K, remaining_edges).
        # Since H is a subgraph of a clique, it's a general graph on k vertices.
        # Minimum clique cover is NP-hard, but k=30 is small.
        
        # Greedy:
        # 1. Pick an uncovered edge (u,v).
        # 2. Find the largest clique C in H containing (u,v).
        # 3. Add C to cover, remove all edges in C from H.
        
        rem_set = set(remaining_edges)
        cliques = []
        
        # Precompute adjacency for H
        adj = {i: set() for i in range(k)}
        for u, v in rem_set:
            adj[u].add(v)
            adj[v].add(u)
        
        while rem_set:
            # Pick an edge, preferably one with high degree to reduce rem_set fast
            # Or just pick the first one
            e = next(iter(rem_set))
            u, v = e
            clique = {u, v}
            
            # Try to expand the clique
            # A vertex x can be added to clique if it is connected to ALL vertices in clique
            # i.e., (x, y) is in rem_set for all y in clique
            # Note: In H, edges are the remaining edges.
            # So yes, x must be adjacent to all in clique in H.
            
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x in clique:
                        continue
                    if all((min(x, y), max(x, y)) in rem_set for y in clique):
                        clique.add(x)
                        changed = True
            
            cliques.append(sorted(clique))
            
            # Remove all edges covered by this clique
            for a, b in itertools.combinations(clique, 2):
                rem_set.discard((min(a, b), max(a, b)))
                
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Precompute all edges in K
        k_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        
        # For each v in I, we want to find a matching in N(v)
        # Strategy: Generate candidate matchings for each v, then solve Set Packing
        
        best_cliques = None
        best_count = float('inf')
        
        # Number of random restarts for candidate generation
        num_restarts = 20
        # Number of candidate matchings per vertex per restart
        num_candidates = 50
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # 1. Generate candidate matchings for each v
            # candidates[v] is a list of matchings (list of edges)
            candidates = {v: [] for v in range(k, n)}
            
            for v in range(k, n):
                N_v = neighbors[v]
                if len(N_v) < 2:
                    continue
                    
                # Generate random matchings in the graph induced by N_v on K
                # Note: N_v is a subset of K. The graph induced by N_v on K is a clique!
                # So ANY matching in N_v is just a set of disjoint pairs from N_v.
                # We want large matchings.
                
                # Simple way: Randomly shuffle N_v and greedily pick disjoint pairs?
                # Since it's a clique, we can just pick any disjoint pairs.
                # To get variety, we can use random sampling.
                
                # Better: Use a simple heuristic to find a near-max matching.
                # Since it's a clique, max matching size is floor(|N_v|/2).
                # We can just generate random subsets of edges that form matchings.
                
                for _ in range(num_candidates):
                    # Random matching
                    avail = list(N_v)
                    random.shuffle(avail)
                    matching = []
                    i = 0
                    while i < len(avail) - 1:
                        # Pick two random available vertices
                        # To ensure it's a matching, we just pick disjoint pairs.
                        # In a clique, any disjoint pairs form a matching.
                        u = avail[i]
                        w = avail[i+1]
                        matching.append((min(u, w), max(u, w)))
                        i += 2
                    candidates[v].append(matching)
            
            # 2. Solve Set Packing
            # Items: (v, matching)
            # Weight: len(matching)
            # Conflict: same v, or shared edge
            
            # Build list of items
            items = []
            for v in range(k, n):
                for m in candidates[v]:
                    if m:
                        items.append((v, m, len(m)))
            
            # Sort by weight descending
            items.sort(key=lambda x: -x[2])
            
            # Greedy selection
            used_edges = set()
            used_vs = set()
            selected = []
            
            for v, m, w in items:
                if v in used_vs:
                    continue
                # Check if edges are disjoint from used_edges
                valid = True
                for e in m:
                    if e in used_edges:
                        valid = False
                        break
                if valid:
                    selected.append((v, m))
                    used_vs.add(v)
                    for e in m:
                        used_edges.add(e)
            
            # 3. Construct Cliques
            cliques = []
            
            # Add cliques for selected (v, matching)
            for v, m in selected:
                # The clique is {v} U {u, w for (u,w) in m}
                verts = [v]
                for u, w in m:
                    verts.append(u)
                    verts.append(w)
                cliques.append(sorted(verts))
            
            # Add cliques for unselected v
            for v in range(k, n):
                if v in used_vs:
                    continue
                # v is unassigned. Cover each edge (v, u) with a clique {v, u}
                for u in neighbors[v]:
                    cliques.append([u, v])
            
            # Add cliques for remaining K edges
            remaining_k_edges = [e for e in k_edges if e not in used_edges]
            if remaining_k_edges:
                k_cliques = get_clique_cover_of_k_edges(remaining_k_edges, k)
                cliques.extend(k_cliques)
            
            # 4. Validate and Count
            # Check types and validity
            valid = True
            edge_count = defaultdict(int)
            for c in cliques:
                # Ensure ints
                if not all(isinstance(x, int) for x in c):
                    valid = False
                    break
                if len(c) < 2:
                    valid = False
                    break
                # Check distinct
                if len(set(c)) != len(c):
                    valid = False
                    break
                # Count edges
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        u, w = min(c[i], c[j]), max(c[i], c[j])
                        edge_count[(u, w)] += 1
            
            if not valid:
                continue
            
            # Check all edges covered exactly once
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
            
            for e in all_edges:
                if edge_count[e] != 1:
                    valid = False
                    break
            for e, cnt in edge_count.items():
                if e not in all_edges:
                    valid = False
                    break
            
            if not valid:
                continue
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [[int(x) for x in c] for c in cliques]
        
        if best_cliques is None:
            # Fallback
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]
        
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    while len(results) < 3:
        results.append([])

    return results
