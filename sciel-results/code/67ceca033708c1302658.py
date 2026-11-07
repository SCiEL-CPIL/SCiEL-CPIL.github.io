import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Map K-edges to indices for bitset representation
        k_edges = []
        edge_map = {}
        for u in range(k):
            for v in range(u + 1, k):
                idx = len(k_edges)
                k_edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
        
        # Calculate A_v (bitset of K-edges covered by clique {v} U N(v))
        # Only consider v with small N(v) to keep holes small
        candidates = []
        for v in range(k, n):
            Nv = neighbors[v]
            if len(Nv) < 2:
                continue
            # Calculate bitset
            mask = 0
            for i in range(len(Nv)):
                for j in range(i + 1, len(Nv)):
                    u, w = Nv[i], Nv[j]
                    if u > w:
                        u, w = w, u
                    mask |= (1 << edge_map[(u, w)])
            candidates.append((len(Nv), v, mask))
            
        # Sort by size of N(v) ascending? 
        # We want small holes. 
        # But gain is proportional to |N(v)|.
        # Let's try sorting by (len(Nv)) and picking those that fit.
        
        candidates.sort(key=lambda x: x[0])
        
        covered_mask = 0
        selected = []
        
        for size, v, mask in candidates:
            # Check if mask is disjoint from covered_mask
            if (mask & covered_mask) == 0:
                # Heuristic: Only accept if gain > estimated penalty
                # Gain = size - 1
                # Penalty is hard to estimate. 
                # If size is small (2,3,4), penalty is likely small.
                if size <= 5: # Only use small stars
                    selected.append(v)
                    covered_mask |= mask
                    
        # Build cliques
        cliques = []
        
        # 1. Star cliques
        for v in selected:
            clique = [v] + neighbors[v]
            cliques.append(clique)
            
        # 2. Cross 2-cliques for unselected v
        for v in range(k, n):
            if v not in selected:
                for u in neighbors[v]:
                    cliques.append([u, v])
        
        # 3. Cover remaining K-edges
        # Find edges not covered by stars
        rem_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                idx = edge_map[(u, v)]
                if not (covered_mask & (1 << idx)):
                    rem_edges.append((u, v))
        
        # Greedy clique cover for rem_edges
        # Build adjacency for rem_edges graph
        adj = {u: set() for u in range(k)}
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        covered_rem = set()
        for u, v in rem_edges:
            covered_rem.add((u, v))
            covered_rem.add((v, u))
            
        while rem_edges:
            # Find a maximal clique
            # Start with a random edge
            u, v = rem_edges[0]
            clique = [u, v]
            # Try to add more vertices
            common = adj[u] & adj[v]
            for w in common:
                if (u, w) in covered_rem and (v, w) in covered_rem:
                    clique.append(w)
                    covered_rem.discard((u, w))
                    covered_rem.discard((w, u))
                    covered_rem.discard((v, w))
                    covered_rem.discard((w, v))
                    adj[u].discard(w)
                    adj[v].discard(w)
                    adj[w].discard(u)
                    adj[w].discard(v)
                    # Remove from rem_edges
                    rem_edges.remove((u, w))
                    rem_edges.remove((v, w))
                    
            cliques.append(clique)
            # Remove edges of clique from rem_edges
            clique_set = set(clique)
            new_rem = []
            for e in rem_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_rem.append(e)
            rem_edges = new_rem

        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
