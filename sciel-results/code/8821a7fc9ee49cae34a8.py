import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges, seed_offset=0):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
            
        # 1. Initial Greedy Partition
        remaining_edges = set(edge_set)
        cliques_sets = []
        
        # Use a fixed random seed for the greedy to be deterministic but varied by offset
        rng_greedy = random.Random(seed_offset)
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques_sets.append(set(e))
                break
                
            deg = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Add some randomness to tie-breaking
            # Sort by degree, but shuffle ties
            vertices = list(range(n))
            # Simple: just pick max degree. To vary, we can shuffle vertices with same degree.
            # For now, keep deterministic max degree for stability, 
            # but we will use multiple restarts with different initial seeds? 
            # No, the graph is fixed. We need different greedy paths.
            # We can randomize the order of candidate selection in the greedy growth.
            
            v_start = max(deg, key=lambda x: (deg[x], rng_greedy.random()))
            
            clique = [v_start]
            candidates = set()
            for u, v in remaining_edges:
                if u == v_start:
                    candidates.add(v)
                elif v == v_start:
                    candidates.add(u)
            
            while candidates:
                # Randomize choice among top candidates?
                # Pick one with max degree, tie-break with random
                u_next = max(candidates, key=lambda x: (deg[x], rng_greedy.random()))
                clique.append(u_next)
                
                new_candidates = set()
                for u, v in remaining_edges:
                    if u == u_next:
                        if v in candidates:
                            new_candidates.add(v)
                    elif v == u_next:
                        if u in candidates:
                            new_candidates.add(u)
                candidates = new_candidates
            
            for u, v in combinations(clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            cliques_sets.append(set(clique))

        # 2. Local Search with Perturbation
        best_cliques = [set(c) for c in cliques_sets]
        best_count = len(best_cliques)
        
        while time.time() < deadline:
            # Optimize current state
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                
                # Merge
                i = 0
                while i < len(cliques_sets):
                    if time.time() > deadline:
                        break
                    j = i + 1
                    while j < len(cliques_sets):
                        if time.time() > deadline:
                            break
                        C_i = cliques_sets[i]
                        C_j = cliques_sets[j]
                        union = C_i.union(C_j)
                        
                        is_clique = True
                        for u, v in combinations(union, 2):
                            if v not in adj[u]:
                                is_clique = False
                                break
                        
                        if is_clique:
                            cliques_sets[i] = union
                            del cliques_sets[j]
                            improved = True
                            break
                        else:
                            j += 1
                    i += 1
                
                # Move (only if reduces count)
                for i in range(len(cliques_sets)):
                    if time.time() > deadline:
                        break
                    C_i = cliques_sets[i]
                    if len(C_i) <= 1:
                        continue
                    for v in list(C_i):
                        moved = False
                        for j in range(len(cliques_sets)):
                            if i == j:
                                continue
                            if time.time() > deadline:
                                break
                            C_j = cliques_sets[j]
                            is_clique = all(u in adj[v] for u in C_j)
                            if is_clique:
                                C_i_rem = C_i.difference({v})
                                if len(C_i_rem) <= 1:
                                    is_valid_rem = True
                                else:
                                    is_valid_rem = True
                                    for u, w in combinations(C_i_rem, 2):
                                        if w not in adj[u]:
                                            is_valid_rem = False
                                            break
                                if is_valid_rem:
                                    cliques_sets[j] = C_j.union({v})
                                    cliques_sets[i] = C_i_rem
                                    if len(cliques_sets[i]) == 0:
                                        del cliques_sets[i]
                                        improved = True
                                        moved = True
                                        break
                        if moved:
                            break
                
                # Update best
                if len(cliques_sets) < best_count:
                    best_count = len(cliques_sets)
                    best_cliques = [set(c) for c in cliques_sets]
            
            # If we are stuck (no improvement in last pass), Perturb
            if not improved:
                # Simple Perturbation: Swap a vertex between two cliques if valid
                # Or just restart greedy with new random seed for tie-breaking?
                # Restarting greedy is expensive but effective.
                # Let's try a few restarts.
                pass # Placeholder for perturbation logic if we want to be fancy.
                # For now, let's just break if no improvement to save time for next graph?
                # No, we should use the time.
                # Let's try a random swap perturbation.
                if len(cliques_sets) > 1:
                    # Pick two random cliques
                    idx1, idx2 = random.sample(range(len(cliques_sets)), 2)
                    C1 = cliques_sets[idx1]
                    C2 = cliques_sets[idx2]
                    if len(C1) > 1 and len(C2) > 1:
                        v1 = random.choice(list(C1))
                        v2 = random.choice(list(C2))
                        
                        # Check if C1 - {v1} + {v2} is clique
                        C1_new = C1.difference({v1}).union({v2})
                        is_c1 = True
                        for u, v in combinations(C1_new, 2):
                            if v not in adj[u]:
                                is_c1 = False
                                break
                        # Check if C2 - {v2} + {v1} is clique
                        C2_new = C2.difference({v2}).union({v1})
                        is_c2 = True
                        for u, v in combinations(C2_new, 2):
                            if v not in adj[u]:
                                is_c2 = False
                                break
                        
                        if is_c1 and is_c2:
                            cliques_sets[idx1] = C1_new
                            cliques_sets[idx2] = C2_new
                            # Continue loop to re-optimize

        return [list(c) for c in best_cliques]

    c1 = solve(random_graph(32, 0.6, 601), 0)
    c2 = solve(random_graph(64, 0.55, 602), 1)
    
    return [c1, c2]
