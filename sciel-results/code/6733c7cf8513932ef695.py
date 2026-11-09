import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P ∪ X with max |P ∩ N(u)|
            pivot = None
            max_inter = -1
            for u in list(P) + list(X):
                inter = len(P & adj[u])
                if inter > max_inter:
                    max_inter = inter
                    pivot = u
            
            if pivot is None:
                for v in list(P):
                    R.add(v)
                    bk(R, P & adj[v], X & adj[v])
                    R.remove(v)
                    P.remove(v)
                    X.add(v)
            else:
                # Only consider v in P not adjacent to pivot
                for v in list(P - adj[pivot]):
                    R.add(v)
                    bk(R, P & adj[v], X & adj[v])
                    R.remove(v)
                    P.remove(v)
                    X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex hitting most remaining cliques."""
        if not cliques:
            return set()
        
        # Map each vertex to set of clique indices it belongs to
        vertex_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        remaining = set(range(len(cliques)))
        hit = set()
        
        while remaining:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in hit:
                    continue
                count = len(vertex_to_cliques[v] & remaining)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count <= 0:
                break
            
            hit.add(best_v)
            remaining -= vertex_to_cliques[best_v]
        
        return hit

    def local_search_minimize(hit, cliques, n, time_limit):
        hit = set(hit)
        cliques_list = list(cliques)
        num_cliques = len(cliques_list)
        
        if num_cliques == 0:
            return hit

        # Precompute vertex to cliques mapping (list of indices)
        vertex_to_cliques = [[] for _ in range(n)]
        clique_sets = [set(c) for c in cliques_list]
        for i, clique in enumerate(cliques_list):
            for v in clique:
                vertex_to_cliques[v].append(i)
        
        # Initialize clique_hits
        clique_hits = [0] * num_cliques
        for v in hit:
            for c_idx in vertex_to_cliques[v]:
                clique_hits[c_idx] += 1
        
        changed = True
        while changed and (time.time() - start_time) < deadline - 10:
            changed = False
            
            # Phase 1: Deletions
            for v in list(hit):
                safe = True
                for c_idx in vertex_to_cliques[v]:
                    if clique_hits[c_idx] == 1:
                        safe = False
                        break
                if safe:
                    hit.remove(v)
                    for c_idx in vertex_to_cliques[v]:
                        clique_hits[c_idx] -= 1
                    changed = True
            
            # Phase 2: Swaps
            if (time.time() - start_time) < deadline - 10:
                for v in list(hit):
                    unique_cliques = []
                    for c_idx in vertex_to_cliques[v]:
                        if clique_hits[c_idx] == 1:
                            unique_cliques.append(c_idx)
                    
                    if unique_cliques:
                        # Intersection of these cliques
                        inter_set = set(clique_sets[unique_cliques[0]])
                        for c_idx in unique_cliques[1:]:
                            inter_set &= clique_sets[c_idx]
                        
                        # Find candidate u
                        best_u = -1
                        for u in inter_set:
                            if u not in hit:
                                best_u = u
                                break
                        
                        if best_u != -1:
                            # Perform swap
                            hit.remove(v)
                            for c_idx in vertex_to_cliques[v]:
                                clique_hits[c_idx] -= 1
                            
                            hit.add(best_u)
                            for c_idx in vertex_to_cliques[best_u]:
                                clique_hits[c_idx] += 1
                            
                            changed = True
                            break 
                            
        return hit

    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        # Filter to size >= 2 (already done in BK, but be safe)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Greedy hitting set
        hit = greedy_hitting_set(cliques, n)
        
        # Local search to minimize
        hit = local_search_minimize(hit, cliques, n, time_budget)
        
        return sorted(hit)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 5:
            # Not enough time, just use greedy
            adj = random_graph(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            cliques = [c for c in cliques if len(c) >= 2]
            if not cliques:
                sol = []
            else:
                sol = sorted(greedy_hitting_set(cliques, n))
        else:
            # Allocate time proportionally, leave buffer
            per_instance_time = (time_left - 5) / (3 - i)
            sol = solve_instance(n, p, seed, per_instance_time)
        
        results.append(sol)

    return results
