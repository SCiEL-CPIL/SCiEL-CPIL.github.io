import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        partition = []
        covered_internal = set()
        
        # Process vertices in I
        for v in I:
            N_v = sorted(adj[v] & set(C))
            S = list(N_v)
            
            # Resolve conflicts for internal edges
            # We need E(S) to be disjoint from covered_internal
            # Greedy removal
            changed = True
            while changed:
                changed = False
                # Check for conflicts
                # To be efficient, we can just check if any edge in S is in covered_internal
                # But we need to remove vertices to break conflicts.
                # Let's find a vertex in S that is part of a conflicting edge.
                conflicting_vertex = None
                for i in range(len(S)):
                    u = S[i]
                    for j in range(i+1, len(S)):
                        w = S[j]
                        edge = (min(u, w), max(u, w))
                        if edge in covered_internal:
                            conflicting_vertex = u
                            break
                    if conflicting_vertex is not None:
                        break
                if conflicting_vertex is not None:
                    S.remove(conflicting_vertex)
                    changed = True
            
            # Add clique(s) for v
            if len(S) >= 1:
                # Clique is {v} U S
                clique = [v] + S
                partition.append(clique)
                # Mark internal edges in S as covered
                for i in range(len(S)):
                    u = S[i]
                    for j in range(i+1, len(S)):
                        w = S[j]
                        covered_internal.add((min(u, w), max(u, w)))
            else:
                # No internal edges can be covered (or S was empty)
                # Cover cross edges individually
                for u in N_v:
                    partition.append([v, u])
        
        # Cover remaining internal edges
        # E(C) is all pairs in C
        # E_rem = E(C) - covered_internal
        E_rem = []
        for i in range(k):
            u = C[i]
            for j in range(i+1, k):
                w = C[j]
                edge = (u, w)
                if edge not in covered_internal:
                    E_rem.append(edge)
        
        # Greedy clique cover for E_rem
        # Vertices involved in E_rem
        rem_adj = {u: set() for u in C}
        for u, w in E_rem:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
            
        active_edges = set(E_rem)
        
        while active_edges:
            # Find largest clique in active_edges
            # Simple heuristic: pick a vertex with max degree, try to form clique
            # Or just pick the first edge and expand?
            # For small k, we can be a bit more thorough.
            # Let's try to find the max clique greedily.
            best_clique = None
            best_size = 0
            
            # Try to build clique from each vertex
            for start_v in rem_adj:
                if not rem_adj[start_v]:
                    continue
                # Greedy expansion
                candidates = [start_v]
                # Sort neighbors by degree descending?
                neighbors = sorted(rem_adj[start_v], key=lambda x: len(rem_adj[x]), reverse=True)
                for nb in neighbors:
                    # Check if nb is connected to all in candidates
                    is_connected = all(nb in rem_adj[c] for c in candidates)
                    if is_connected:
                        candidates.append(nb)
                
                if len(candidates) > best_size:
                    best_size = len(candidates)
                    best_clique = candidates
                    
            if best_clique is None or len(best_clique) < 2:
                # Should not happen if active_edges is not empty
                break
                
            # Add clique to partition
            partition.append(best_clique)
            
            # Remove edges covered by best_clique from active_edges
            clique_set = set(best_clique)
            for i in range(len(best_clique)):
                u = best_clique[i]
                for j in range(i+1, len(best_clique)):
                    w = best_clique[j]
                    edge = (min(u, w), max(u, w))
                    if edge in active_edges:
                        active_edges.remove(edge)
                        
            # Check deadline
            if time.time() > deadline:
                break
                
        return partition

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to simple valid solution if timeout
            # Just edges
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        part = solve_instance(*spec)
        # Verify types
        # Convert to lists of ints
        clean_part = []
        for clique in part:
            clean_clique = [int(x) for x in clique]
            clean_part.append(clean_clique)
        results.append(clean_part)
        
    return results
