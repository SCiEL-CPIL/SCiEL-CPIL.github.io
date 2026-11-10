"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Pivot selection to reduce recursion depth
            # Choose pivot u from P U X that maximizes |P \ N(u)|
            P_U_X = P.union(X)
            if P_U_X:
                # Simple heuristic: pick first or random, or max degree in P
                # For small n, just pick one
                u = next(iter(P_U_X))
                candidates = P.difference(adj[u])
            else:
                candidates = P
                
            for v in list(candidates):
                expand(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)
                
        expand(set(), set(range(n)), set())
        return maximal_cliques

    def solve_transversal(adj):
        n = len(adj)
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        # Map cliques to indices for faster checking
        # We want to find min S such that for all C, S ∩ C != empty
        
        # Greedy Set Cover
        best_s = set()
        best_size = n + 1
        
        # Multiple restarts to handle randomness in tie-breaking
        for _ in range(20):
            unhit = list(range(len(cliques)))
            s = set()
            
            while unhit:
                # Count how many unhit cliques each vertex covers
                counts = [0] * n
                for idx in unhit:
                    for v in cliques[idx]:
                        counts[v] += 1
                
                # Find max count
                max_count = max(counts)
                if max_count == 0:
                    # Should not happen if unhit is not empty and cliques are valid
                    break
                
                # Randomly pick one of the max count vertices
                candidates = [v for v in range(n) if counts[v] == max_count]
                v = random.choice(candidates)
                s.add(v)
                
                # Remove hit cliques
                new_unhit = []
                for idx in unhit:
                    if v not in cliques[idx]:
                        new_unhit.append(idx)
                unhit = new_unhit
            
            if len(s) < best_size:
                best_size = len(s)
                best_s = set(s)
                
            # Check time
            if time.time() - start_time > time_limit - 5:
                break

        # Local Search: Minimalization
        # Try to remove vertices from best_s if possible
        s = set(best_s)
        changed = True
        while changed:
            changed = False
            for v in list(s):
                # Check if removing v leaves any clique unhit
                valid = True
                for c in cliques:
                    if v in c:
                        # Check if there is another vertex in c that is in s
                        if not any(u in s for u in c if u != v):
                            valid = False
                            break
                if valid:
                    s.remove(v)
                    changed = True
                    if time.time() - start_time > time_limit - 5:
                        return sorted(s)
        
        return sorted(s)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > time_limit - 10:
            # Fallback: return all vertices (safe but large)
            adj = instance(n, p, seed)
            touched = set()
            for u in range(n):
                if adj[u]:
                    touched.add(u)
            out.append(sorted(touched))
            continue
        
        adj = instance(n, p, seed)
        sol = solve_transversal(adj)
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
