import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency list and bitmask for fast intersection
        adj = [set() for _ in range(n)]
        adj_mask = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Bron-Kerbosch with pivoting to find all maximal cliques
        # R: current clique, P: candidates, X: excluded
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Choose pivot u from P union X to minimize branching
            # We want u with max degree in P
            U = P.union(X)
            if not U:
                return
            u = max(U, key=lambda x: len(adj[x].intersection(P)))
            
            # Iterate over P \ N(u)
            candidates = P.difference(adj[u])
            for v in list(candidates):
                bronk(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)
                
        bronk(set(), set(range(n)), set())
        
        # If Bron-Kerbosch didn't find enough or timed out, fallback to greedy
        if not cliques:
            # Fallback: just use edges as cliques of size 2
            return [list(e) for e in edges]

        # Represent cliques as sets of edges for Set Cover
        clique_edges = []
        for c in cliques:
            c_edges = set()
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    u, v = c[i], c[j]
                    if u > v:
                        u, v = v, u
                    c_edges.add((u, v))
            if c_edges:
                clique_edges.append((c, c_edges))
        
        # Greedy Set Cover
        remaining_edges = set(edges)
        partition = []
        
        # Sort cliques by size descending to potentially pick larger ones first?
        # Or just iterate. The greedy step picks the one covering most remaining edges.
        # To speed up, we can pre-compute edge counts or just calculate on the fly.
        
        # Optimization: If we have many cliques, checking intersection for each is O(|cliques| * |edges|).
        # For n=64, max edges ~ 2000. Max cliques could be thousands.
        # 2000 * 5000 = 10^7 operations per step. If we need 100 steps, 10^9 ops. Might be slow in Python.
        # Let's try to be efficient.
        
        # We can use bitmasks for edges if we map edges to indices?
        # Or just rely on the fact that maximal cliques in dense graphs are large, so few are needed.
        
        # Let's use a simple greedy:
        # 1. For each candidate clique, count how many of its edges are in remaining_edges.
        # 2. Pick the one with max count.
        # 3. Remove those edges from remaining_edges.
        # 4. Repeat until remaining_edges is empty.
        
        # To speed up, we can maintain a list of active cliques.
        active_cliques = [(c, ce) for c, ce in clique_edges]
        
        while remaining_edges and time.time() < deadline:
            best_idx = -1
            best_count = 0
            best_clique = None
            best_clique_edges = None
            
            # We can prune cliques that have 0 remaining edges
            new_active = []
            for idx, (c, ce) in enumerate(active_cliques):
                # Calculate intersection size
                # ce is a set of tuples. remaining_edges is a set of tuples.
                # len(ce & remaining_edges)
                count = len(ce & remaining_edges)
                if count > 0:
                    new_active.append((c, ce))
                    if count > best_count:
                        best_count = count
                        best_idx = idx
                        best_clique = c
                        best_clique_edges = ce
            
            if best_idx == -1:
                # No clique covers any remaining edge? This shouldn't happen if we enumerated all maximal cliques.
                # Unless the graph is not covered by maximal cliques? Impossible.
                # Or if we ran out of time in Bron-Kerbosch and didn't find all.
                # Fallback: pick arbitrary edge
                break
                
            # Add best clique to partition
            partition.append(best_clique)
            
            # Remove covered edges
            remaining_edges -= best_clique_edges
            
            # Update active cliques list (remove the one we used? No, we can reuse? No, edges are gone.)
            # Actually, we just rebuild active_cliques from new_active in next iteration?
            # It's better to just filter active_cliques.
            active_cliques = new_active
            
        # Fallback for any remaining edges (should be empty if maximal cliques were found correctly)
        if remaining_edges:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
