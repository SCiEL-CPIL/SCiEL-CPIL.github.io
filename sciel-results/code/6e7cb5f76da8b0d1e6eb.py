import time
from random import Random
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Bitmask adjacency of the remaining-edge graph
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        cliques = []

        # Fast Greedy Clique Finder
        # Tries to expand from a given start vertex
        def expand_clique(start, adj, n):
            clique = [start]
            P = adj[start] & ~(1 << start)
            while P:
                # Select vertex in P with max degree within P
                best_v = -1
                best_d = -1
                tmp = P
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    # Degree of v in P is popcount(adj[v] & P)
                    d = bin(adj[v] & P).count('1')
                    if d > best_d:
                        best_d = d
                        best_v = v
                    tmp ^= lsb
                
                if best_v < 0:
                    break
                
                clique.append(best_v)
                # Update P to be neighbors of best_v within current P
                P = P & adj[best_v]
                P &= ~(1 << best_v)
            
            return clique

        # Find best clique by trying top K vertices by degree
        def find_best_clique(adj, n, K=10):
            if not any(adj):
                return None
            
            # Get degrees
            degrees = [(bin(adj[v]).count('1'), v) for v in range(n) if adj[v] != 0]
            if not degrees:
                return None
            
            # Sort by degree descending
            degrees.sort(reverse=True)
            
            best_clique = None
            best_size = 0
            
            for deg, v in degrees[:K]:
                c = expand_clique(v, adj, n)
                if len(c) > best_size:
                    best_size = len(c)
                    best_clique = c
            
            return best_clique

        # Track remaining edges as a set of (min,max) tuples
        remaining = set((min(u, v), max(u, v)) for u, v in edges)

        while remaining:
            if time.time() > deadline - 10:
                # Fallback: add remaining edges as individual cliques
                for e in list(remaining):
                    cliques.append(list(e))
                break

            # Find a large clique in the remaining-edge graph
            clique = find_best_clique(adj, n, K=10)

            if clique is None or len(clique) < 2:
                # If no clique found (shouldn't happen if edges exist, unless n=1?), pick an edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                continue

            # Remove all edges inside the clique from remaining and adj
            cset = set(clique)
            to_remove = set()
            for u, v in remaining:
                if u in cset and v in cset:
                    to_remove.add((u, v))
            
            if to_remove:
                cliques.append(clique)
                for u, v in to_remove:
                    remaining.discard((u, v))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                # Should not happen if clique was valid, but safety check
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
