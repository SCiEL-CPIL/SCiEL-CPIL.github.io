import time
from random import Random

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

        # Build adjacency bitmask
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Greedy phase: repeatedly find a large clique and remove its edges
        def expand_clique(start, adj):
            clique = [start]
            P = adj[start] & ~(1 << start)
            while P:
                best_v = -1
                best_d = -1
                tmp = P
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = bin(adj[v] & P).count('1')
                    if d > best_d:
                        best_d = d
                        best_v = v
                    tmp ^= lsb
                if best_v < 0:
                    break
                clique.append(best_v)
                P = P & adj[best_v]
                P &= ~(1 << best_v)
            return clique

        def find_best_clique(adj):
            if not any(adj):
                return None
            degrees = [(bin(adj[v]).count('1'), v) for v in range(n) if adj[v] != 0]
            if not degrees:
                return None
            degrees.sort(reverse=True)
            best_clique = None
            best_size = 0
            for deg, v in degrees[:15]:
                c = expand_clique(v, adj)
                if len(c) > best_size:
                    best_size = len(c)
                    best_clique = c
            return best_clique

        remaining = set((min(u, v), max(u, v)) for u, v in edges)
        cliques = []

        while remaining:
            if time.time() > deadline - 20:
                for e in list(remaining):
                    cliques.append(list(e))
                break

            clique = find_best_clique(adj)
            if clique is None or len(clique) < 2:
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                continue

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
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])

        # Local search improvement: try to merge cliques whose union is a clique
        # or reassign edges to reduce clique count
        def is_clique(vertices, full_adj):
            """Check if all pairs in vertices form edges in full_adj."""
            vs = set(vertices)
            for i, u in enumerate(vertices):
                for v in vertices[i+1:]:
                    if not (full_adj[u] >> v) & 1:
                        return False
            return True

        # Rebuild full adjacency for the original graph
        full_adj = [0] * n
        for u, v in edges:
            full_adj[u] |= (1 << v)
            full_adj[v] |= (1 << u)

        # Local search: try to merge pairs of cliques if their union is a clique
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            new_cliques = []
            merged = [False] * len(cliques)
            
            for i in range(len(cliques)):
                if merged[i]:
                    continue
                for j in range(i + 1, len(cliques)):
                    if merged[j]:
                        continue
                    # Try merging cliques[i] and cliques[j]
                    union = set(cliques[i]) | set(cliques[j])
                    if is_clique(list(union), full_adj):
                        # Check that all edges in the union are covered by these two cliques
                        # Actually, we need to verify that merging doesn't create conflicts
                        # The union must be a clique AND every edge in the union must be in exactly one of the original cliques
                        # Since they were separate cliques, edges within each are fine.
                        # We need: no edge between cliques[i] and cliques[j] exists in the original graph
                        # Wait - if there's an edge between them, it's not covered by either clique alone,
                        # so it must be in some other clique. Merging would then double-cover it.
                        # So we can only merge if there are NO edges between the two cliques.
                        has_cross_edge = False
                        for u in cliques[i]:
                            for v in cliques[j]:
                                if (full_adj[u] >> v) & 1:
                                    has_cross_edge = True
                                    break
                            if has_cross_edge:
                                break
                        if not has_cross_edge:
                            new_cliques.append(list(union))
                            merged[i] = True
                            merged[j] = True
                            improved = True
                            break
                if merged[i]:
                    continue
                new_cliques.append(cliques[i])
            
            if improved:
                cliques = new_cliques

        # Second local search: try to move edges from one clique to another to enable merges
        # This is more complex; for now, try a simpler approach:
        # For each clique, try removing one vertex and see if the remaining still covers its edges
        # and if the removed vertex's edges can be absorbed into other cliques
        
        # Simpler improvement: try to find if any clique has a proper subset that's also a clique
        # and if the "extra" edges can be reassigned. This is complex, so we'll do a heuristic:
        # For each pair of cliques, check if moving one edge from a larger clique to a smaller
        # one reduces the total count (unlikely in simple cases).
        
        # Actually, let's try a different structural improvement:
        # After the greedy phase, try to find larger cliques by looking at the complement structure.
        # But given time constraints, we'll stick with the merge-based local search.
        
        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
