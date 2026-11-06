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

    def popcount(x):
        return bin(x).count('1')

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Track remaining edges as a set of tuples for easy removal, 
        # but we will reconstruct active adj from it or update it.
        # For efficiency, we maintain a set of remaining edges.
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        
        cliques = []

        def find_max_clique_bk(active_adj, n):
            """Bron-Kerbosch with pivot on graph defined by active_adj bitmasks."""
            best_clique = []
            best_size = 0
            full_mask = (1 << n) - 1
            # Limit depth or time to prevent explosion, though for dense graphs it's usually fast
            # We use a simple recursive BK with pivot
            
            def bronk(R, P, X):
                nonlocal best_clique, best_size
                if time.time() > deadline - 15:
                    return
                if P == 0 and X == 0:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                if P == 0:
                    return
                # Pivot selection: choose v in P U X that maximizes |N(v) \cap P|
                union = P | X
                if union == 0:
                    return
                max_deg = -1
                pivot = -1
                tmp = union
                while tmp:
                    lsb = tmp & (-tmp)
                    u = lsb.bit_length() - 1
                    d = popcount(active_adj[u] & P)
                    if d > max_deg:
                        max_deg = d
                        pivot = u
                    tmp ^= lsb
                
                # Candidates: vertices in P not adjacent to pivot (and not pivot itself)
                candidates = P & ~(active_adj[pivot] | (1 << pivot))
                
                while candidates:
                    lsb = candidates & (-candidates)
                    v = lsb.bit_length() - 1
                    candidates ^= lsb
                    bronk(R + [v], P & active_adj[v], X & active_adj[v])
                    P ^= lsb
                    X |= lsb
                    if time.time() > deadline - 15:
                        return

            bronk([], full_mask, 0)
            return best_clique if len(best_clique) >= 2 else None

        def greedy_clique_bitset(active_adj, n):
            """Fast greedy clique construction using bitsets."""
            active_vertices = [v for v in range(n) if active_adj[v] != 0]
            if not active_vertices:
                return None
            
            # Sort by degree in the active graph to start with high-degree nodes
            active_vertices.sort(key=lambda v: popcount(active_adj[v]), reverse=True)
            
            best_clique = None
            best_size = 0

            # Try a subset of start vertices to save time
            limit = min(len(active_vertices), 50)
            for start in active_vertices[:limit]:
                if time.time() > deadline - 15:
                    break
                clique = [start]
                P = active_adj[start] & ~(1 << start)
                while P:
                    # Pick vertex in P with max degree in P
                    best_v = -1
                    best_d = -1
                    tmp = P
                    while tmp:
                        lsb = tmp & (-tmp)
                        v = lsb.bit_length() - 1
                        d = popcount(active_adj[v] & P)
                        if d > best_d:
                            best_d = d
                            best_v = v
                        tmp ^= lsb
                    if best_v < 0:
                        break
                    clique.append(best_v)
                    P = P & active_adj[best_v]
                    P &= ~(1 << best_v)
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique

            return best_clique if best_clique and len(best_clique) >= 2 else None

        while remaining_edges:
            if time.time() > deadline - 20:
                # Emergency: pack remaining edges as pairs
                rem_list = list(remaining_edges)
                for e in rem_list:
                    cliques.append(list(e))
                break

            # Build active adjacency from remaining edges
            active_adj = [0] * n
            for u, v in remaining_edges:
                active_adj[u] |= (1 << v)
                active_adj[v] |= (1 << u)

            # Try greedy first (fast)
            clique = greedy_clique_bitset(active_adj, n)

            # If greedy found a small clique and we have time, try BK for larger
            # BK is expensive, so only try if greedy is poor (< 4) and time is safe
            if clique is not None and len(clique) < 4 and time.time() < deadline - 30:
                bk_clique = find_max_clique_bk(active_adj, n)
                if bk_clique is not None and len(bk_clique) > len(clique):
                    clique = bk_clique

            if clique is None or len(clique) < 2:
                # Fallback: take one edge
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.discard(e)
                continue

            # Remove all edges within the clique from remaining_edges
            clique_set = set(clique)
            to_remove = set()
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    to_remove.add((u, v))
            
            if to_remove:
                cliques.append(clique)
                remaining_edges.difference_update(to_remove)
            else:
                # Shouldn't happen if clique is valid, but safety fallback
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.discard(e)

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
