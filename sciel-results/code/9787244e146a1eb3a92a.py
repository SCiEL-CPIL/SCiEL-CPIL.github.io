"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t_start = time.time()
    DEADLINE = 145.0

    def generate_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        m = len(edges)

        # Track which edges are still uncovered
        edge_set = set(range(m))
        # edge_to_vertices: for each edge index, the two vertices
        # We need to find cliques in the graph and assign all their edges

        # Strategy: repeatedly find a large clique, assign all its edges, remove them
        # Use greedy: pick a vertex with high degree, extend to maximal clique

        remaining_edges = list(edges)
        remaining_set = set(range(m))
        result_cliques = []

        # Precompute adjacency for quick clique checks
        # For finding cliques: use greedy coloring-like approach

        # Better: sort vertices by degree descending
        deg = [len(adj[v]) for v in range(n)]
        vertices_sorted = sorted(range(n), key=lambda v: -deg[v])

        # For each "center" vertex, try to build a large clique containing it
        # Actually, let's use a different approach:
        # Find maximal cliques greedily

        # Simpler greedy: 
        # 1. While there are uncovered edges:
        #    a. Find the edge with the most "potential" (both endpoints have many common neighbors)
        #    b. Build a maximal clique around it
        #    c. Assign all edges of that clique to this group

        uncovered = set(range(m))
        # edge_endpoints[i] = (u, v)
        edge_endpoints = edges

        while uncovered:
            if time.time() > DEADLINE:
                # Fallback: assign each remaining edge as its own clique
                for ei in uncovered:
                    u, v = edge_endpoints[ei]
                    result_cliques.append([u, v])
                uncovered = set()
                break

            # Find the edge whose endpoints have the most common neighbors (among uncovered edges)
            # This is expensive; use a heuristic: pick edge where both endpoints have high degree
            best_e = None
            best_score = -1
            # Sample or iterate
            for ei in uncovered:
                u, v = edge_endpoints[ei]
                # Score: number of common neighbors
                common = len(adj[u] & adj[v])
                if common > best_score:
                    best_score = common
                    best_e = ei
                if time.time() > DEADLINE - 5:
                    break

            if best_e is None:
                best_e = next(iter(uncovered))
                u, v = edge_endpoints[best_e]
                result_cliques.append([u, v])
                uncovered.remove(best_e)
                continue

            u, v = edge_endpoints[best_e]
            # Build maximal clique containing u and v
            # Start with {u, v}, try to add vertices
            clique = {u, v}
            candidates = adj[u] & adj[v]  # common neighbors
            
            # Greedily add vertices with highest degree within candidates
            cand_list = list(candidates)
            cand_list.sort(key=lambda w: -len(adj[w] & clique))
            
            for w in cand_list:
                # Check if w is adjacent to all in clique
                if all(w in adj[x] for x in clique):
                    clique.add(w)
                    # Update candidates: only those adjacent to w
                    candidates = candidates & adj[w]
            
            # Now clique is a maximal-ish clique. Assign all edges within clique.
            clique_list = sorted(clique)
            assigned_edges = []
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if (a, b) in adj[a]:  # should be, since clique
                        # Find edge index
                        pass
            
            # Need to map (a,b) to edge index
            # Build lookup
            # This is getting complex. Let me use a dict.
            
            # Actually, let me precompute edge lookup
            break  # This approach is getting too complex for the time budget

        # Simpler approach: just use the line-graph coloring greedy
        # Reset
        result_cliques = []
        remaining = set(range(m))
        
        # Build vertex-to-edges map
        vert_edges = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            vert_edges[u].append(i)
            vert_edges[v].append(i)
            
        # Greedy: repeatedly find a large clique in the remaining graph
        # A clique in original graph = set of vertices all mutually adjacent
        # All edges among them go into one clique partition element
        
        # For efficiency, limit clique size search
        max_clique_size = 12  # heuristic limit
        
        while remaining:
            if time.time() > DEADLINE:
                for ei in remaining:
                    u, v = edges[ei]
                    result_cliques.append([u, v])
                remaining = set()
                break
            
            # Find a good seed edge
            # Pick edge with max common neighbors among remaining
            best_e = None
            best_common = -1
            for ei in remaining:
                u, v = edges[ei]
                # Count common neighbors that are still "available"
                common = 0
                for w in adj[u]:
                    if w in adj[v]:
                        common += 1
                if common > best_common:
                    best_common = common
                    best_e = ei
            
            if best_e is None:
                break
            
            u, v = edges[best_e]
            # Build clique around (u,v)
            clique = {u, v}
            # Try to extend
            candidates = set(adj[u] & adj[v])
            
            # Greedy extension
            changed = True
            while changed and len(clique) < max_clique_size:
                changed = False
                for w in sorted(candidates, key=lambda x: -len(adj[x])):
                    if all(w in adj[x] for x in clique):
                        clique.add(w)
                        candidates &= adj[w]
                        changed = True
                        break
            
            # Assign all edges within clique
            clique_list = sorted(clique)
            edges_in_clique = []
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if b in adj[a]:
                        # Find edge index
                        # Linear search is slow; use a set for remaining
                        for ei in list(remaining):
                            eu, ev = edges[ei]
                            if (eu == a and ev == b) or (eu == b and ev == a):
                                edges_in_clique.append(ei)
                                break
            
            if edges_in_clique:
                result_cliques.append(clique_list)
                for ei in edges_in_clique:
                    remaining.discard(ei)
            else:
                # Just assign this single edge
                result_cliques.append([u, v])
                remaining.discard(best_e)
        
        return result_cliques

    # The edge lookup is O(m) per edge which is too slow. Let me use a dict.
    def solve_instance_v3(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        m = len(edges)
        
        # Edge lookup: (min(u,v), max(u,v)) -> edge index
        edge_lookup = {}
        for i, (u, v) in enumerate(edges):
            key = (min(u, v), max(u, v))
            edge_lookup[key] = i
        
        remaining = set(range(m))
        result_cliques = []
        max_clique_size = 15
        
        while remaining:
            if time.time() > DEADLINE:
                for ei in remaining:
                    u, v = edges[ei]
                    result_cliques.append([u, v])
                remaining = set()
                break
            
            # Find seed edge with max common neighbors
            best_e = None
            best_common = -1
            for ei in remaining:
                u, v = edges[ei]
                common = len(adj[u] & adj[v])
                if common > best_common:
                    best_common = common
                    best_e = ei
            
            if best_e is None:
                break
            
            u, v = edges[best_e]
            clique = {u, v}
            candidates = adj[u] & adj[v]
            
            # Greedy extension
            for _ in range(max_clique_size - 2):
                best_w = None
                best_score = -1
                for w in candidates:
                    # Check w adjacent to all in clique
                    if all(w in adj[x] for x in clique):
                        score = len(adj[w])
                        if score > best_score:
                            best_score = score
                            best_w = w
                if best_w is None:
                    break
                clique.add(best_w)
                candidates &= adj[best_w]
            
            # Assign all edges within clique
            clique_list = sorted(clique)
            edges_in_clique = []
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    key = (a, b)
                    if key in edge_lookup:
                        ei = edge_lookup[key]
                        if ei in remaining:
                            edges_in_clique.append(ei)
            
            if edges_in_clique:
                result_cliques.append(clique_list)
                for ei in edges_in_clique:
                    remaining.discard(ei)
            else:
                result_cliques.append([u, v])
                remaining.discard(best_e)
        
        return result_cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > DEADLINE:
            edges, adj = generate_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        results.append(solve_instance_v3(n, p, seed))

    return results
# EVOLVE-BLOCK-END
