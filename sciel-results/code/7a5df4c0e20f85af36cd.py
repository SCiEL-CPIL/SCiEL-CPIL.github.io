"""Program for clique partition task using line-graph edge-coloring heuristic."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(edges, n):
        # adj[v] = set of vertices adjacent to v
        adj = [set() for _ in range(n)]
        # edge_to_edges: map from vertex to list of edges incident to it
        vertex_edges = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            adj[u].add(v)
            adj[v].add(u)
            vertex_edges[u].append(idx)
            vertex_edges[v].append(idx)
        return adj, vertex_edges

    def is_clique(vertex_set, adj):
        """Check if vertex_set forms a clique in the original graph."""
        vs = list(vertex_set)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[vs[i]]:
                    return False
        return True

    def solve_clique_partition(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        adj, vertex_edges = build_adjacency(edges, n)
        
        # For each edge, precompute which edges are "compatible" (can be in same clique)
        # Two edges (a,b) and (c,d) can be in the same clique if {a,b,c,d} forms a clique
        # i.e., all pairs among {a,b,c,d} must be edges.
        
        # Greedy: process edges, assign each to smallest-indexed clique where it fits
        # A clique is valid if adding the new edge keeps it a clique.
        # Two edges e1=(u1,v1) and e2=(u2,v2) are compatible if:
        #   - They share a vertex: say u1==u2, then need v1-v2 edge (and u1 is common)
        #   - They are disjoint: need all 4 cross-edges present
        #   - They share both vertices: same edge (not possible here)
        
        # For efficiency, precompute edge compatibility
        # edge_compatible[i][j] = True if edges i and j can be in same clique
        m = len(edges)
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        def edges_compatible(e1, e2):
            u1, v1 = e1
            u2, v2 = e2
            # Check all pairs among {u1,v1,u2,v2}
            verts = {u1, v1, u2, v2}
            for a in verts:
                for b in verts:
                    if a == b:
                        continue
                    if (min(a, b), max(a, b)) not in edge_set:
                        return False
            return True
        
        # Order edges by some heuristic: process edges in a way that helps packing
        # Sort by degree sum or just process in order
        # Let's try processing in order but with a twist: 
        # Actually, let's just do greedy in natural order first
        
        cliques = []  # list of lists of edges (each clique is a set of edges)
        edge_to_clique = [-1] * m
        
        for i in range(m):
            ei = edges[i]
            placed = False
            for ci in range(len(cliques)):
                # Check if ei can be added to clique ci
                compatible = True
                for ej in cliques[ci]:
                    if not edges_compatible(ei, ej):
                        compatible = False
                        break
                if compatible:
                    cliques[ci].append(ei)
                    edge_to_clique[i] = ci
                    placed = True
                    break
            if not placed:
                cliques.append([ei])
                edge_to_clique[i] = len(cliques) - 1
        
        # Convert edge cliques to vertex cliques
        # Each clique of edges corresponds to a vertex set (the union of endpoints)
        # But we need to ensure the vertex set itself is a clique
        result = []
        for clique_edges in cliques:
            verts = set()
            for u, v in clique_edges:
                verts.add(u)
                verts.add(v)
            if len(verts) >= 2:
                result.append(sorted(verts))
        
        return result

    g1 = solve_clique_partition(32, 0.6, 601)
    g2 = solve_clique_partition(64, 0.55, 602)
    return [g1, g2]
# EVOLVE-BLOCK-END
