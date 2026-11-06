import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s

    def build_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_cliques(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X to minimize branching
            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
            # Pick pivot with max degree in P
            best_u = max(pivot_candidates, key=lambda u: len(adj[u] & P))
            # Branch on P \ N(u)
            for v in list(P - adj[best_u]):
                bk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_clique_transversal(cliques):
        """Greedy set cover: pick vertex covering most uncovered cliques."""
        if not cliques:
            return set()
        
        uncovered = list(cliques)
        # Count how many cliques each vertex appears in
        from collections import defaultdict
        vert_count = defaultdict(int)
        for c in cliques:
            for v in c:
                vert_count[v] += 1
        
        selected = set()
        uncovered_set = set(range(len(cliques)))
        
        while uncovered_set:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in vert_count:
                if v in selected:
                    continue
                count = sum(1 for i in uncovered_set if v in cliques[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            # Remove covered cliques
            uncovered_set = {i for i in uncovered_set if best_v not in cliques[i]}
        
        return selected

    def local_improve(selected, cliques):
        """Try to remove vertices from selected while maintaining validity."""
        selected = set(selected)
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                candidate = selected - {v}
                # Check if candidate still meets all cliques
                valid = all(any(x in candidate for x in c) for c in cliques)
                if valid:
                    selected = candidate
                    improved = True
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > DEADLINE:
            # Fallback: return all non-isolated vertices
            edges = build_graph(n, p, seed)
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
            continue
        
        edges = build_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques
        cliques = bron_kerbosch_cliques(adj, n)
        
        # Filter to size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy set cover
        selected = greedy_clique_transversal(cliques)
        
        # Local improvement
        selected = local_improve(selected, cliques)
        
        results.append(sorted(selected))
    
    return results
