"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot: vertex in P ∪ X with max degree in P
        pivot = None
        max_degree = -1
        for v in P | X:
            deg = len(adj[v] & P)
            if deg > max_degree:
                max_degree = deg
                pivot = v
        if pivot is None:
            for v in list(P):
                P.remove(v)
                X.add(v)
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                X.remove(v)
                P.add(v)
        else:
            for v in list(P - adj[pivot]):
                P.remove(v)
                X.add(v)
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                X.remove(v)
                P.add(v)
    
    def find_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        bron_kerbosch(adj, frozenset(), frozenset(range(n)), frozenset(), cliques)
        return cliques
    
    def greedy_hitting_set(cliques, n):
        if not cliques:
            return set()
        
        # Map each vertex to the set of clique indices it belongs to
        vertex_cliques = defaultdict(set)
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        uncovered = set(range(len(cliques)))
        hitting_set = set()
        
        while uncovered:
            # Find vertex that covers most uncovered cliques
            best_vertex = None
            best_count = -1
            for v in range(n):
                count = len(vertex_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_vertex = v
            if best_vertex is None or best_count == 0:
                break
            hitting_set.add(best_vertex)
            uncovered -= vertex_cliques[best_vertex]
        
        # Try to remove redundant vertices
        improved = True
        while improved:
            improved = False
            for v in list(hitting_set):
                # Check if v is needed
                without_v = hitting_set - {v}
                all_covered = True
                for clique in cliques:
                    if not (clique & without_v):
                        all_covered = False
                        break
                if all_covered:
                    hitting_set.discard(v)
                    improved = True
        
        return hitting_set
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(adj)
        # Filter to cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            out.append([])
        else:
            hs = greedy_hitting_set(cliques, n)
            out.append(sorted(hs))
    return out
# EVOLVE-BLOCK-END
