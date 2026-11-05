"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def generate_graph(n, p, seed):
        """Generate a random graph using the same method as the seed program."""
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        """Build adjacency list from edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch algorithm with pivot.
        Returns list of frozensets (each a maximal clique).
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot to minimize branching
            # Pivot is a vertex in P ∪ X with maximum degree in P
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_transversal(cliques, n):
        """
        Greedy set cover: repeatedly pick the vertex that covers the most uncovered cliques.
        """
        if not cliques:
            return []
        
        # Map each clique to its index
        uncovered = set(range(len(cliques)))
        # For each vertex, which cliques does it appear in?
        vertex_to_cliques = defaultdict(set)
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        selected = []
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v, cset in vertex_to_cliques.items():
                count = len(cset & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count <= 0:
                # Should not happen if cliques are non-empty
                break
            
            selected.append(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        return selected

    def local_search_improve(selected, cliques, n, max_iters=5000):
        """
        Try to remove vertices from the transversal if possible, and try swaps.
        """
        selected = set(selected)
        if not cliques:
            return selected
        
        # Precompute: for each clique, which selected vertices are in it
        # A clique is covered if at least one selected vertex is in it
        
        def is_valid(s):
            s = set(s)
            for clique in cliques:
                if not (clique & s):
                    return False
            return True
        
        # Try removing each vertex
        changed = True
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for v in list(selected):
                new_set = selected - {v}
                if is_valid(new_set):
                    selected = new_set
                    changed = True
                    break
        
        # Try swaps: remove one, add another
        all_vertices = set(range(n))
        changed = True
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for v in list(selected):
                for w in all_vertices - selected:
                    new_set = (selected - {v}) | {w}
                    if is_valid(new_set):
                        selected = new_set
                        changed = True
                        break
                if changed:
                    break
        
        return selected

    def solve_instance(n, p, seed):
        """Solve one instance: find minimum transversal."""
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Get all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Greedy set cover
        selected = greedy_transversal(cliques, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            selected = local_search_improve(selected, cliques, n, max_iters=10000)
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Not enough time, return what we have with empty for remaining
            while len(results) < 3:
                results.append([])
            return results
        
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results

# EVOLVE-BLOCK-END
