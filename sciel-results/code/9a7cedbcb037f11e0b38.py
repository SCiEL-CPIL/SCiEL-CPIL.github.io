"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    import time
    deadline = time.time() + 130.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, adj):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            U = P | X
            if not U:
                return
            # Pick pivot with max degree in P
            pivot = max(U, key=lambda u: len(P & adj[u]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R_new, P_new, X_new)
                P = P - {v}
                X = X | {v}
        
        bk(frozenset(), set(range(n)), set())
        return cliques

    def maximal_matching_vertex_cover(edges, n):
        """2-approximation: find maximal matching, take both endpoints."""
        matched = set()
        cover = set()
        # Sort edges by degree sum for better quality
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        sorted_edges = sorted(edges, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        for u, v in sorted_edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                cover.add(u)
                cover.add(v)
        return cover

    def greedy_hitting_set(cliques, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        uncovered_set = set(range(len(cliques)))
        # Map each vertex to the set of clique indices it hits
        vertex_cliques = defaultdict(set)
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        selected = set()
        while uncovered_set:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in vertex_cliques:
                count = len(vertex_cliques[v] & uncovered_set)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            uncovered_set -= vertex_cliques[best_v]
        
        return selected

    def local_search_improve(selected, cliques, n, max_time=10.0):
        """Try to improve by removing vertices that are redundant, or swapping."""
        selected = set(selected)
        start_time = time.time()
        
        def is_valid(sel):
            for clique in cliques:
                if not sel & clique:
                    return False
            return True
        
        def count_unique_coverage(sel):
            """For each v in sel, count cliques that are hit only by v in sel."""
            unique = {v: 0 for v in sel}
            for clique in cliques:
                hits = [v for v in sel if v in clique]
                if len(hits) == 1:
                    unique[hits[0]] += 1
            return unique
        
        improved = True
        while improved:
            if time.time() - start_time > max_time:
                break
            improved = False
            
            # Phase 1: Try removing each vertex (in order of lowest unique coverage)
            unique_cov = count_unique_coverage(selected)
            for v in sorted(selected, key=lambda x: unique_cov.get(x, 0)):
                if time.time() - start_time > max_time:
                    break
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swapping: remove one, add another
            unique_cov = count_unique_coverage(selected)
            for v_out in sorted(selected, key=lambda x: unique_cov.get(x, 0)):
                if time.time() - start_time > max_time:
                    break
                for v_in in range(n):
                    if v_in in selected:
                        continue
                    new_sel = (selected - {v_out}) | {v_in}
                    if len(new_sel) < len(selected) and is_valid(new_sel):
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(n, adj)
        
        if not cliques:
            return []
        
        # Try both initializations and pick the better one after local search
        # Option 1: Maximal matching 2-approximation
        sel1 = maximal_matching_vertex_cover(edges, n)
        sel1 = local_search_improve(sel1, cliques, n)
        
        # Option 2: Greedy hitting set
        sel2 = greedy_hitting_set(cliques, n)
        sel2 = local_search_improve(sel2, cliques, n)
        
        # Pick the smaller one
        selected = sel1 if len(sel1) <= len(sel2) else sel2
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices as a safe (but large) solution
            results.append(list(range(n)))
            continue
        results.append(solve_instance(n, p, seed))
    
    return results
# EVOLVE-BLOCK-END
