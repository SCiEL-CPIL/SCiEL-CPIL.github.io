"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P∪X with max degree in P
            pivot = max(P | X, key=lambda v: len(adj[v] & P)) if (P | X) else None
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques
    
    def greedy_hit(cliques, n, rng):
        """Greedy: repeatedly pick vertex hitting most uncovered cliques."""
        uncovered = set(range(len(cliques)))
        selected = set()
        # Precompute for each vertex which cliques it hits
        vertex_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None:
                break
            selected.add(best_v)
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        return selected
    
    def local_search(selected, cliques, n, rng, max_iters=5000):
        """Try to improve by removing vertices or swapping."""
        selected = set(selected)
        # Precompute coverage
        def is_valid(sel):
            for c in cliques:
                if not (c & sel):
                    return False
            return True
        
        if not is_valid(selected):
            return selected
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try removing each vertex
            for v in list(selected):
                trial = selected - {v}
                if is_valid(trial):
                    selected = trial
                    improved = True
                    break
            # Try swapping: remove v, add w
            if not improved:
                for v in list(selected):
                    trial = selected - {v}
                    # Find uncovered cliques after removing v
                    uncovered = [c for c in cliques if not (c & trial)]
                    if not uncovered:
                        selected = trial
                        improved = True
                        break
                    # Try adding one vertex to cover all uncovered
                    for w in range(n):
                        if w in selected:
                            continue
                        if all(w in c for c in uncovered):
                            trial2 = trial | {w}
                            if len(trial2) < len(selected):
                                selected = trial2
                                improved = True
                                break
                    if improved:
                        break
        return selected
    
    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return set()
        
        best = None
        best_size = n + 1
        rng = random.Random(seed * 1000 + 42)
        
        # Initial greedy
        init = greedy_hit(cliques, n, rng)
        init = local_search(init, cliques, n, rng)
        best = init
        best_size = len(init)
        
        # Multiple randomized restarts
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            return best
        
        attempts = 0
        while time.time() < deadline - 10 and attempts < 50:
            attempts += 1
            # Random perturbation of greedy
            perm = list(range(n))
            rng.shuffle(perm)
            # Greedy with random tie-breaking
            uncovered = set(range(len(cliques)))
            selected = set()
            vertex_cliques = {v: [] for v in range(n)}
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
            
            order_idx = {v: i for i, v in enumerate(perm)}
            while uncovered:
                best_v = None
                best_count = -1
                for v in range(n):
                    if v in selected:
                        continue
                    count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                    if count > best_count or (count == best_count and best_v is not None and order_idx[v] < order_idx[best_v]):
                        best_count = count
                        best_v = v
                if best_v is None:
                    break
                selected.add(best_v)
                for ci in vertex_cliques[best_v]:
                    uncovered.discard(ci)
            
            selected = local_search(selected, cliques, n, rng, max_iters=3000)
            if len(selected) < best_size:
                best_size = len(selected)
                best = selected
            if best_size == 1:
                break
        
        return best
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15:
            # Fallback: just return all vertices
            results.append(set(range(n)))
            continue
        sol = solve_instance(n, p, seed)
        results.append(sol)
    
    return [sorted(s) for s in results]
# EVOLVE-BLOCK-END
