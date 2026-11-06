import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, adj):
        # adj is a list of bitmasks or sets
        # Using sets for simplicity, n is small
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose v in P union X with max |P intersect N(v)|
            pivot = None
            max_count = -1
            for v in P.union(X):
                count = len(P.intersection(adj[v]))
                if count > max_count:
                    max_count = count
                    pivot = v
            
            # Iterate over P \ N(pivot)
            candidates = list(P - adj[pivot])
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        P = set(range(n))
        X = set()
        bk(set(), P, X)
        return maximal_cliques

    def greedy_disjoint_cliques(cliques):
        # Returns a lower bound: max number of disjoint cliques
        # Greedy approach: pick smallest, remove intersecting
        remaining = list(cliques)
        count = 0
        while remaining:
            # Pick smallest clique
            remaining.sort(key=len)
            c = remaining.pop(0)
            count += 1
            # Remove all cliques intersecting c
            new_remaining = []
            for rc in remaining:
                if not c.intersection(rc):
                    new_remaining.append(rc)
            remaining = new_remaining
        return count

    def solve_hitting_set(cliques, time_limit):
        if not cliques:
            return []
            
        best_solution = None
        best_size = float('inf')
        
        # Pre-compute vertex frequencies for ordering
        # We will do this dynamically in the search to be accurate
        
        def search(current_set, remaining_cliques, depth):
            nonlocal best_solution, best_size
            
            if time.time() > time_limit:
                return
                
            if not remaining_cliques:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = set(current_set)
                return
                
            # Lower bound
            lb = len(current_set) + greedy_disjoint_cliques(remaining_cliques)
            if lb >= best_size:
                return
                
            # Select clique to branch on
            # Heuristic: smallest size, then highest sum of vertex degrees in remaining
            min_len = min(len(c) for c in remaining_cliques)
            candidates = [c for c in remaining_cliques if len(c) == min_len]
            
            # Tie-breaker: clique with vertices that appear in more other cliques?
            # Or just pick the first one.
            target_clique = candidates[0]
            
            # Order vertices in target_clique
            # Vertices that cover more cliques should be tried first
            # Count occurrences in remaining_cliques
            freq = {}
            for c in remaining_cliques:
                for v in c:
                    freq[v] = freq.get(v, 0) + 1
            
            sorted_vertices = sorted(target_clique, key=lambda v: -freq.get(v, 0))
            
            for v in sorted_vertices:
                new_set = current_set | {v}
                # Filter remaining cliques
                new_remaining = [c for c in remaining_cliques if v not in c]
                search(new_set, new_remaining, depth + 1)

        search(set(), cliques, 0)
        
        if best_solution is None:
            # Fallback: greedy
            # This shouldn't happen if time limit is generous, but just in case
            return greedy_hitting_set(cliques)
            
        return list(best_solution)

    def greedy_hitting_set(cliques):
        selected = set()
        remaining = list(cliques)
        while remaining:
            # Pick vertex that covers most cliques
            best_v = None
            max_cover = -1
            for c in remaining:
                for v in c:
                    count = sum(1 for rc in remaining if v in rc)
                    if count > max_cover:
                        max_cover = count
                        best_v = v
            selected.add(best_v)
            remaining = [c for c in remaining if best_v not in c]
        return list(selected)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        edges = random_graph(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find maximal cliques
        cliques = bron_kerbosch_maximal_cliques(n, adj)
        
        # Solve hitting set
        solution = solve_hitting_set(cliques, deadline)
        results.append(solution)
        
    return results
