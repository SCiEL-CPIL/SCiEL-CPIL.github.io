import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe, limit is 150s

    def instance(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex from P ∪ X with max |N(u) ∩ P|
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda u: len(adj[u] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P = P - {v}
                X = X | {v}
        
        bk(frozenset(), set(range(n)), frozenset())
        return cliques

    def min_hitting_set(cliques, n, time_limit):
        """Find minimum set of vertices that hits every clique in cliques."""
        if not cliques:
            return []
        
        # Convert cliques to frozensets for fast intersection
        cliques = [frozenset(c) for c in cliques]
        
        # Precompute: for each vertex, which cliques it hits
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(i)
        
        # Greedy upper bound
        remaining_cliques = set(range(len(cliques)))
        greedy_cover = []
        temp_remaining = set(remaining_cliques)
        
        while temp_remaining:
            # Find vertex that hits most remaining cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = 0
                for ci in vertex_cliques[v]:
                    if ci in temp_remaining:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
                
            greedy_cover.append(best_v)
            # Remove all cliques hit by best_v
            for ci in vertex_cliques[best_v]:
                temp_remaining.discard(ci)
        
        best_cover = greedy_cover
        best_size = len(greedy_cover)
        
        if best_size == 0:
            return []
        
        # Branch and bound with lower bound from greedy set cover relaxation
        def lower_bound(remaining_clique_indices):
            """Greedy set cover lower bound: repeatedly pick vertex covering most uncovered cliques."""
            if not remaining_clique_indices:
                return 0
            
            remaining = set(remaining_clique_indices)
            lb = 0
            while remaining:
                # Find vertex covering most remaining cliques
                max_covered = 0
                for v in range(n):
                    covered = 0
                    for ci in vertex_cliques[v]:
                        if ci in remaining:
                            covered += 1
                    if covered > max_covered:
                        max_covered = covered
                
                if max_covered == 0:
                    return lb + len(remaining)  # Should not happen if problem is valid
                
                lb += 1
                # Remove all cliques covered by best vertex
                # We need to find which vertex achieves max_covered
                for v in range(n):
                    covered_count = 0
                    for ci in vertex_cliques[v]:
                        if ci in remaining:
                            covered_count += 1
                    if covered_count == max_covered:
                        for ci in vertex_cliques[v]:
                            if ci in remaining:
                                remaining.discard(ci)
                        break
            
            return lb
        
        def bbs(remaining_clique_set, current_size, current_cover):
            nonlocal best_size, best_cover, deadline
            
            if time.time() > deadline:
                return
            
            if not remaining_clique_set:
                if current_size < best_size:
                    best_size = current_size
                    best_cover = list(current_cover)
                return
            
            # Lower bound
            lb = current_size + lower_bound(remaining_clique_set)
            if lb >= best_size:
                return
            
            # Choose a clique to branch on
            # Pick the smallest remaining clique (hardest to satisfy)
            min_clique_idx = min(remaining_clique_set, key=lambda i: len(cliques[i]))
            branch_clique = cliques[min_clique_idx]
            
            # Branch: for each vertex in branch_clique, try adding it
            # Order vertices by how many remaining cliques they cover (most first)
            vertex_scores = []
            for v in branch_clique:
                score = 0
                for ci in vertex_cliques[v]:
                    if ci in remaining_clique_set:
                        score += 1
                vertex_scores.append((score, v))
            vertex_scores.sort(reverse=True)
            
            for score, v in vertex_scores:
                new_remaining = set(remaining_clique_set)
                for ci in vertex_cliques[v]:
                    new_remaining.discard(ci)
                bbs(new_remaining, current_size + 1, current_cover + [v])
                
                if time.time() > deadline:
                    return
        
        # Start B&B
        initial_remaining = set(range(len(cliques)))
        bbs(initial_remaining, 0, [])
        
        return best_cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = instance(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cover = min_hitting_set(cliques, n, deadline)
        out.append(sorted(cover))
        
    return out
