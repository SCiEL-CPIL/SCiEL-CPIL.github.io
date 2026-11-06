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
        m = len(cliques)
        
        # Precompute: for each vertex, which cliques it hits (as a set for fast ops)
        vertex_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        # Greedy upper bound
        remaining_cliques = set(range(m))
        greedy_cover = []
        temp_remaining = set(remaining_cliques)
        
        while temp_remaining:
            # Find vertex that hits most remaining cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = len(vertex_cliques[v] & temp_remaining)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
                
            greedy_cover.append(best_v)
            temp_remaining -= vertex_cliques[best_v]
        
        best_cover = list(greedy_cover)
        best_size = len(greedy_cover)
        
        if best_size == 0:
            return []
        
        # Local search: 1-swap improvement
        def try_local_search(cover):
            """Try to improve cover by removing vertices that are redundant."""
            cover_set = set(cover)
            changed = True
            while changed:
                changed = False
                for v in list(cover_set):
                    # Check if removing v still covers all cliques
                    new_cover = cover_set - {v}
                    # Check all cliques are still hit
                    all_hit = True
                    for clique in cliques:
                        if not (clique & new_cover):
                            all_hit = False
                            break
                    if all_hit:
                        cover_set = new_cover
                        changed = True
            return list(cover_set)
        
        best_cover = try_local_search(best_cover)
        best_size = len(best_cover)
        
        if best_size == 0:
            return []
        
        # Branch and bound
        # Precompute clique sizes for branching heuristic
        clique_sizes = [len(c) for c in cliques]
        
        def lower_bound(remaining_clique_set):
            """Greedy set cover lower bound."""
            if not remaining_clique_set:
                return 0
            
            remaining = set(remaining_clique_set)
            lb = 0
            while remaining:
                # Find vertex covering most remaining cliques
                max_covered = 0
                best_v = -1
                for v in range(n):
                    covered = len(vertex_cliques[v] & remaining)
                    if covered > max_covered:
                        max_covered = covered
                        best_v = v
                
                if max_covered == 0:
                    return lb + len(remaining)
                
                lb += 1
                remaining -= vertex_cliques[best_v]
            
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
            
            # Choose a clique to branch on: pick smallest remaining clique
            min_clique_idx = min(remaining_clique_set, key=lambda i: clique_sizes[i])
            branch_clique = cliques[min_clique_idx]
            
            # Branch: for each vertex in branch_clique, try adding it
            # Order vertices by how many remaining cliques they cover (most first)
            vertex_scores = []
            for v in branch_clique:
                score = len(vertex_cliques[v] & remaining_clique_set)
                vertex_scores.append((score, v))
            vertex_scores.sort(reverse=True)
            
            for score, v in vertex_scores:
                new_remaining = remaining_clique_set - vertex_cliques[v]
                bbs(new_remaining, current_size + 1, current_cover + [v])
                
                if time.time() > deadline:
                    return
        
        # Start B&B
        initial_remaining = set(range(m))
        bbs(initial_remaining, 0, [])
        
        # Final local search polish
        best_cover = try_local_search(best_cover)
        
        return best_cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = instance(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cover = min_hitting_set(cliques, n, deadline)
        out.append(sorted(cover))
        
    return out
