import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate maximal cliques of size >= 2 using Bron-Kerbosch with pivoting."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            best_pivot = None
            max_count = -1
            for v in P:
                count = len(P & adj[v])
                if count > max_count:
                    max_count = count
                    best_pivot = v
            for v in X:
                count = len(P & adj[v])
                if count > max_count:
                    max_count = count
                    best_pivot = v
            
            if best_pivot is None:
                candidates = list(P)
            else:
                candidates = [v for v in P if v not in adj[best_pivot]]
            
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hit_set(cliques, n, vert_cliques, rng=None):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        m = len(cliques)
        if m == 0:
            return set()
        
        uncovered = set(range(m))
        result = set()
        
        while uncovered:
            best_v = -1
            best_cov = 0
            for v in range(n):
                if v in result:
                    continue
                cov = len(uncovered & vert_cliques[v])
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            if best_v == -1 or best_cov == 0:
                break
            result.add(best_v)
            uncovered -= vert_cliques[best_v]
        
        return result

    def local_search(sol, cliques, n, vert_cliques, max_iters=200):
        """Local search: try removals and swaps to reduce solution size."""
        m = len(cliques)
        result = set(sol)
        if not result:
            return result
        
        # Build coverage counts
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE:
            improved = False
            iters += 1
            
            # Try removals
            for v in list(result):
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: replace v with u not in result
            for v in list(result):
                # Find cliques uniquely covered by v
                unique = set()
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        unique.add(ci)
                
                if not unique:
                    continue
                
                # Find candidate u that covers all unique cliques
                best_u = -1
                best_gain = 0
                for u in range(n):
                    if u in result:
                        continue
                    # Check if u covers all unique cliques
                    covers_all = all(ci in vert_cliques[u] for ci in unique)
                    if not covers_all:
                        continue
                    # Gain: how many additional cliques u covers
                    gain = sum(1 for ci in vert_cliques[u] if cover_count[ci] >= 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                
                if best_u != -1:
                    # Perform swap
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    for ci in vert_cliques[best_u]:
                        cover_count[ci] += 1
                    result.discard(v)
                    result.add(best_u)
                    improved = True
                    break
        
        return result

    def branch_and_bound(cliques, n, vert_cliques, initial_ub, time_limit):
        """
        Exact branch-and-bound for minimum hitting set.
        Branches on the smallest uncovered clique.
        Uses disjoint-clique lower bound for pruning.
        """
        m = len(cliques)
        if m == 0:
            return set()
        
        clique_verts = [list(c) for c in cliques]
        clique_size = [len(c) for c in cliques]
        
        best_sol = set()
        best_size = initial_ub
        
        # Disjoint clique lower bound
        def disjoint_lb(uncovered):
            """Greedy maximal set of pairwise disjoint cliques from uncovered."""
            used = set()
            count = 0
            # Sort by size ascending for tighter bound
            for ci in sorted(uncovered, key=lambda i: clique_size[i]):
                cverts = clique_verts[ci]
                if not any(v in used for v in cverts):
                    used.update(cverts)
                    count += 1
            return count
        
        def bb(selected, uncovered):
            nonlocal best_sol, best_size
            
            if time.time() > time_limit:
                return
            
            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = set(selected)
                return
            
            # Lower bound pruning
            lb = disjoint_lb(uncovered)
            if len(selected) + lb >= best_size:
                return
            
            # Pick smallest uncovered clique
            min_ci = min(uncovered, key=lambda i: clique_size[i])
            cverts = clique_verts[min_ci]
            
            # Order vertices by coverage of uncovered cliques
            cverts_sorted = sorted(cverts, key=lambda v: -len(uncovered & vert_cliques[v]))
            
            for v in cverts_sorted:
                if time.time() > time_limit:
                    return
                
                new_selected = selected | {v}
                new_uncovered = uncovered - vert_cliques[v]
                
                bb(new_selected, new_uncovered)
        
        bb(set(), set(range(m)))
        
        return best_sol

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            results.append(list(range(n)))
            continue
        
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Build vertex -> clique membership
        m = len(cliques)
        vert_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cliques[v].add(ci)
        
        # Get initial upper bound via greedy
        initial_sol = greedy_hit_set(cliques, n, vert_cliques)
        initial_ub = len(initial_sol)
        
        # Try B&B with time limit
        remaining_time = DEADLINE - time.time()
        bb_time_limit = min(remaining_time - 5.0, 60.0)  # Leave 5s for other instances + local search
        
        if bb_time_limit > 1.0:
            bb_sol = branch_and_bound(cliques, n, vert_cliques, initial_ub, time.time() + bb_time_limit)
            if len(bb_sol) < initial_ub:
                initial_sol = bb_sol
                initial_ub = len(bb_sol)
        
        # Local search polish
        if time.time() < DEADLINE - 2.0:
            polished = local_search(initial_sol, cliques, n, vert_cliques, max_iters=300)
            if len(polished) < len(initial_sol):
                initial_sol = polished
        
        results.append(sorted(initial_sol))

    return results
# EVOLVE-BLOCK-END
