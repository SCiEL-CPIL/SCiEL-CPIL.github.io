"""Construct a large triangle-free graph with alpha < 10."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    deadline = start + 115.0  # 5s safety margin for 120s limit

    # The seed: Andrásfai graph And(9) on 26 vertices
    # Z_26, edges when (j-i) mod 26 == 1 mod 3
    def seed_graph():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return n, edges

    # Verify a graph: no triangle, alpha < 10
    def is_valid(n, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check triangle-free
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        # Check alpha < 10 using branch and bound
        # Max independent set
        def max_independent(adj, n):
            # Greedy lower bound
            def greedy_indep():
                remaining = set(range(n))
                indep = set()
                while remaining:
                    # Pick vertex with min degree in remaining
                    v = min(remaining, key=lambda x: len(adj[x] & remaining))
                    indep.add(v)
                    remaining -= adj[v]
                    remaining.discard(v)
                return len(indep)
            lb = greedy_indep()
            
            # Branch and bound to find exact max
            # Use complement: max indep in G = max clique in complement
            # Or just branch directly
            best = [0]
            
            def branch(candidates, current_size):
                if time.time() > deadline - 5:
                    return
                if current_size + len(candidates) <= best[0]:
                    return
                if not candidates:
                    best[0] = max(best[0], current_size)
                    return
                cand_list = sorted(candidates, key=lambda x: len(adj[x] & candidates))
                # Branch: include or exclude first vertex
                v = cand_list[0]
                # Include v
                new_cands = [c for c in cand_list[1:] if c not in adj[v]]
                branch(new_cands, current_size + 1)
                # Exclude v
                branch(cand_list[1:], current_size)
            
            branch(list(range(n)), 0)
            return best[0]
        
        alpha = max_independent(adj, n)
        return alpha < 10

    # Try to find a graph with n > 26
    # Strategy: start from a known good graph and try to add vertices
    
    # Known: Andrásfai And(9) on 26 vertices
    # We can try blow-ups or modifications
    
    # Approach: Random local search
    # Start with seed, try to add vertices one at a time
    
    def try_extend(n_start, edges, max_n=35):
        adj = [set() for _ in range(n_start)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        current_n = n_start
        current_edges = list(edges)
        
        for target_n in range(current_n + 1, max_n + 1):
            if time.time() > deadline - 10:
                break
            
            # Try to add a vertex connected to some subset
            # For a new vertex to maintain triangle-free:
            # Its neighbors must form an independent set
            # For alpha < 10: the non-neighbors must have alpha <= 9 in the existing graph
            # but restricted...
            
            # Simpler: try random subsets of existing vertices as neighbors
            found = False
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                # Choose a random independent set as neighbors
                # Sample a random subset, check if independent
                size = random.randint(2, current_n // 3)
                subset = set(random.sample(range(current_n), size))
                
                # Check if subset is independent
                is_indep = True
                for i in subset:
                    for j in subset:
                        if i != j and j in adj[i]:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                
                if not is_indep:
                    continue
                
                # Check alpha of non-neighbors in existing graph
                non_neigh = [i for i in range(current_n) if i not in subset]
                # The new vertex is adjacent to subset, non-adjacent to non_neigh
                # Independent set in new graph: either includes new vertex (then can't include any from subset,
                # can include from non_neigh that are mutually non-adjacent)
                # or doesn't include new vertex (independent set in old graph from non_neigh)
                
                # This is complex; let's just try and verify
                new_adj = [set() for _ in range(current_n + 1)]
                for i in range(current_n):
                    new_adj[i] = adj[i].copy()
                for v in subset:
                    new_adj[v].add(current_n)
                    new_adj[current_n].add(v)
                
                # Quick check: is it triangle-free?
                tri_free = True
                for v in subset:
                    for w in subset:
                        if v != w and w in new_adj[v]:
                            tri_free = False
                            break
                    if not tri_free:
                        break
                
                if not tri_free:
                    continue
                
                # Check alpha with a quick bound
                # The independent set can include the new vertex plus an independent set
                # from non_neigh, OR an independent set from all existing vertices not containing new
                # For new vertex case: need indep set in G[non_neigh]
                # We need both to be < 10
                
                # Just do a quick greedy check
                def quick_alpha(adj_local, n_local):
                    # Greedy independent set
                    remaining = set(range(n_local))
                    indep = set()
                    while remaining:
                        v = min(remaining, key=lambda x: len(adj_local[x] & remaining))
                        indep.add(v)
                        remaining -= adj_local[v]
                        remaining.discard(v)
                    return len(indep)
                
                # For new graph
                if quick_alpha(new_adj, current_n + 1) >= 10:
                    continue
                
                # Accept
                adj = new_adj
                current_edges = [[i, j] for i in range(current_n + 1) for j in range(i+1, current_n + 1) if j in new_adj[i]]
                current_n += 1
                found = True
                break
            
            if not found:
                break
        
        return current_n, current_edges

    # Also try: different circulant constructions
    def try_circulant(n):
        # Find S subset of Z_n such that:
        # 1. S symmetric (s in S iff n-s in S)
        # 2. S sum-free mod n (no a,b in S with a+b in S)
        # 3. alpha(C_n(S)) < 10
        # For odd n, S must have even size (pairs)
        
        if n % 2 == 0:
            half = n // 2
        else:
            half = (n - 1) // 2
        
        # Candidate sets: use elements from 1 to half
        # For triangle-free: S must be sum-free
        # Try small symmetric sets
        
        import itertools
        
        best = None
        
        # For n up to ~30, we can try various sizes
        max_degree_pairs = n // 6  # keep degree reasonable
        
        for d in range(1, min(max_degree_pairs + 1, 6)):
            if time.time() > deadline - 15:
                return best
            for combo in itertools.combinations(range(1, half + 1), d):
                if time.time() > deadline - 15:
                    return best
                S = set(combo)
                S = S | {n - s for s in combo if n - s != s}
                
                # Check sum-free
                sum_free = True
                for a in S:
                    for b in S:
                        if (a + b) % n in S:
                            sum_free = False
                            break
                    if not sum_free:
                        break
                if not sum_free:
                    continue
                
                # Build graph and check alpha
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                
                # Check triangle-free (should be by construction)
                # Check alpha
                # Quick greedy
                def greedy_alpha(adj_l, n_l):
                    remaining = set(range(n_l))
                    indep = set()
                    while remaining:
                        v = min(remaining, key=lambda x: len(adj_l[x] & remaining))
                        indep.add(v)
                        remaining -= adj_l[v]
                        remaining.discard(v)
                    return len(indep)
                
                alpha_est = greedy_alpha(adj, n)
                if alpha_est < 10:
                    edges = [[i, j] for i in range(n) for j in range(i+1, n) if j in adj[i]]
                    return n, edges
        
        return best

    # Main strategy:
    # 1. Try circulant constructions for n = 27 to 40
    # 2. If that fails, try local search from seed
    
    best_n, best_edges = seed_graph()
    
    # Try circulants first (deterministic, fast)
    for n_try in range(30, 26, -1):  # 30 down to 27
        if time.time() > deadline - 20:
            break
        result = try_circulant(n_try)
        if result:
            n_candidate, edges_candidate = result
            # Verify
            if is_valid(n_candidate, edges_candidate):
                if n_candidate > best_n:
                    best_n, best_edges = n_candidate, edges_candidate
                break
    
    # Try local search from seed
    if time.time() < deadline - 15:
        for n_try in range(35, 26, -1):
            if time.time() > deadline - 10:
                break
            # Try random restarts
            for restart in range(20):
                if time.time() > deadline - 10:
                    break
                random.seed(n_try * 1000 + restart)
                n_cand, edges_cand = try_extend(best_n, best_edges, max_n=n_try)
                if n_cand > best_n:
                    if is_valid(n_cand, edges_cand):
                        best_n, best_edges = n_cand, edges_cand
                        break
                if time.time() > deadline - 10:
                    break
    
    # Final verification
    if best_n > 1:
        # Quick sanity check
        adj_check = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj_check[u].add(v)
            adj_check[v].add(u)
        # Check triangle-free
        for i in range(best_n):
            for j in adj_check[i]:
                if j > i:
                    for k in adj_check[j]:
                        if k > j and k in adj_check[i]:
                            # Triangle found, fall back to seed
                            best_n, best_edges = seed_graph()
                            break
                    if best_n == 26:
                        break
            if best_n == 26:
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
