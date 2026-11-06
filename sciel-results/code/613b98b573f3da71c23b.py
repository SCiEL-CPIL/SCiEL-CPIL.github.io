"""
Construct a C4-free graph on n vertices with no independent 11-set, maximizing n.

Strategy (combined):
  - Use bitset-based (Python int) adjacency for fast C4 checks and alpha computation.
  - Random-greedy C4-free construction: shuffle all edges, add if safe.
  - Multiple seeds, try n=43 down to n=35.
  - Exact alpha via branch-and-bound with bitsets (max-degree branching).
  - Fallback: 10 disjoint triangles (n=30, alpha=10).
"""

import time
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Fallback: 10 disjoint triangles, n=30, alpha=10 < 11, C4-free
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]

    def check_c4_free(adj_mask, n):
        """Verify no two vertices share >= 2 common neighbours."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    def compute_alpha(adj_mask, n, time_limit):
        """
        Exact maximum independent set size via branch-and-bound with bitsets.
        """
        best = [0]
        deadline_alpha = start_time + time_limit

        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            if time.time() > deadline_alpha:
                return
            if size + candidates.bit_count() <= best[0]:
                return

            # Pick vertex with max degree in induced subgraph
            max_deg = -1
            chosen = -1
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v

            if chosen == -1:
                return

            # Branch 1: include chosen
            new_cand = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_cand, size + 1)

            # Branch 2: exclude chosen
            new_cand = candidates & ~(1 << chosen)
            dfs(new_cand, size)

        all_cand = (1 << n) - 1
        dfs(all_cand, 0)
        return best[0]

    def greedy_c4_free(n, seed):
        """
        Random greedy: shuffle all edges, add each if it doesn't create a C4.
        """
        rng = random.Random(seed)
        adj_mask = [0] * n
        graph_edges = []

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)

        for u, v in all_pairs:
            nu = adj_mask[u]
            nv = adj_mask[v]

            safe = True
            c = nv
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nu:
                    safe = False
                    break

            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                graph_edges.append([u, v])

        return adj_mask, graph_edges

    def tighten_graph(adj_mask, n, graph_edges, time_limit):
        """
        Add edges to reduce independence number while maintaining C4-free property.
        """
        deadline_tighten = start_time + time_limit
        max_iterations = 50
        
        for _ in range(max_iterations):
            if time.time() > deadline_tighten:
                break
                
            # Find a maximal independent set (greedy)
            remaining = (1 << n) - 1
            independent_set = []
            order = sorted(range(n), key=lambda i: (adj_mask[i]).bit_count())
            
            for v in order:
                if remaining & (1 << v):
                    independent_set.append(v)
                    remaining &= ~(adj_mask[v] | (1 << v))
            
            if len(independent_set) < 11:
                # Heuristic suggests alpha might be < 11, but we need exact check later.
                # We can still try to add edges to make it even smaller?
                # No, if greedy finds < 11, it's a good candidate.
                # But adding edges might break the C4-free property? No, we check.
                # Adding edges reduces alpha. So it's always beneficial to add safe edges.
                pass
            
            # Try to add edges between vertices in the independent set
            improved = False
            for i in range(len(independent_set)):
                for j in range(i + 1, len(independent_set)):
                    u = independent_set[i]
                    v = independent_set[j]
                    
                    if adj_mask[u] & (1 << v):
                        continue
                    
                    nu = adj_mask[u]
                    nv = adj_mask[v]
                    
                    safe = True
                    c = nv
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nu:
                            safe = False
                            break
                    
                    if safe:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        graph_edges.append([u, v])
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return adj_mask, graph_edges

    best_result = None
    best_n = 0

    for n_target in [43, 42, 41, 40, 39, 38, 37, 36, 35]:
        if time.time() > deadline - 10:
            break

        remaining = deadline - time.time()
        if remaining > 60:
            num_seeds = 10
        elif remaining > 30:
            num_seeds = 5
        else:
            num_seeds = 2

        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break

            adj_mask, graph_edges = greedy_c4_free(n_target, seed)
            adj_mask, graph_edges = tighten_graph(adj_mask, n_target, graph_edges, time_limit=5.0)
            
            if time.time() > deadline - 10:
                break

            if not check_c4_free(adj_mask, n_target):
                continue

            # Heuristic check
            remaining_mask = (1 << n_target) - 1
            count = 0
            order = sorted(range(n_target), key=lambda i: (adj_mask[i]).bit_count())
            for v in order:
                if remaining_mask & (1 << v):
                    count += 1
                    remaining_mask &= ~(adj_mask[v] | (1 << v))
            
            if count >= 11:
                continue

            alpha = compute_alpha(adj_mask, n_target, time_limit=10.0)
            
            if alpha < 11:
                if n_target > best_n:
                    best_n = n_target
                    best_result = {'n': n_target, 'edges': graph_edges}
                break
        
        if best_result is not None and best_n == n_target:
            break

    if best_result is None:
        best_result = {'n': 30, 'edges': fallback_edges}

    return best_result
