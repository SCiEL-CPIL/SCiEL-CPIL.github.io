"""Combined program: structural candidates + hypergraph-aware greedy + local search swaps + Exact MIS."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def construct():
    from random import Random

    t0 = time.time()

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_conflicts(elements):
        """Build conflict structures for fast sum-free checking."""
        n = len(elements)
        idx = {v: i for i, v in enumerate(elements)}
        conf_c = defaultdict(list)   # c -> list of (a_idx, b_idx)
        conf_a = defaultdict(list)   # a_idx -> list of (b_idx, c_idx)
        # Adjacency bitsets for MIS
        adj = [0] * n
        for i in range(n):
            a = elements[i]
            for j in range(i, n):
                b = elements[j]
                c = a + b
                if c in idx:
                    k = idx[c]
                    conf_c[k].append((i, j))
                    conf_a[i].append((j, k))
                    if j != i:
                        conf_a[j].append((i, k))
                    # Add edges to conflict graph
                    # i connected to j, i connected to k, j connected to k
                    adj[i] |= (1 << j)
                    adj[i] |= (1 << k)
                    adj[j] |= (1 << i)
                    adj[j] |= (1 << k)
                    adj[k] |= (1 << i)
                    adj[k] |= (1 << j)
        return idx, conf_c, conf_a, adj

    def can_add(conf_c, conf_a, chosen_set, candidate):
        """Check if candidate can be added to chosen_set without creating a triple."""
        for (a, b) in conf_c[candidate]:
            if a in chosen_set and b in chosen_set:
                return False
        for (b, c) in conf_a[candidate]:
            if b in chosen_set and c in chosen_set:
                return False
        return True

    def greedy_from_order(elements, order, conf_c, conf_a):
        """Greedy: add elements in given order if safe."""
        chosen = set()
        for i in order:
            if can_add(conf_c, conf_a, chosen, i):
                chosen.add(i)
        return chosen

    def local_search_1_for_2(elements, chosen, conf_c, conf_a, t0, deadline):
        """Local search: try removing one element and adding two others."""
        n = len(elements)
        improved = True
        while improved:
            improved = False
            for r in list(chosen):
                if time.time() - t0 > deadline:
                    return chosen
                new_chosen = chosen - {r}
                # Find candidates that can be added to new_chosen
                candidates = []
                for i in range(n):
                    if i not in new_chosen and can_add(conf_c, conf_a, new_chosen, i):
                        candidates.append(i)
                # Try pairs from candidates
                for pi in range(len(candidates)):
                    p = candidates[pi]
                    temp = new_chosen | {p}
                    for qj in range(pi + 1, len(candidates)):
                        q = candidates[qj]
                        if can_add(conf_c, conf_a, temp, q):
                            chosen = temp | {q}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return chosen

    def exact_mis(adj, n, time_limit, initial_bound):
        """
        Branch and Bound for Maximum Independent Set.
        adj: list of integers (bitmasks) representing adjacency.
        n: number of nodes.
        time_limit: max seconds to run (deadline).
        initial_bound: lower bound on size of MIS.
        Returns best set found (as a list of indices).
        """
        best_size = initial_bound
        best_set_mask = 0
        
        # To speed up, we can use a greedy coloring bound on the remaining graph.
        # This is the standard Tomita algorithm.
        
        def greedy_coloring(candidates_mask):
            """
            Returns a list of colors for nodes in candidates_mask.
            Also returns the maximum color used, which is an upper bound on the size of the MIS in the subgraph.
            """
            # Extract nodes
            nodes = []
            mask = candidates_mask
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                nodes.append(idx)
                mask ^= lsb
            
            # Simple greedy coloring
            color = [0] * n
            color_count = 0
            for u in nodes:
                # Find smallest color not used by neighbors in nodes
                used_colors = set()
                adj_u = adj[u]
                for v in nodes:
                    if v != u and (adj_u >> v) & 1:
                        used_colors.add(color[v])
                
                c = 0
                while c in used_colors:
                    c += 1
                color[u] = c
                if c + 1 > color_count:
                    color_count = c + 1
            return color, color_count

        def dfs(candidates_mask, current_set_mask, current_size):
            nonlocal best_size, best_set_mask
            
            if time.time() > time_limit:
                return
            
            if not candidates_mask:
                if current_size > best_size:
                    best_size = current_size
                    best_set_mask = current_set_mask
                return
            
            # Upper bound: current_size + max color in greedy coloring of candidates
            color, max_color = greedy_coloring(candidates_mask)
            if current_size + max_color <= best_size:
                return
            
            # Branch on a node
            # Let's pick the first available node in the mask.
            lsb = candidates_mask & -candidates_mask
            u = lsb.bit_length() - 1
            
            # Branch 1: Include u
            # New candidates: candidates without u and without neighbors of u
            new_candidates = candidates_mask & ~(adj[u] | (1 << u))
            new_set = current_set_mask | (1 << u)
            dfs(new_candidates, new_set, current_size + 1)
            
            # Branch 2: Exclude u
            new_candidates = candidates_mask & ~(1 << u)
            dfs(new_candidates, current_set_mask, current_size)

        # Initial call
        dfs((1 << n) - 1, 0, 0)
        
        # Convert best_set_mask to list of indices
        best_indices = []
        mask = best_set_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            best_indices.append(idx)
            mask ^= lsb
        return best_indices, best_size

    def solve(elements):
        n = len(elements)
        if n == 0:
            return []
        idx, conf_c, conf_a, adj = build_conflicts(elements)
        
        # Generate structural candidates
        candidates = []
        
        # Odd numbers
        odd_set = set(i for i in range(n) if elements[i] % 2 != 0)
        candidates.append(odd_set)
        
        # Upper half (x > M/2)
        M = elements[-1]
        m = elements[0]
        upper = set(i for i in range(n) if elements[i] > M / 2.0)
        candidates.append(upper)
        
        # Lower half (x < m/2)
        lower = set(i for i in range(n) if elements[i] < m / 2.0)
        candidates.append(lower)
        
        # Positive numbers
        pos = set(i for i in range(n) if elements[i] > 0)
        candidates.append(pos)
        
        # Negative numbers
        neg = set(i for i in range(n) if elements[i] < 0)
        candidates.append(neg)
        
        # Large positive (x > M/3)
        large_pos = set(i for i in range(n) if elements[i] > 0 and elements[i] > M / 3.0)
        candidates.append(large_pos)
        
        # Large negative (x < m/3)
        large_neg = set(i for i in range(n) if elements[i] < 0 and elements[i] < m / 3.0)
        candidates.append(large_neg)
        
        # Absolute value thresholds
        abs_vals = sorted(set(abs(x) for x in elements))
        for K in abs_vals[:10]:
            S_K = set(i for i in range(n) if abs(elements[i]) > K)
            candidates.append(S_K)
        
        # Greedy from multiple orderings
        degree = [len(conf_c[i]) + len(conf_a[i]) for i in range(n)]
        
        # Low degree first
        order_low_deg = sorted(range(n), key=lambda i: degree[i])
        candidates.append(greedy_from_order(elements, order_low_deg, conf_c, conf_a))
        
        # High abs first
        order_abs_desc = sorted(range(n), key=lambda i: abs(elements[i]), reverse=True)
        candidates.append(greedy_from_order(elements, order_abs_desc, conf_c, conf_a))
        
        # Sorted ascending
        order_asc = list(range(n))
        candidates.append(greedy_from_order(elements, order_asc, conf_c, conf_a))
        
        # Sorted descending
        order_desc = list(range(n - 1, -1, -1))
        candidates.append(greedy_from_order(elements, order_desc, conf_c, conf_a))
        
        # Random shuffle
        rng = Random(42)
        order_rand = list(range(n))
        rng.shuffle(order_rand)
        candidates.append(greedy_from_order(elements, order_rand, conf_c, conf_a))
        
        # Validate and pick best candidate
        best_chosen = set()
        best_len = 0
        for cand in candidates:
            if len(cand) <= best_len:
                continue
            # Verify sum-free
            valid = True
            for i in cand:
                for (a, b) in conf_c[i]:
                    if a in cand and b in cand:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_chosen = cand
                best_len = len(cand)
        
        # If no valid candidate found (shouldn't happen), use odd numbers as fallback
        if not best_chosen:
            best_chosen = set(i for i in range(n) if elements[i] % 2 != 0)
        
        # Local search improvement
        if time.time() - t0 < _DEADLINE - 20:
            best_chosen = local_search_1_for_2(elements, best_chosen, conf_c, conf_a, t0, _DEADLINE - 10)
            best_len = len(best_chosen)
            
        # Exact MIS attempt
        # Only try if we have enough time and the instance is small enough
        # N=240 might be too slow for pure Python B&B without heavy optimization.
        # Let's try it with a strict time limit.
        if time.time() - t0 < _DEADLINE - 5:
            # Allocate remaining time minus safety margin
            remaining_time = _DEADLINE - 5 - time.time()
            # Cap time for MIS to ensure we return something
            mis_time_limit = min(remaining_time, 10.0) 
            
            try:
                mis_indices, mis_size = exact_mis(adj, n, time.time() + mis_time_limit, best_len)
                if mis_size > best_len:
                    best_chosen = set(mis_indices)
                    best_len = mis_size
            except Exception:
                pass # Fallback to heuristic result

        return [elements[i] for i in best_chosen]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - t0 > _DEADLINE - 5:
            # Fallback: return odd numbers for remaining
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        results.append(solve(vals))

    return results
# EVOLVE-BLOCK-END
