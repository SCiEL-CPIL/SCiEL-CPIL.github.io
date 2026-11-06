import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 90s limit, 8s margin
    
    N = 512
    limit = N

    # Precompute compatibility sets
    comp = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        ci = comp[i]
        for j in range(i, N + 1):
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= limit:
                ci.add(j)
                if i != j:
                    comp[j].add(i)

    def is_valid_add(S, x):
        cx = comp[x]
        for y in S:
            if y not in cx:
                return False
        return True

    def complete(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 3:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    def local_search(S, max_iters=30):
        S = set(S)
        S = complete(S)
        improved = True
        iters = 0
        while improved and time.time() < deadline - 10 and iters < max_iters:
            improved = False
            iters += 1
            # 1-for-1 swaps
            for e in list(S):
                if time.time() > deadline - 10:
                    break
                S_minus = S - {e}
                for x in range(1, N + 1):
                    if time.time() > deadline - 10:
                        break
                    if x != e and x not in S and is_valid_add(S_minus, x):
                        S2 = complete(S_minus | {x})
                        if len(S2) > len(S):
                            S = S2
                            improved = True
                        break
            # 1-for-2 swaps
            for e in list(S):
                if time.time() > deadline - 10:
                    break
                S_minus = S - {e}
                added = 0
                S2 = set(S_minus)
                for x in range(1, N + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in S2 and is_valid_add(S2, x):
                        S2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(S2) > len(S):
                    S = complete(S2)
                    improved = True
        return S

    # --- Heuristic phase: get a good initial solution ---
    best = set(range(1, 24))
    best = complete(best)
    best = local_search(best)
    best_size = len(best)

    # Strategy 1: Power-of-2 construction
    for k in range(10):
        if time.time() > deadline - 25:
            break
        pow2 = 1 << k
        L = N // pow2
        odd_nums = [m for m in range(1, L + 1) if m % 2 == 1]
        M = set()
        for m in odd_nums:
            ok = True
            for y in M:
                g = gcd(m, y)
                if (m // g) * y > L:
                    ok = False
                    break
            if ok:
                M.add(m)
        S = set()
        for m in M:
            S.add(pow2 * m)
        S = complete(S)
        S = local_search(S, max_iters=15)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Strategy 2: Random restarts
    rng = random.Random(12345)
    for it in range(80):
        if time.time() > deadline - 20:
            break
        seed = {1}
        order = list(range(2, N + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline - 20:
                break
            if is_valid_add(seed, x):
                seed.add(x)
        S = local_search(seed, max_iters=10)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # --- Exact Branch-and-Bound phase ---
    # We model the problem as finding the maximum clique in the compatibility graph.
    # Use branch-and-bound with greedy coloring for upper bounds.
    
    # Build adjacency as bitmasks for speed (512 bits fits in 8 uint64 or Python int)
    # Python ints are arbitrary precision, so we can use them as bitmasks.
    adj_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in comp[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    # Greedy coloring of a set of vertices (represented as a list of node numbers)
    # Returns a list of color classes (each a set of nodes)
    def greedy_color(vertices):
        """Color the subgraph induced by vertices. Returns list of color classes."""
        remaining = list(vertices)
        colors = []
        while remaining:
            color_class = []
            new_remaining = []
            for v in remaining:
                # v can join current color class if it's not adjacent to any in color_class
                can_join = True
                for u in color_class:
                    if (adj_mask[v] >> u) & 1:
                        can_join = False
                        break
                if can_join:
                    color_class.append(v)
                else:
                    new_remaining.append(v)
            colors.append(color_class)
            remaining = new_remaining
        return colors

    # Branch and bound for maximum clique
    def max_clique_bb(nodes, current_clique, best_ref):
        """
        nodes: list of candidate vertices (all mutually adjacent to current_clique and each other checked)
        current_clique: list of vertices in current clique
        best_ref: [best_size] mutable reference
        Returns the best clique found.
        """
        if time.time() > deadline - 2:
            return None
        
        # Upper bound: len(current_clique) + len(nodes)
        if len(current_clique) + len(nodes) <= best_ref[0]:
            return None
        
        if not nodes:
            if len(current_clique) > best_ref[0]:
                best_ref[0] = len(current_clique)
                return list(current_clique)
            return None
        
        # Greedy coloring for upper bound
        color_classes = greedy_color(nodes)
        upper_bound = len(current_clique) + len(color_classes)
        if upper_bound <= best_ref[0]:
            return None
        
        # Order nodes by color (process last color class first for better pruning)
        # Process vertices in order of decreasing color
        node_order = []
        for ci in range(len(color_classes) - 1, -1, -1):
            node_order.extend(color_classes[ci])
        
        result = None
        for v in node_order:
            if time.time() > deadline - 2:
                break
            
            # Prune: v's color index gives an upper bound
            # Find v's color
            v_color = -1
            for ci, cc in enumerate(color_classes):
                if v in cc:
                    v_color = ci
                    break
            
            if len(current_clique) + v_color + 1 <= best_ref[0]:
                break
            
            # Branch: include v
            # New candidates: nodes that come after v in order AND are adjacent to v
            new_nodes = []
            for u in node_order:
                if u == v:
                    break
                if (adj_mask[v] >> u) & 1:
                    new_nodes.append(u)
            
            # Also need to check u is adjacent to all in current_clique
            # (but since we started from a valid clique and only added compatible nodes, this should hold)
            
            res = max_clique_bb(new_nodes, current_clique + [v], best_ref)
            if res is not None:
                result = res
                if len(res) > best_ref[0]:
                    best_ref[0] = len(res)
            
            # Prune: if even the best remaining can't beat
            remaining_upper = len(current_clique) + v_color
            if remaining_upper <= best_ref[0]:
                break
        
        return result

    # Run B&B on the full graph
    all_nodes = list(range(1, N + 1))
    # Pre-sort by degree (descending) for better initial ordering
    all_nodes.sort(key=lambda x: len(comp[x]), reverse=True)
    
    bb_best = [best_size]
    bb_result = max_clique_bb(all_nodes, [], bb_best)
    
    if bb_result is not None and len(bb_result) > best_size:
        best = set(bb_result)
        best_size = len(bb_result)
        # Polish with local search
        best = complete(best)
        best = local_search(best, max_iters=5)
        best_size = len(best)

    return sorted(best)
