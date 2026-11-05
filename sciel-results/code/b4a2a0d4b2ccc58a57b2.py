"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    max_m = 128

    # Precompute gcd for all pairs
    gcd_table = {}
    for m1 in range(2, max_m + 1):
        for m2 in range(m1, max_m + 1):
            gcd_table[(m1, m2)] = gcd(m1, m2)
    def get_gcd(m1, m2):
        if m1 <= m2:
            return gcd_table[(m1, m2)]
        else:
            return gcd_table[(m2, m1)]

    def is_compatible(r, m, chosen):
        """Check if residue r mod m is compatible with all chosen classes."""
        for r2, m2 in chosen:
            g = get_gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def valid_residues(m, chosen):
        """Return list of valid residues for modulus m given chosen classes."""
        valid = []
        for r in range(m):
            if is_compatible(r, m, chosen):
                valid.append(r)
        return valid

    def try_greedy(order, chosen_init=None):
        """Greedy: for each modulus in order, pick a valid residue (first one found)."""
        chosen = list(chosen_init) if chosen_init else []
        for m in order:
            valid = valid_residues(m, chosen)
            if valid:
                chosen.append((valid[0], m))
        return chosen

    def try_greedy_last(order, chosen_init=None):
        """Greedy variant: pick the last valid residue."""
        chosen = list(chosen_init) if chosen_init else []
        for m in order:
            valid = valid_residues(m, chosen)
            if valid:
                chosen.append((valid[-1], m))
        return chosen

    def local_search(chosen, max_iter=200):
        """Try to swap residues to free up space, or add removed moduli back."""
        chosen = list(chosen)
        chosen_set = set(m for _, m in chosen)
        all_moduli = set(range(2, max_m + 1))
        missing = list(all_moduli - chosen_set)
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            for m_add in missing:
                valid = valid_residues(m_add, chosen)
                if valid:
                    chosen.append((valid[0], m_add))
                    missing.remove(m_add)
                    improved = True
                    break
                else:
                    for i, (r_i, m_i) in enumerate(chosen):
                        for r_new in range(m_i):
                            if r_new == r_i:
                                continue
                            ok = True
                            for j, (r_j, m_j) in enumerate(chosen):
                                if j == i:
                                    continue
                                g = get_gcd(m_i, m_j)
                                if r_new % g == r_j % g:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            chosen[i] = (r_new, m_i)
                            if is_compatible(0, m_add, chosen):
                                valid_add = valid_residues(m_add, chosen)
                                if valid_add:
                                    chosen.append((valid_add[0], m_add))
                                    missing.remove(m_add)
                                    improved = True
                                    break
                            else:
                                chosen[i] = (r_i, m_i)
                        if improved:
                            break
                    if improved:
                        break
        return chosen

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Build compatibility graph and use maximum bipartite matching
    # Left side: moduli 2..128
    # Right side: residue assignments (r, m) where 0 <= r < m
    # Edge exists if the assignment is compatible with all other selected assignments
    
    # Alternative approach: use backtracking with memoization for small subsets
    # Or use the fact that we need pairwise disjoint classes
    
    def hopcroft_karp(adj):
        """
        Maximum bipartite matching using Hopcroft-Karp.
        adj: list of lists, adj[i] = list of neighbors of left node i
        Returns: matching dict {left_node: right_node}
        """
        n_left = len(adj)
        n_right = max(max(adj[i]) for i in range(n_left)) + 1 if any(adj) else 0
        
        pair_u = [-1] * n_left  # pair_u[u] = v matched to u
        pair_v = [-1] * n_right  # pair_v[v] = u matched to v
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        return {u: pair_u[u] for u in range(n_left) if pair_u[u] != -1}

    # Strategy: Use greedy + local search, then try to improve with matching-based approach
    candidates = list(range(2, max_m + 1))
    
    order_by_pf = sorted(candidates, key=lambda m: num_prime_factors(m))
    order_by_pf_desc = sorted(candidates, key=lambda m: -num_prime_factors(m))
    order_asc = sorted(candidates)
    order_desc = sorted(candidates, reverse=True)
    
    def order_by_gcd_heuristic():
        remaining = set(candidates)
        order = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]:
            if p in remaining:
                order.append(p)
                remaining.remove(p)
                break
        while remaining:
            if not order:
                m = min(remaining)
            else:
                best_m = None
                best_score = float('inf')
                for m in remaining:
                    max_g = max(get_gcd(m, om) for om in order)
                    if max_g < best_score:
                        best_score = max_g
                        best_m = m
                m = best_m
            order.append(m)
            remaining.remove(m)
        return order

    order_gcd_heur = order_by_gcd_heuristic()

    best_sol = []
    best_len = 0

    orders = [
        order_asc,
        order_desc,
        order_by_pf,
        order_by_pf_desc,
        order_gcd_heur,
    ]

    for order in orders:
        for greedy_fn in [try_greedy, try_greedy_last]:
            sol = greedy_fn(order)
            if len(sol) > best_len:
                best_len = len(sol)
                best_sol = sol

    if best_sol:
        improved_sol = local_search(best_sol, max_iter=500)
        if len(improved_sol) > best_len:
            best_len = len(improved_sol)
            best_sol = improved_sol

    while time.time() < deadline - 5.0:
        improved_sol = local_search(best_sol, max_iter=200)
        if len(improved_sol) > best_len:
            best_len = len(improved_sol)
            best_sol = improved_sol
        else:
            break

    for i in range(200):
        if time.time() > deadline - 2.0:
            break
        random.seed(i * 7919)
        order = candidates[:]
        random.shuffle(order)
        sol = try_greedy(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_sol = sol
        if len(sol) >= best_len - 2:
            improved_sol = local_search(sol, max_iter=100)
            if len(improved_sol) > best_len:
                best_len = len(improved_sol)
                best_sol = improved_sol

    if time.time() < deadline - 1.0:
        improved_sol = local_search(best_sol, max_iter=1000)
        if len(improved_sol) > best_len:
            best_len = len(improved_sol)
            best_sol = improved_sol

    return [[r, m] for r, m in best_sol]
# EVOLVE-BLOCK-END
