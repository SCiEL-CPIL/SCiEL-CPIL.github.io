import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        def components(self):
            comp_map = {}
            for i in range(len(self.parent)):
                root = self.find(i)
                if root not in comp_map:
                    comp_map[root] = []
                comp_map[root].append(i)
            return list(comp_map.values())

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score(values, edge_list):
        """Fast score computation using local variables."""
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))

    def fast_score_with_counts(values, edge_list):
        """Returns (score, sum_count, prod_count) for better analysis."""
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        sc = max(len(s), len(p))
        return sc, len(s), len(p)
    
    def smooth_max(sums_count, prods_count, beta=0.5):
        """Soft-max surrogate: smooth approximation of max(a, b) via log-sum-exp."""
        if sums_count == prods_count:
            return sums_count
        m = max(sums_count, prods_count)
        return m + math.log(math.exp(-beta * (m - sums_count)) + math.exp(-beta * (m - prods_count))) / beta

    def best_edges_for_values(values, budget):
        """Greedy + local search: pick 35 edges to minimize score, for fixed values."""
        # Greedy init: pick edges that reuse existing sum/product
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        # Sort all edges by a heuristic: prefer pairs with small values
        edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        for e in edge_order:
            if len(chosen) == m:
                break
            chosen.append(e)
            chosen_set.add(e)
            sums.add(values[e[0]] + values[e[1]])
            prods.add(values[e[0]] * values[e[1]])
        # Local search to improve with fast scoring
        cur = list(chosen)
        cur_score = fast_score(values, cur)
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            new_cur = list(cur)
            new_cur[i] = e_in
            new_score = fast_score(values, new_cur)
            if new_score <= cur_score:
                cur = new_cur
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000 and random.random() < 0.01:
                    no_improve = 0
        return cur, cur_score
    
    def csp_edge_search(values, target_k, budget):
        """Constraint satisfaction: find 35 edges with at most target_k distinct sums AND target_k distinct products.
        Uses Union-Find to track equivalence classes of edges sharing sums/products."""
        # Precompute sum and product for all edges
        edge_sum = {}
        edge_prod = {}
        for e in all_edges:
            u, v = e
            edge_sum[e] = values[u] + values[v]
            edge_prod[e] = values[u] * values[v]
        
        # Group edges by sum and by product
        sum_groups = {}
        prod_groups = {}
        for e in all_edges:
            s = edge_sum[e]
            p = edge_prod[e]
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append(e)
            if p not in prod_groups:
                prod_groups[p] = []
            prod_groups[p].append(e)
        
        # Sort groups by size (larger groups give more flexibility)
        sum_group_keys = sorted(sum_groups.keys(), key=lambda k: -len(sum_groups[k]))
        prod_group_keys = sorted(prod_groups.keys(), key=lambda k: -len(prod_groups[k]))
        
        best_edges = None
        best_score = float('inf')
        
        # Try to construct using first target_k sum groups and first target_k prod groups
        for trial in range(min(budget, 20)):
            if time.time() > deadline - 3.0:
                break
            
            # Randomly sample from top groups
            candidate_sum_keys = random.sample(sum_group_keys, min(target_k, len(sum_group_keys)))
            candidate_prod_keys = random.sample(prod_group_keys, min(target_k, len(prod_group_keys)))
            
            allowed_edges = set()
            for sk in candidate_sum_keys:
                allowed_edges.update(sum_groups[sk])
            # Intersect with product constraint
            allowed_prod_edges = set()
            for pk in candidate_prod_keys:
                allowed_prod_edges.update(prod_groups[pk])
            
            # Edges must satisfy BOTH: sum in allowed sums AND product in allowed prods
            valid_edges = [e for e in all_edges if edge_sum[e] in set(candidate_sum_keys) and edge_prod[e] in set(candidate_prod_keys)]
            
            if len(valid_edges) < m:
                continue
            
            # Pick m edges from valid_edges, preferring diversity
            random.shuffle(valid_edges)
            chosen = valid_edges[:m]
            
            # Verify and score
            sc, sc_count, pc_count = fast_score_with_counts(values, chosen)
            if sc <= target_k and (best_edges is None or sc < best_score):
                best_edges = list(chosen)
                best_score = sc
                if best_score < target_k:
                    break
        
        if best_edges is None:
            # Fallback: just pick m edges that minimize max distinct sums/prods
            chosen = []
            chosen_set = set()
            sums = set()
            prods = set()
            edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
            for e in edge_order:
                if len(chosen) == m:
                    break
                chosen.append(e)
                chosen_set.add(e)
                sums.add(values[e[0]] + values[e[1]])
                prods.add(values[e[0]] * values[e[1]])
            best_edges = chosen
            best_score = fast_score(values, chosen)
        
        return best_edges, best_score

    def greedy_edges_max_collision(values):
        """Build edges greedily to maximize sum/product collisions."""
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        
        # Pre-compute all edge sums and products
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Sort by how many collisions they would create
        def collision_score(e):
            _, _, s, p = e
            score = 0
            if s in sums:
                score += 10
            if p in prods:
                score += 10
            # Bonus for small values
            score -= (e[2] + e[3]) / 10000.0
            return score
        
        remaining = list(range(len(all_edges)))
        for _ in range(m):
            if not remaining:
                break
            # Find best edge
            best_idx = -1
            best_score = float('-inf')
            for idx in remaining:
                e = edge_data[idx]
                s = collision_score(e)
                if s > best_score:
                    best_score = s
                    best_idx = idx
            if best_idx == -1:
                break
            u, v, s, p = edge_data[best_idx]
            chosen.append((u, v))
            chosen_set.add((u, v))
            sums.add(s)
            prods.add(p)
            remaining.remove(best_idx)
        
        return chosen

    def best_values_for_edges(edges, budget):
        """Local search on values for fixed edges."""
        # Start with small distinct values
        vals = random.sample(range(1, 41), n)
        elist = list(edges)
        cur_score = fast_score(vals, elist)
        best = list(vals)
        best_score = cur_score
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            if random.random() < 0.5:
                # replace
                new_val = random.randint(1, 100)
            else:
                new_val = vals[idx] + random.randint(-15, 15)
            if not (1 <= new_val <= 10000) or new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = fast_score(new_vals, elist)
            if s < cur_score:
                vals = new_vals
                cur_score = s
                no_improve = 0
                if s < best_score:
                    best = list(vals)
                    best_score = s
            elif s == cur_score and random.random() < 0.1:
                vals = new_vals
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 3000:
                    no_improve = 0
        return best, best_score

    def targeted_value_moves(vals, elist, budget):
        """Targeted moves to create collisions in sums/products."""
        cur = list(vals)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(cur, elist)
        best = list(cur)
        best_score = cur_score
        no_improve = 0
        it = 0
        
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            
            # Find neighbors
            neighbors = set()
            for u, v in elist:
                if u == idx:
                    neighbors.add(v)
                elif v == idx:
                    neighbors.add(u)
            
            if not neighbors:
                continue
            
            # Get current sums and products involving idx
            existing_sums = set()
            existing_prods = set()
            for u, v in elist:
                if u == idx or v == idx:
                    other = v if u == idx else u
                    existing_sums.add(cur[idx] + cur[other])
                    existing_prods.add(cur[idx] * cur[other])
            
            # Generate candidates that create collisions
            candidates = set()
            
            # For each neighbor, try to match sum with existing sum
            for nb in neighbors:
                for s_val in existing_sums:
                    cand = s_val - cur[nb]
                    if 1 <= cand <= 10000 and cand != cur[idx]:
                        candidates.add(cand)
            
            # For each neighbor, try to match product with existing product
            for nb in neighbors:
                for p_val in existing_prods:
                    if cur[nb] != 0 and p_val % cur[nb] == 0:
                        cand = p_val // cur[nb]
                        if 1 <= cand <= 10000 and cand != cur[idx]:
                            candidates.add(cand)
            
            # Also try to create new collisions by matching with other edges' sums/products
            for u, v in elist:
                if u == idx or v == idx:
                    continue
                other_sum = cur[u] + cur[v]
                other_prod = cur[u] * cur[v]
                # Try to make idx + some_neighbor = other_sum
                for nb in neighbors:
                    cand = other_sum - cur[nb]
                    if 1 <= cand <= 10000 and cand != cur[idx]:
                        candidates.add(cand)
                # Try to make idx * some_neighbor = other_prod
                for nb in neighbors:
                    if cur[nb] != 0 and other_prod % cur[nb] == 0:
                        cand = other_prod // cur[nb]
                        if 1 <= cand <= 10000 and cand != cur[idx]:
                            candidates.add(cand)
            
            if not candidates:
                candidates.add(random.randint(1, 50))
            
            new_val = random.choice(list(candidates))
            if new_val in cur:
                continue
            
            new_vals = list(cur)
            new_vals[idx] = new_val
            s, sc, pc = fast_score_with_counts(new_vals, elist)
            
            # Accept if it improves or ties with better balance
            if s < cur_score or (s == cur_score and (sc < cur_sc or pc < cur_pc)):
                cur = new_vals
                cur_score, cur_sc, cur_pc = s, sc, pc
                no_improve = 0
                if s < best_score:
                    best = list(cur)
                    best_score = s
            elif s == cur_score and random.random() < 0.1:
                cur = new_vals
                cur_score, cur_sc, cur_pc = s, sc, pc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000:
                    no_improve = 0
        
        return best, best_score

    def joint_search(vals0, edges0, budget):
        """Alternate value and edge moves with targeted collision-seeking and smooth surrogate."""
        vals = list(vals0)
        edges = set(edges0)
        elist = list(edges)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(edges)
        it = 0
        beta = 0.3  # smoothness parameter for soft-max
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            r = random.random()
            if r < 0.35:
                # edge move
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                new_elist = list(elist)
                new_elist[i] = e_in
                s, sc, pc = fast_score_with_counts(vals, new_elist)
                # Use smooth surrogate for acceptance: accept if smooth max improves
                cur_smooth = smooth_max(cur_sc, cur_pc, beta)
                new_smooth = smooth_max(sc, pc, beta)
                if s < cur_score or (s == cur_score and new_smooth < cur_smooth):
                    elist = new_elist
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score, cur_sc, cur_pc = s, sc, pc
            elif r < 0.65:
                # value move: random
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 80)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s, sc, pc = fast_score_with_counts(new_vals, elist)
                cur_smooth = smooth_max(cur_sc, cur_pc, beta)
                new_smooth = smooth_max(sc, pc, beta)
                if s < cur_score or (s == cur_score and new_smooth < cur_smooth):
                    vals = new_vals
                    cur_score, cur_sc, cur_pc = s, sc, pc
            else:
                # targeted collision-seeking: pick a vertex and try to make
                # its edge sums/products collide with existing ones
                idx = random.randint(0, n - 1)
                # Find neighbors of idx in current edge set
                neighbors = [v for u, v in elist if u == idx or v == idx]
                if not neighbors:
                    continue
                # Get current sums and products involving idx
                existing_sums = set()
                existing_prods = set()
                for u, v in elist:
                    if u == idx or v == idx:
                        other = v if u == idx else u
                        existing_sums.add(vals[idx] + vals[other])
                        existing_prods.add(vals[idx] * vals[other])
                # Try to set vals[idx] so that for some neighbor, the sum
                # matches an existing sum or product matches an existing product
                candidates = set()
                for s_val in existing_sums:
                    for nb in neighbors:
                        cand = s_val - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                for p_val in existing_prods:
                    for nb in neighbors:
                        if vals[nb] != 0 and p_val % vals[nb] == 0:
                            cand = p_val // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                # Also try to match with non-neighbor edges
                for u, v in elist:
                    if u == idx or v == idx:
                        continue
                    other_sum = vals[u] + vals[v]
                    other_prod = vals[u] * vals[v]
                    for nb in neighbors:
                        cand = other_sum - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                        if vals[nb] != 0 and other_prod % vals[nb] == 0:
                            cand = other_prod // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                # Also try small random values
                candidates.add(random.randint(1, 50))
                if not candidates:
                    continue
                new_val = random.choice(list(candidates))
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s, sc, pc = fast_score_with_counts(new_vals, elist)
                cur_smooth = smooth_max(cur_sc, cur_pc, beta)
                new_smooth = smooth_max(sc, pc, beta)
                if s < cur_score or (s == cur_score and new_smooth < cur_smooth):
                    vals = new_vals
                    cur_score, cur_sc, cur_pc = s, sc, pc
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    def try_direct_construction():
        """Direct construction: find values + edges with high collision using CSP approach."""
        best = None
        # Strategy: pick 14 values from a small pool, then find 35 edges.
        # Small pool -> small product range -> collisions.
        pools = [
            list(range(1, 25)),
            list(range(2, 30)),
            list(range(1, 20)),
            [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
            [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
            [2,3,4,5,6,7,8,9,10,11,12,13,14,16],
            [1,2,3,4,5,6,7,8,9,11,12,14,16,18],
            # Add more structured pools
            [1,2,3,4,5,6,7,8,9,10,12,14,16,18],
            [1,2,3,4,5,6,7,8,9,10,11,12,14,16],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28],
            [1,3,5,7,9,11,13,15,17,19,21,23,25,27],
            [1,2,4,8,16,3,6,12,24,5,10,20,40,7],
        ]
        for pool in pools:
            for _ in range(4):
                vals = random.sample(pool, n)
                # Try both greedy and local search
                edges_greedy = greedy_edges_max_collision(vals)
                sc_greedy = fast_score(vals, edges_greedy)
                
                edges_ls, sc_ls = best_edges_for_values(vals, budget=2000)
                
                # Also try CSP approach with target K
                target_k = min(sc_greedy, sc_ls)
                edges_csp, sc_csp = csp_edge_search(vals, target_k, budget=50)
                
                # Pick best among all three
                candidates = [(sc_greedy, edges_greedy), (sc_ls, edges_ls), (sc_csp, edges_csp)]
                candidates.sort(key=lambda x: x[0])
                sc, edges = candidates[0]
                
                if best is None or sc < best[0]:
                    best = (sc, list(vals), edges)
                if time.time() > deadline - 35:
                    return best
        return best

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: direct construction
    d = try_direct_construction()
    if d is not None:
        best_score, best_vals, best_edges = d

    # Phase 2: joint search from multiple starts
    random.seed(1234)
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), set(best_edges)))
    # more random starts with small values
    for _ in range(10):
        vals = random.sample(range(1, 60), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))
    
    # Add CSP-informed starts: try to find value sets where many edges share sums/products
    for _ in range(5):
        vals = random.sample(range(1, 40), n)
        # Use CSP to find good edges
        edges_csp, sc_csp = csp_edge_search(vals, 20, budget=30)
        starts.append((vals, set(edges_csp)))

    remaining = deadline - time.time() - 10.0
    per = max(3000, int(remaining / len(starts) / 0.00005))
    for vals0, edges0 in starts:
        if time.time() > deadline - 10.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges

    # Phase 3: polish edges for best values with smooth surrogate
    if time.time() < deadline - 5.0:
        elist = list(best_edges)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(best_vals, elist)
        it = 0
        beta = 0.3
        while it < 25000 and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            new_elist = list(elist)
            new_elist[i] = e_in
            s, sc, pc = fast_score_with_counts(best_vals, new_elist)
            cur_smooth = smooth_max(cur_sc, cur_pc, beta)
            new_smooth = smooth_max(sc, pc, beta)
            if s < best_score or (s == best_score and new_smooth < cur_smooth):
                elist = new_elist
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s
                cur_score, cur_sc, cur_pc = s, sc, pc

    # Phase 4: polish values for best edges with targeted moves and smooth surrogate
    if time.time() < deadline - 2.0:
        elist = list(best_edges)
        vals = list(best_vals)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(vals, elist)
        it = 0
        beta = 0.3
        while it < 20000 and time.time() < deadline - 1.0:
            it += 1
            r = random.random()
            if r < 0.5:
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 100)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s, sc, pc = fast_score_with_counts(new_vals, elist)
                cur_smooth = smooth_max(cur_sc, cur_pc, beta)
                new_smooth = smooth_max(sc, pc, beta)
                if s < best_score or (s == best_score and new_smooth < cur_smooth):
                    vals = new_vals
                    best_score = s
                    best_vals = new_vals
                    cur_score, cur_sc, cur_pc = s, sc, pc
            else:
                # targeted: try to create a collision
                idx = random.randint(0, n - 1)
                neighbors = [v for u, v in elist if u == idx or v == idx]
                if not neighbors:
                    continue
                existing_sums = set()
                existing_prods = set()
                for u, v in elist:
                    if u == idx or v == idx:
                        other = v if u == idx else u
                        existing_sums.add(vals[idx] + vals[other])
                        existing_prods.add(vals[idx] * vals[other])
                candidates = set()
                for s_val in existing_sums:
                    for nb in neighbors:
                        cand = s_val - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                for p_val in existing_prods:
                    for nb in neighbors:
                        if vals[nb] != 0 and p_val % vals[nb] == 0:
                            cand = p_val // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                # Also try to match with non-neighbor edges
                for u, v in elist:
                    if u == idx or v == idx:
                        continue
                    other_sum = vals[u] + vals[v]
                    other_prod = vals[u] * vals[v]
                    for nb in neighbors:
                        cand = other_sum - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                        if vals[nb] != 0 and other_prod % vals[nb] == 0:
                            cand = other_prod // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                candidates.add(random.randint(1, 50))
                if not candidates:
                    continue
                new_val = random.choice(list(candidates))
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s, sc, pc = fast_score_with_counts(new_vals, elist)
                cur_smooth = smooth_max(cur_sc, cur_pc, beta)
                new_smooth = smooth_max(sc, pc, beta)
                if s < best_score or (s == best_score and new_smooth < cur_smooth):
                    vals = new_vals
                    best_score = s
                    best_vals = new_vals
                    cur_score, cur_sc, cur_pc = s, sc, pc

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
