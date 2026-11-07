import time
import random
import math
from itertools import combinations

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
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
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
    
    def components(self):
        comp = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in comp:
                comp[r] = []
            comp[r].append(i)
        return comp

def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score_delta(values, edge_list, e_out, e_in):
        """Compute new score after removing e_out and adding e_in."""
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        u_out, v_out = e_out
        s_out = values[u_out] + values[v_out]
        p_out = values[u_out] * values[v_out]
        sum_counts[s_out] -= 1
        prod_counts[p_out] -= 1
        if sum_counts[s_out] == 0:
            del sum_counts[s_out]
        if prod_counts[p_out] == 0:
            del prod_counts[p_out]
        u_in, v_in = e_in
        s_in = values[u_in] + values[v_in]
        p_in = values[u_in] * values[v_in]
        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
        return max(len(sum_counts), len(prod_counts))

    def collision_score(values):
        """Score how many edge sum/product collisions exist for given values.
        Higher is better (more collisions means fewer distinct values needed)."""
        edge_sums = {}
        edge_prods = {}
        for e in all_edges:
            s = values[e[0]] + values[e[1]]
            p = values[e[0]] * values[e[1]]
            edge_sums[e] = s
            edge_prods[e] = p
        
        # Count how many edges share each sum/product
        sum_freq = {}
        prod_freq = {}
        for e in all_edges:
            s = edge_sums[e]
            p = edge_prods[e]
            sum_freq[s] = sum_freq.get(s, 0) + 1
            prod_freq[p] = prod_freq.get(p, 0) + 1
        
        # Score: sum of (freq-1) for each value = number of "saved" distinct values
        sum_saves = sum(max(0, f - 1) for f in sum_freq.values())
        prod_saves = sum(max(0, f - 1) for f in prod_freq.values())
        
        # Also penalize too many distinct values
        distinct_sums = len(sum_freq)
        distinct_prods = len(prod_freq)
        
        return sum_saves + prod_saves - 0.5 * (distinct_sums + distinct_prods)

    def constrained_edge_search(vals, K, time_limit=3.0):
        """Use Union-Find to group edges by sum/product, then backtrack to find
        a set of m edges using at most K distinct sums and K distinct products."""
        t_end = time.time() + time_limit
        
        # Precompute
        edge_sums = {}
        edge_prods = {}
        for e in all_edges:
            edge_sums[e] = vals[e[0]] + vals[e[1]]
            edge_prods[e] = vals[e[0]] * vals[e[1]]
        
        # Group edges by sum
        sum_to_edges = {}
        for e in all_edges:
            s = edge_sums[e]
            if s not in sum_to_edges:
                sum_to_edges[s] = []
            sum_to_edges[s].append(e)
        
        # Group edges by product
        prod_to_edges = {}
        for e in all_edges:
            p = edge_prods[e]
            if p not in prod_to_edges:
                prod_to_edges[p] = []
            prod_to_edges[p].append(e)
        
        # Sort sum groups by size (larger groups first for better coverage)
        sum_groups_sorted = sorted(sum_to_edges.items(), key=lambda x: -len(x[1]))
        prod_groups_sorted = sorted(prod_to_edges.items(), key=lambda x: -len(x[1]))
        
        # Strategy: select sum groups and prod groups to cover m edges
        # Use a greedy + local search approach
        
        best_edges = None
        best_score = float('inf')
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() > t_end:
                break
            
            # Randomly initialize with some edges
            chosen = set()
            chosen_list = []
            active_sums = set()
            active_prods = set()
            
            # Add random edges first
            for _ in range(min(5, m)):
                e = random.choice(all_edges)
                if e not in chosen:
                    chosen.add(e)
                    chosen_list.append(e)
                    active_sums.add(edge_sums[e])
                    active_prods.add(edge_prods[e])
            
            # Greedily add edges that maximize collision
            for _ in range(m - len(chosen_list)):
                if time.time() > t_end:
                    break
                best_e = None
                best_sc = -float('inf')
                
                # Candidate edges: those with sum or product already active
                candidates = [e for e in all_edges if e not in chosen]
                
                for e in candidates:
                    s = edge_sums[e]
                    p = edge_prods[e]
                    sc = 0
                    if s in active_sums:
                        sc += 20
                    if p in active_prods:
                        sc += 20
                    # Prefer edges from large groups
                    sc += len(sum_to_edges.get(s, [])) * 0.5
                    sc += len(prod_to_edges.get(p, [])) * 0.5
                    if sc > best_sc:
                        best_sc = sc
                        best_e = e
                
                if best_e is None:
                    break
                chosen.add(best_e)
                chosen_list.append(best_e)
                active_sums.add(edge_sums[best_e])
                active_prods.add(edge_prods[best_e])
            
            if len(chosen_list) < m:
                continue
            
            cur_sums = set(edge_sums[e] for e in chosen_list)
            cur_prods = set(edge_prods[e] for e in chosen_list)
            cur_score = max(len(cur_sums), len(cur_prods))
            
            if cur_score < best_score:
                best_score = cur_score
                best_edges = list(chosen_list)
                if best_score <= K:
                    break
            
            # Local search to improve
            cur = list(chosen_list)
            cur_set = set(cur)
            it = 0
            while it < 2000 and time.time() < t_end:
                it += 1
                i = random.randint(0, m - 1)
                e_out = cur[i]
                e_in = random.choice(all_edges)
                if e_in in cur_set:
                    continue
                
                # Compute delta
                new_cur = list(cur)
                new_cur[i] = e_in
                new_sums = set(edge_sums[e] for e in new_cur)
                new_prods = set(edge_prods[e] for e in new_cur)
                new_score = max(len(new_sums), len(new_prods))
                
                if new_score <= cur_score:
                    cur = new_cur
                    cur_set.discard(e_out)
                    cur_set.add(e_in)
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = list(cur)
                        if best_score <= K:
                            break
                else:
                    # Allow some uphill moves for exploration
                    if random.random() < 0.1 and new_score - cur_score <= 2:
                        cur = new_cur
                        cur_set.discard(e_out)
                        cur_set.add(e_in)
                        cur_score = new_score
        
        if best_edges is not None and best_score <= K:
            return best_edges
        elif best_edges is not None:
            return best_edges
        return None

    def smart_value_search(edges, time_limit=3.0):
        """Search for values that maximize collisions on the given edges."""
        t_end = time.time() + time_limit
        best_vals = None
        best_score = float('inf')
        
        # Start from several random initializations
        for restart in range(30):
            if time.time() > t_end:
                break
            
            # Random initial values
            vals = random.sample(range(1, 50), n)
            cur_score = score_of(vals, edges)
            
            # Hill climbing
            improved = True
            it = 0
            while improved and it < 500 and time.time() < t_end:
                improved = False
                it += 1
                for idx in range(n):
                    # Try nearby values
                    for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20]:
                        new_val = vals[idx] + delta
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in vals and vals.index(new_val) != idx:
                            continue
                        new_vals = list(vals)
                        new_vals[idx] = new_val
                        s = score_of(new_vals, edges)
                        if s < cur_score:
                            vals = new_vals
                            cur_score = s
                            improved = True
                            break
            
            # Also try random jumps
            for _ in range(50):
                if time.time() > t_end:
                    break
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 100)
                if new_val in vals and vals.index(new_val) != idx:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, edges)
                if s < cur_score:
                    vals = new_vals
                    cur_score = s
            
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
        
        return best_score, best_vals

    def joint_refinement(vals0, edges0, budget):
        """Alternate value and edge moves with fast scoring."""
        vals = list(vals0)
        edges = set(edges0)
        elist = list(edges)
        cur_score = score_of(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(edges)
        it = 0
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            if random.random() < 0.5:
                # edge move
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                s = fast_score_delta(vals, elist, e_out, e_in)
                if s <= cur_score:
                    elist[i] = e_in
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score = s
            else:
                # value move
                idx = random.randint(0, n - 1)
                # Try values that create more collisions
                new_val = random.randint(1, 80)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: Constrained edge search with various value pools
    pools = [
        list(range(1, 25)),
        list(range(2, 30)),
        list(range(1, 20)),
        [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
        [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
        [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
        list(range(1, 35)),
        [2,3,4,5,6,7,8,9,10,11,12,13,14,16],
    ]
    
    for pool in pools:
        if time.time() > deadline - 50:
            break
        for _ in range(8):
            vals = random.sample(pool, n)
            # Try multiple target K values
            for K in [8, 10, 12, 14, 16, 18]:
                result = constrained_edge_search(vals, K, time_limit=2.0)
                if result is not None:
                    sc = score_of(vals, result)
                    if sc < best_score:
                        best_score = sc
                        best_vals = list(vals)
                        best_edges = set(result)
            if time.time() > deadline - 45:
                break

    # Phase 2: Smart value search on best edges
    if best_edges is not None and time.time() < deadline - 30:
        edges_list = list(best_edges)
        sc, vals = smart_value_search(edges_list, time_limit=20.0)
        if sc < best_score:
            best_score = sc
            best_vals = vals

    # Phase 3: Joint search from multiple starts
    random.seed(1234)
    starts = []
    if best_vals is not None and best_edges is not None:
        starts.append((list(best_vals), set(best_edges)))
    for _ in range(8):
        vals = random.sample(range(1, 60), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    remaining = deadline - time.time() - 8.0
    if remaining > 10:
        per = max(2000, int(remaining / len(starts) / 0.0001))
        for vals0, edges0 in starts:
            if time.time() > deadline - 8.0:
                break
            sc, vals, edges = joint_refinement(vals0, edges0, per)
            if sc < best_score:
                best_score = sc
                best_vals = vals
                best_edges = edges

    # Phase 4: Polish edges for best values
    if best_vals is not None and time.time() < deadline - 4.0:
        elist = list(best_edges)
        it = 0
        while it < 20000 and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            s = fast_score_delta(best_vals, elist, e_out, e_in)
            if s < best_score:
                elist[i] = e_in
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s

    # Phase 5: Polish values for best edges
    if best_vals is not None and time.time() < deadline - 1.0:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 15000 and time.time() < deadline - 0.5:
            it += 1
            idx = random.randint(0, n - 1)
            # Try values that create more collisions
            new_val = random.randint(1, 100)
            if new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = score_of(new_vals, elist)
            if s < best_score:
                vals = new_vals
                best_score = s
                best_vals = new_vals

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
