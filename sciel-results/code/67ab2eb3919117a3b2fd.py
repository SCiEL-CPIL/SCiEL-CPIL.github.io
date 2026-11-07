"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def build_compatibility_graph(values):
        """Build graph where vertices are edge-indices, edge exists if they share sum or product."""
        sums = [values[u] + values[v] for u, v in all_pairs]
        prods = [values[u] * values[v] for u, v in all_pairs]
        
        # For each pair of edge-indices, check if they share a sum or product
        # Use bitsets for efficiency
        compat = [0] * num_pairs
        for i in range(num_pairs):
            for j in range(i + 1, num_pairs):
                if sums[i] == sums[j] or prods[i] == prods[j]:
                    compat[i] |= (1 << j)
                    compat[j] |= (1 << i)
        return compat, sums, prods

    def greedy_clique(compat, sums, prods, values):
        """Greedy clique construction: repeatedly add the vertex that maximizes overlap."""
        remaining = set(range(num_pairs))
        clique = []
        clique_sums = set()
        clique_prods = set()
        
        for _ in range(m):
            if not remaining:
                break
            best_v = -1
            best_key = None
            for v in remaining:
                # Count how many of v's neighbors are already in clique
                in_clique = sum(1 for c in clique if (compat[v] >> c) & 1)
                # New distinct sums/prods if we add v
                s = sums[v]
                p = prods[v]
                new_sums = len(clique_sums) if s in clique_sums else len(clique_sums) + 1
                new_prods = len(clique_prods) if p in clique_prods else len(clique_prods) + 1
                key = (max(new_sums, new_prods), new_sums, new_prods, -in_clique)
                if best_key is None or key < best_key:
                    best_key = key
                    best_v = v
            
            if best_v < 0:
                break
            
            clique.append(best_v)
            remaining.discard(best_v)
            clique_sums.add(sums[best_v])
            clique_prods.add(prods[best_v])
            # Remove vertices not compatible with best_v
            remaining = {v for v in remaining if (compat[best_v] >> v) & 1}
        
        if len(clique) < m:
            return None
        edges = [list(all_pairs[v]) for v in clique]
        return edges

    def branch_and_bound_clique(compat, sums, prods, values, deadline):
        """Branch and bound to find clique of size m with min distinct sums/prods."""
        # Sort vertices by degree descending for better pruning
        degrees = [bin(compat[i]).count('1') for i in range(num_pairs)]
        order = sorted(range(num_pairs), key=lambda x: -degrees[x])
        
        best_clique = None
        best_score = m  # worst case
        
        def dfs(candidates, clique, clique_sums, clique_prods, depth):
            nonlocal best_clique, best_score
            if time.time() > deadline:
                return
            if len(clique) == m:
                score = max(len(clique_sums), len(clique_prods))
                if score < best_score:
                    best_score = score
                    best_clique = list(clique)
                return
            if not candidates:
                return
            if len(clique) + len(candidates) < m:
                return
            
            # Try each candidate
            for idx, v in enumerate(candidates):
                s = sums[v]
                p = prods[v]
                new_sums = len(clique_sums) if s in clique_sums else len(clique_sums) + 1
                new_prods = len(clique_prods) if p in clique_prods else len(clique_prods) + 1
                cur_score = max(new_sums, new_prods)
                if cur_score >= best_score and len(clique) + len(candidates) - idx <= m:
                    continue
                
                # Find compatible candidates
                new_candidates = [c for c in candidates[idx+1:] if (compat[v] >> c) & 1]
                
                clique.append(v)
                if s not in clique_sums:
                    clique_sums.add(s)
                if p not in clique_prods:
                    clique_prods.add(p)
                
                dfs(new_candidates, clique, clique_sums, clique_prods, depth + 1)
                
                if s in clique_sums and len(clique_sums) > 1:
                    # Check if s is still used by other clique members
                    still_used = False
                    for c in clique[:-1]:
                        if sums[c] == s:
                            still_used = True
                            break
                    if not still_used:
                        clique_sums.discard(s)
                if p in clique_prods and len(clique_prods) > 1:
                    still_used = False
                    for c in clique[:-1]:
                        if prods[c] == p:
                            still_used = True
                            break
                    if not still_used:
                        clique_prods.discard(p)
                clique.pop()
        
        dfs(order, [], set(), set(), 0)
        
        if best_clique is None:
            return None
        edges = [list(all_pairs[v]) for v in best_clique]
        return edges

    def generate_structured_values(rng):
        """Generate values that create many sum/product collisions."""
        strategy = rng.randint(0, 4)
        
        if strategy == 0:
            # Small integers 1-14: many sum collisions (e.g., sum=15 has 7 pairs)
            values = list(range(1, 15))
            rng.shuffle(values)
            return values
        elif strategy == 1:
            # Arithmetic progression: a, a+d, a+2d, ...
            a = rng.randint(1, 5)
            d = rng.randint(2, 4)
            values = [a + i * d for i in range(14)]
            return values
        elif strategy == 2:
            # Pairs that share products: e.g., {2,3,4,6,8,12,16,24,32,48,64,96,128,192}
            # Products of (2^a * 3^b) collide
            vals = set()
            for a in range(0, 7):
                for b in range(0, 3):
                    v = (2 ** a) * (3 ** b)
                    if 1 <= v <= max_val:
                        vals.add(v)
            vals = list(vals)
            rng.shuffle(vals)
            values = vals[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        elif strategy == 3:
            # Geometric progression
            r = rng.choice([2, 3])
            values = [r ** i for i in range(14)]
            return values
        else:
            # Mix of small and structured
            base = list(range(1, 8))
            extra = [2 * x for x in range(1, 8)]
            values = base + extra
            rng.shuffle(values)
            return values

    def local_search_edges(values, edges, rng, max_iters=5000):
        """Local search: swap edges to reduce distinct sums/prods."""
        edges_set = set(tuple(sorted(e)) for e in edges)
        sums_dict = {}
        prods_dict = {}
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sums_dict[s] = sums_dict.get(s, 0) + 1
            prods_dict[p] = prods_dict.get(p, 0) + 1
        
        best_score = max(len(sums_dict), len(prods_dict))
        best_edges = [list(e) for e in edges]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random edge to replace
            idx = rng.randint(0, m - 1)
            old_u, old_v = edges[idx]
            old_s = values[old_u] + values[old_v]
            old_p = values[old_u] * values[old_v]
            
            # Try random new edge
            new_idx = rng.randint(0, num_pairs - 1)
            new_u, new_v = all_pairs[new_idx]
            new_s = values[new_u] + values[new_v]
            new_p = values[new_u] * values[new_v]
            
            new_edge_sorted = tuple(sorted((new_u, new_v)))
            old_edge_sorted = tuple(sorted((old_u, old_v)))
            
            if new_edge_sorted == old_edge_sorted or new_edge_sorted in edges_set:
                continue
            
            # Compute new score
            sc = dict(sums_dict)
            pc = dict(prods_dict)
            sc[old_s] -= 1
            if sc[old_s] == 0:
                del sc[old_s]
            sc[new_s] = sc.get(new_s, 0) + 1
            pc[old_p] -= 1
            if pc[old_p] == 0:
                del pc[old_p]
            pc[new_p] = pc.get(new_p, 0) + 1
            
            new_score = max(len(sc), len(pc))
            if new_score <= best_score:
                # Accept
                edges[idx] = [new_u, new_v]
                edges_set.discard(old_edge_sorted)
                edges_set.add(new_edge_sorted)
                sums_dict = sc
                prods_dict = pc
                if new_score < best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in edges]
        
        return best_edges, best_score

    def optimize_values(values, edges, rng, max_iters=200):
        """Perturb values to improve score."""
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        best_score = evaluate(best_values, best_edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            i = rng.randint(0, n - 1)
            old_val = best_values[i]
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val or new_val == old_val:
                continue
            if new_val in best_values:
                continue
            
            new_values = list(best_values)
            new_values[i] = new_val
            new_score = evaluate(new_values, best_edges)
            
            if new_score < best_score:
                best_values = new_values
                best_score = new_score
        
        return best_values, best_score

    # Main search
    rng = random.Random(12345)
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Structured values + clique search
    for restart in range(30):
        if time.time() > deadline:
            break
        
        values = generate_structured_values(rng)
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]
        
        compat, sums, prods = build_compatibility_graph(values)
        
        # Greedy clique
        edges = greedy_clique(compat, sums, prods, values)
        if edges:
            score = evaluate(values, edges)
            if score < best_score:
                best_score = score
                best_result = (list(values), [list(e) for e in edges])
        
        # Local search refinement
        if edges:
            edges_ls, score_ls = local_search_edges(values, edges, rng, max_iters=2000)
            if score_ls < best_score:
                best_score = score_ls
                best_result = (list(values), [list(e) for e in edges_ls])
        
        if best_score <= 8:
            break
    
    # Phase 2: Branch and bound on best values
    if best_result is not None:
        values, edges = best_result
        compat, sums, prods = build_compatibility_graph(values)
        edges_bnb = branch_and_bound_clique(compat, sums, prods, values, deadline)
        if edges_bnb:
            score = evaluate(values, edges_bnb)
            if score < best_score:
                best_score = score
                best_result = (list(values), [list(e) for e in edges_bnb])
        
        # Value optimization
        values_opt, edges_opt = optimize_values(values, edges, rng, max_iters=100)
        score_opt = evaluate(values_opt, edges_opt)
        if score_opt < best_score:
            best_score = score_opt
            best_result = (list(values_opt), [list(e) for e in edges_opt])
    
    # Phase 3: More diverse restarts
    for restart in range(40):
        if time.time() > deadline:
            break
        
        # Random values with structure
        base = rng.randint(1, 3)
        step = rng.randint(2, 6)
        values = [base + i * step for i in range(14)]
        rng.shuffle(values)
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]
        
        compat, sums, prods = build_compatibility_graph(values)
        edges = greedy_clique(compat, sums, prods, values)
        if edges:
            score = evaluate(values, edges)
            if score < best_score:
                best_score = score
                best_result = (list(values), [list(e) for e in edges])
            
            edges_ls, score_ls = local_search_edges(values, edges, rng, max_iters=1500)
            if score_ls < best_score:
                best_score = score_ls
                best_result = (list(values), [list(e) for e in edges_ls])
        
        if best_score <= 6:
            break
    
    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_pairs[:m]]
    else:
        values, edges = best_result
    
    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
