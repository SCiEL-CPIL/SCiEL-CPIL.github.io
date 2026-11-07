import time
import random
import math
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def find_edges_for_targets(values, target_sums, target_prods, budget_iters=3000):
        """
        Given target sums and products, find 35 edges that achieve them.
        Uses constraint satisfaction: for each edge, check if its sum/product match targets.
        """
        # Build lookup: for each possible edge, what are its sum and product?
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            edge_sum[(u, v)] = values[u] + values[v]
            edge_prod[(u, v)] = values[u] * values[v]
        
        # For each edge, determine if it's "good" (matches at least one target)
        # We want to pick edges that match targets to minimize distinct values
        
        best_edges = None
        best_score = float('inf')
        
        for restart in range(8):
            edges = []
            edge_set = set()
            
            # Greedy: repeatedly pick the edge that best fits remaining targets
            remaining_sums = set(target_sums)
            remaining_prods = set(target_prods)
            
            # Score each edge by how well it fits
            def edge_fitness(e):
                s = edge_sum[e]
                p = edge_prod[e]
                fit = 0
                if s in remaining_sums:
                    fit += 2
                if p in remaining_prods:
                    fit += 2
                # Bonus for edges that cover multiple targets
                return fit
            
            for _ in range(m):
                if time.time() > deadline - 5.0:
                    break
                # Pick best edge not already selected
                best_e = None
                best_fit = -1
                for e in all_edges:
                    if e in edge_set:
                        continue
                    fit = edge_fitness(e)
                    # Add randomness to avoid getting stuck
                    fit += random.random() * 0.5
                    if fit > best_fit:
                        best_fit = fit
                        best_e = e
                if best_e is None:
                    # All edges used, pick random
                    available = [e for e in all_edges if e not in edge_set]
                    if not available:
                        break
                    best_e = random.choice(available)
                edges.append(best_e)
                edge_set.add(best_e)
                # Update remaining targets
                s = edge_sum[best_e]
                p = edge_prod[best_e]
                # Don't remove from remaining since other edges might also use them
            
            # Local search to improve
            cur_score = score_of(values, edges)
            no_improve = 0
            it = 0
            while it < budget_iters and time.time() < deadline - 5.0 and len(edges) == m:
                it += 1
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edge_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(values, new_edges)
                
                if new_score < cur_score:
                    edges = new_edges
                    edge_set.discard(e_out)
                    edge_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 200 and random.random() < 0.03:
                        edges = new_edges
                        edge_set.discard(e_out)
                        edge_set.add(e_in)
                        cur_score = new_score
                        no_improve = 0
            
            if len(edges) == m and cur_score < best_score:
                best_score = cur_score
                best_edges = list(edges)
        
        return best_edges, best_score

    def optimize_values_for_edges(values, edges, budget_iters=2000):
        """
        Optimize values given fixed edges to minimize distinct sums/prods.
        Uses targeted perturbation: try to make specific edge sums/products collide.
        """
        vals = list(values)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        
        # Precompute which edges share which sums/prods
        def get_collision_info(vals):
            sum_map = defaultdict(list)
            prod_map = defaultdict(list)
            for u, v in edges:
                s = vals[u] + vals[v]
                p = vals[u] * vals[v]
                sum_map[s].append((u, v))
                prod_map[p].append((u, v))
            return sum_map, prod_map
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            # Try to create collisions: pick a neighbor and adjust to match another edge's sum/prod
            neighbors = [v for u, v in edges if u == idx or v == idx]
            
            if random.random() < 0.5 and neighbors:
                # Targeted: try to make sum or product match an existing one
                target_neighbor = random.choice(neighbors)
                edge = (min(idx, target_neighbor), max(idx, target_neighbor))
                other_edges = [e for e in edges if e != edge]
                
                if other_edges:
                    other_u, other_v = random.choice(other_edges)
                    # Try to make vals[idx] + vals[target_neighbor] == vals[other_u] + vals[other_v]
                    target_sum = vals[other_u] + vals[other_v]
                    candidate = target_sum - vals[target_neighbor]
                    if 1 <= candidate <= 10000 and candidate != old_val:
                        if not any(v == candidate for j, v in enumerate(vals) if j != idx):
                            new_vals = list(vals)
                            new_vals[idx] = candidate
                            new_score = score_of(new_vals, edges)
                            if new_score < cur_score:
                                vals = new_vals
                                cur_score = new_score
                                no_improve = 0
                                if cur_score < best_score:
                                    best_score = cur_score
                                    best_vals = list(vals)
                                continue
                    # Try product collision
                    target_prod = vals[other_u] * vals[other_v]
                    if vals[target_neighbor] != 0:
                        candidate = target_prod // vals[target_neighbor]
                        if candidate * vals[target_neighbor] == target_prod:
                            if 1 <= candidate <= 10000 and candidate != old_val:
                                if not any(v == candidate for j, v in enumerate(vals) if j != idx):
                                    new_vals = list(vals)
                                    new_vals[idx] = candidate
                                    new_score = score_of(new_vals, edges)
                                    if new_score < cur_score:
                                        vals = new_vals
                                        cur_score = new_score
                                        no_improve = 0
                                        if cur_score < best_score:
                                            best_score = cur_score
                                            best_vals = list(vals)
                                        continue
            
            # Random perturbation
            if random.random() < 0.7:
                new_val = old_val + random.randint(-20, 20)
            else:
                new_val = random.randint(1, 200)
            
            if new_val <= 0 or new_val > 10000:
                continue
            if any(v == new_val for j, v in enumerate(vals) if j != idx):
                continue
                
            new_vals = list(vals)
            new_vals[idx] = new_val
            new_score = score_of(new_vals, edges)
            
            if new_score < cur_score:
                vals = new_vals
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(vals)
            else:
                no_improve += 1
                if no_improve > 150 and random.random() < 0.04:
                    vals = new_vals
                    cur_score = new_score
                    no_improve = 0

        return best_vals, best_score

    def generate_structured_values():
        """
        Generate value sets designed to have many additive/multiplicative collisions.
        """
        candidates = []
        
        # Arithmetic progressions with different steps
        for start in range(1, 10):
            for step in range(1, 5):
                vals = [start + i * step for i in range(n)]
                if all(1 <= v <= 10000 for v in vals) and len(set(vals)) == n:
                    candidates.append(vals)
        
        # Geometric-like with small ratios
        for base in range(2, 8):
            vals = [base ** i for i in range(n)]
            if all(v <= 10000 for v in vals):
                candidates.append(vals)
        
        # Mixed: some small, some large
        for _ in range(50):
            small = random.sample(range(1, 30), 7)
            large = random.sample(range(100, 500), 7)
            vals = small + large
            random.shuffle(vals)
            if len(set(vals)) == n and all(1 <= v <= 10000 for v in vals):
                candidates.append(vals)
        
        # Pairs that sum to same value
        for _ in range(30):
            target_sum = random.randint(20, 100)
            vals = []
            for i in range(n // 2):
                a = random.randint(1, target_sum - 1)
                b = target_sum - a
                if a != b and a not in vals and b not in vals:
                    vals.extend([a, b])
            while len(vals) < n:
                v = random.randint(1, 100)
                if v not in vals:
                    vals.append(v)
            random.shuffle(vals)
            if len(set(vals)) == n:
                candidates.append(vals)
        
        return candidates

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Generate structured value candidates
    value_candidates = generate_structured_values()
    
    # Add some random ones
    for _ in range(20):
        vals = random.sample(range(1, 300), n)
        value_candidates.append(vals)
    
    random.shuffle(value_candidates)
    
    # Limit candidates to fit in time
    max_candidates = 40
    value_candidates = value_candidates[:max_candidates]
    
    for init_vals in value_candidates:
        if time.time() > deadline - 15.0:
            break
        
        # Get current score with random edges
        random_edges = random.sample(all_edges, m)
        cur_score = score_of(init_vals, random_edges)
        
        # Optimize values first
        vals, score_v = optimize_values_for_edges(init_vals, random_edges, budget_iters=800)
        if score_v < best_score:
            best_score = score_v
            best_vals = vals
            best_edges = list(random_edges)
        
        # Now optimize edges for the optimized values
        if best_vals is not None:
            # Extract target sums and prods from current best
            target_sums = set()
            target_prods = set()
            for u, v in best_edges:
                target_sums.add(best_vals[u] + best_vals[v])
                target_prods.add(best_vals[u] * best_vals[v])
            
            edges, score_e = find_edges_for_targets(best_vals, target_sums, target_prods, budget_iters=1500)
            if edges is not None and score_e < best_score:
                best_score = score_e
                best_edges = edges
        
        # Iterate a few times
        for _ in range(2):
            if time.time() > deadline - 8.0:
                break
            if best_vals is None or best_edges is None:
                break
            
            target_sums = set()
            target_prods = set()
            for u, v in best_edges:
                target_sums.add(best_vals[u] + best_vals[v])
                target_prods.add(best_vals[u] * best_vals[v])
            
            edges, score_e = find_edges_for_targets(best_vals, target_sums, target_prods, budget_iters=1000)
            if edges is not None and score_e < best_score:
                best_score = score_e
                best_edges = edges
            
            vals, score_v = optimize_values_for_edges(best_vals, best_edges, budget_iters=600)
            if score_v < best_score:
                best_score = score_v
                best_vals = vals

    # Final polish
    if best_vals is not None and best_edges is not None:
        target_sums = set()
        target_prods = set()
        for u, v in best_edges:
            target_sums.add(best_vals[u] + best_vals[v])
            target_prods.add(best_vals[u] * best_vals[v])
        
        edges, score_e = find_edges_for_targets(best_vals, target_sums, target_prods, budget_iters=3000)
        if edges is not None and score_e < best_score:
            best_score = score_e
            best_edges = edges

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
