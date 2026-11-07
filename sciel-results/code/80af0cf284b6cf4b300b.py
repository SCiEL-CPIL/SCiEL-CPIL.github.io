import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
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

    def greedy_edge_select(values, m_edges):
        """Select m edges greedily to maximize sum/product collisions."""
        selected = []
        selected_set = set()
        sum_counts = {}
        prod_counts = {}
        
        for edge in all_edges:
            u, v = edge
            s = values[u] + values[v]
            p = values[u] * values[v]
            # Score: how many times this sum/prod already appears
            s_count = sum_counts.get(s, 0)
            p_count = prod_counts.get(p, 0)
            # Prefer edges that create collisions
            priority = s_count * 10 + p_count * 10 + random.random() * 0.1
            selected.append((priority, edge))
            sum_counts[s] = s_count + 1
            prod_counts[p] = p_count + 1
        
        selected.sort(key=lambda x: -x[0])
        result = [edge for _, edge in selected[:m_edges]]
        return result

    def local_search(values, edges, budget_iters=50000):
        """Local search with edge swaps and value perturbations."""
        vals = list(values)
        edges = list(edges)
        edges_set = set(edges)
        
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.99995
        
        it = 0
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            
            if random.random() < 0.7:
                # Edge swap
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(vals, new_edges)
                
                delta = new_score - cur_score
                if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
            else:
                # Value perturbation
                idx = random.randint(0, n - 1)
                old_val = vals[idx]
                # Try nearby values or structured values
                candidates = [old_val + random.randint(-5, 5), 
                              random.randint(1, 100),
                              random.choice([1,2,3,4,5,6,7,8,9,10,12,15,16,18,20,24,25,30,32,36,40,48,50,60,64,72,80,90,100])]
                for new_val in candidates:
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val == old_val:
                        continue
                    if any(v == new_val for k, v in enumerate(vals) if k != idx):
                        continue
                    
                    vals[idx] = new_val
                    new_score = score_of(vals, edges)
                    delta = new_score - cur_score
                    if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        cur_score = new_score
                        break
                    else:
                        vals[idx] = old_val

            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = list(edges)
            
            T = max(T_min, T * cooling_rate)

        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Candidate value sets with strong multiplicative/additive structure
    value_sets = [
        # Powers of 2: products are powers of 2, sums have structure
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Small integers with many factor pairs
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20],
        # Multiples of small numbers
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48],
        # Mixed: small + powers
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        # Geometric-ish
        [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160],
    ]
    
    for vs in value_sets:
        if time.time() > deadline - 15.0:
            break
        # Greedy edge selection
        edges = greedy_edge_select(vs, m)
        sc, v, e = local_search(vs, edges, budget_iters=30000)
        if sc < best_score:
            best_score, best_vals, best_edges = sc, v, e
    
    # Also try random value sets
    for _ in range(3):
        if time.time() > deadline - 15.0:
            break
        vals = random.sample(range(1, 100), n)
        edges = greedy_edge_select(vals, m)
        sc, v, e = local_search(vals, edges, budget_iters=30000)
        if sc < best_score:
            best_score, best_vals, best_edges = sc, v, e

    # Final polish
    if best_vals is not None and time.time() < deadline - 10.0:
        sc4, v4, e4 = local_search(best_vals, best_edges, budget_iters=50000)
        if sc4 < best_score:
            best_score, best_vals, best_edges = sc4, v4, e4

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
