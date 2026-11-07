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

    def score_of(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def optimize_edges(values, budget_iters=2000):
        """
        Optimizes edges for fixed values.
        Returns (edges, score).
        """
        # Precompute sums and products for all edges
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Greedy initialization
        # Sort by "density": how many other edges share s or p?
        # Or just random? Let's try sorting by sum then product to cluster.
        # Actually, we want to MINIMIZE distinct sums/prods.
        # So we want to pick edges that share sums/prods with already picked edges.
        
        edges = []
        edge_set = set()
        sums = set()
        prods = set()
        
        # Strategy: Start with a random edge, then add edges that maximize overlap.
        # This is expensive.
        # Alternative: Random start, then local search.
        
        # Let's do a few random restarts for edge optimization
        best_edges = None
        best_score = float('inf')
        
        for restart in range(5):
            edges = random.sample(all_edges, m)
            edge_set = set(edges)
            sums = set()
            prods = set()
            for u, v in edges:
                sums.add(values[u] + values[v])
                prods.add(values[u] * values[v])
            cur_score = max(len(sums), len(prods))
            
            no_improve = 0
            it = 0
            while it < budget_iters and time.time() < deadline - 5.0:
                it += 1
                # Random swap
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edge_set:
                    continue
                
                # Check if swap improves score
                # Remove e_out
                s_out = values[e_out[0]] + values[e_out[1]]
                p_out = values[e_out[0]] * values[e_out[1]]
                
                # Add e_in
                s_in = values[e_in[0]] + values[e_in[1]]
                p_in = values[e_in[0]] * values[e_in[1]]
                
                # We need to know if s_out/p_out become unique, and if s_in/p_in are new.
                # This requires counting frequencies.
                # For speed, we can just recompute score if the swap looks promising?
                # Or maintain counts.
                
                # Let's maintain counts for efficiency
                # This is getting complex. Let's just recompute score if a simple heuristic passes.
                # Heuristic: If s_in in sums and p_in in prods, it's likely good.
                # If s_out not in sums (after removal) or p_out not in prods, it's likely bad to remove.
                
                # To be safe and fast enough, let's just recompute if:
                # 1. s_in is already in sums OR s_out becomes unique (bad)
                # Actually, just recomputing score for 35 edges is ~70 ops.
                # 2000 iters * 70 ops = 140k ops. Very fast.
                
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
                    if no_improve > 500 and random.random() < 0.1:
                        no_improve = 0 # Simulated annealing step?
                        # Accept worse move with small prob
                        if random.random() < 0.05:
                            edges = new_edges
                            edge_set.discard(e_out)
                            edge_set.add(e_in)
                            cur_score = new_score
                            no_improve = 0

            if cur_score < best_score:
                best_score = cur_score
                best_edges = edges
                
        return best_edges, best_score

    def optimize_values(values, edges, budget_iters=1000):
        """
        Optimizes values for fixed edges.
        """
        vals = list(values)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            # Perturbation
            if random.random() < 0.7:
                new_val = old_val + random.randint(-10, 10)
            else:
                new_val = random.randint(1, 100)
            
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
                # Accept worse moves to escape local minima
                if no_improve > 100 and random.random() < 0.05:
                    vals = new_vals
                    cur_score = new_score
                    no_improve = 0

        return best_vals, best_score

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Initial Candidates
    candidates = [
        list(range(1, 15)),
        [2**i for i in range(14)],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
        random.sample(range(1, 50), 14),
        random.sample(range(1, 100), 14)
    ]

    for init_vals in candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Optimize edges first
        edges, score_e = optimize_edges(init_vals, budget_iters=1000)
        if score_e < best_score:
            best_score = score_e
            best_vals = list(init_vals)
            best_edges = edges
            
        # Optimize values
        vals, score_v = optimize_values(best_vals, best_edges, budget_iters=500)
        if score_v < best_score:
            best_score = score_v
            best_vals = vals
            best_edges = edges
            
        # Iterate a few times
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            edges, score_e = optimize_edges(best_vals, budget_iters=500)
            if score_e < best_score:
                best_score = score_e
                best_edges = edges
                
            vals, score_v = optimize_values(best_vals, best_edges, budget_iters=500)
            if score_v < best_score:
                best_score = score_v
                best_vals = vals

    # Final polish
    if best_vals and best_edges:
        edges, score_e = optimize_edges(best_vals, budget_iters=5000)
        if score_e < best_score:
            best_score = score_e
            best_edges = edges

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
