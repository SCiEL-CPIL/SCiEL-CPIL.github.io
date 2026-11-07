"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 90s limit
    
    n = 14
    m = 35
    
    # Strategy: 
    # 1. Use values 1..14 to minimize the range of sums and products.
    # 2. Greedily select edges to maximize overlap in sums/products.
    # 3. Use a simulated annealing-like local search with swap moves.
    # 4. Periodic restarts with perturbations to escape local optima.
    # 5. Final polish phase to fine-tune the best solution found.
    
    values = list(range(1, n + 1))
    
    # All possible edges (91 total)
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    # Precompute sum and product for all possible edges
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        edge_sums[(u, v)] = values[u] + values[v]
        edge_prods[(u, v)] = values[u] * values[v]
    
    def get_score(edges):
        """Compute max(distinct sums, distinct products) for a set of edges."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(edge_sums[(u, v)])
            prods.add(edge_prods[(u, v)])
        return max(len(sums), len(prods))
    
    # Greedy initial solution: pick edges that reuse existing sum/product values
    def greedy_initial():
        current_edges = []
        current_edge_set = set()
        current_sum_counts = {}
        current_prod_counts = {}
        
        # Sort edges by heuristic: prefer edges with smaller sums and products
        sorted_edges = sorted(all_edges, key=lambda e: (edge_sums[e], edge_prods[e]))
        
        for e in sorted_edges:
            if len(current_edges) >= m:
                break
            s = edge_sums[e]
            p = edge_prods[e]
            # Accept edge
            current_edges.append(e)
            current_edge_set.add(e)
            current_sum_counts[s] = current_sum_counts.get(s, 0) + 1
            current_prod_counts[p] = current_prod_counts.get(p, 0) + 1
        
        return current_edges, current_edge_set
    
    current_edges, current_edge_set = greedy_initial()
    current_score = get_score(current_edges)
    
    best_score = current_score
    best_edges = list(current_edges)
    best_edge_set = set(current_edges)
    
    # Simulated annealing / local search
    steps = 0
    no_improve = 0
    max_no_improve = 3000
    temperature = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        steps += 1
        
        # Pick a random edge to remove from current set
        idx_remove = random.randint(0, m - 1)
        e_remove = current_edges[idx_remove]
        
        # Pick a random non-edge to add
        # Sample from all_edges until we find one not in current set
        # Expected trials: 91 / (91-35) = 91/56 ≈ 1.6
        while True:
            e_add = random.choice(all_edges)
            if e_add not in current_edge_set:
                break
        
        # Perform swap
        new_edges = list(current_edges)
        new_edges[idx_remove] = e_add
        
        # Evaluate new state
        new_score = get_score(new_edges)
        
        # Acceptance criterion: always accept improvement, sometimes accept worse
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_edges = new_edges
            current_edge_set.discard(e_remove)
            current_edge_set.add(e_add)
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = list(current_edges)
                best_edge_set = set(current_edges)
                no_improve = 0
            else:
                no_improve += 1
        elif delta == 0:
            # Plateau: accept with some probability to wander
            if random.random() < 0.5:
                current_edges = new_edges
                current_edge_set.discard(e_remove)
                current_edge_set.add(e_add)
                # Score unchanged, so best_score unchanged
                no_improve += 1
        else:
            # Worse: accept with probability exp(-delta/temperature)
            if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                current_edges = new_edges
                current_edge_set.discard(e_remove)
                current_edge_set.add(e_add)
                current_score = new_score
                no_improve += 1
            else:
                no_improve += 1
        
        temperature *= cooling
        
        # Occasional restart to escape local optima
        if no_improve > max_no_improve:
            # Perturb: swap 3-5 random edges
            num_swaps = random.randint(3, 5)
            for _ in range(num_swaps):
                idx_r = random.randint(0, m - 1)
                e_r = current_edges[idx_r]
                while True:
                    e_a = random.choice(all_edges)
                    if e_a not in current_edge_set:
                        break
                current_edges[idx_r] = e_a
                current_edge_set.discard(e_r)
                current_edge_set.add(e_a)
            current_score = get_score(current_edges)
            no_improve = 0
            temperature = 1.0
    
    # Final polish: try to improve best_edges with targeted swaps
    if time.time() < deadline - 5:
        polish_edges = list(best_edges)
        polish_set = set(polish_edges)
        polish_score = get_score(polish_edges)
        polish_no_improve = 0
        max_polish_steps = 5000
        
        while time.time() < deadline - 5 and polish_no_improve < max_polish_steps:
            idx_r = random.randint(0, m - 1)
            e_r = polish_edges[idx_r]
            while True:
                e_a = random.choice(all_edges)
                if e_a not in polish_set:
                    break
            
            new_pe = list(polish_edges)
            new_pe[idx_r] = e_a
            new_ps = get_score(new_pe)
            
            if new_ps < polish_score:
                polish_edges = new_pe
                polish_set.discard(e_r)
                polish_set.add(e_a)
                polish_score = new_ps
                polish_no_improve = 0
                if polish_score < best_score:
                    best_score = polish_score
                    best_edges = list(polish_edges)
                    best_edge_set = set(polish_edges)
            elif new_ps == polish_score:
                # Accept equal to wander
                if random.random() < 0.3:
                    polish_edges = new_pe
                    polish_set.discard(e_r)
                    polish_set.add(e_a)
                    polish_no_improve += 1
            else:
                polish_no_improve += 1
    
    # Ensure types are correct: list of ints, list of lists of ints
    return {
        'values': [int(v) for v in values],
        'edges': [[int(u), int(v)] for u, v in best_edges]
    }
# EVOLVE-BLOCK-END
