import time
import random
from itertools import combinations
from collections import defaultdict

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    random.seed(42)
    
    n = 14
    target_edges = 35
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # State: values (list of 14 ints), edge_set (set of 35 edge tuples)
    # We start with a decent heuristic initialization
    
    def init_state():
        # Generate values with some structure to encourage collisions
        # Try a mix of small numbers and multiples
        strategies = [
            lambda: list(range(1, 15)),
            lambda: [i * 2 for i in range(1, 15)],
            lambda: [i * 3 for i in range(1, 15)],
            lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (bad for sums/prods diversity? Actually sums are unique, prods unique. Bad start.)
            lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            lambda: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Invalid (distinct required), but good for SA to fix? No, constraints must be met.
        ]
        
        # Pick a random valid starting set
        # Let's use a randomized small set
        base = list(range(1, 50))
        random.shuffle(base)
        values = base[:14]
        
        # Ensure distinct
        while len(set(values)) < 14:
            values[random.randint(0, 13)] = random.randint(1, 10000)
        
        # Initial edges: Pick 35 random edges
        edges = random.sample(all_edges, target_edges)
        edge_set = set(edges)
        
        return values, edge_set

    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # We will run multiple SA restarts or a long single SA
    # Given 58s, we can do a long SA with cooling.
    
    # Precompute sums and products for fast updates?
    # Since values change, we must recompute sums/prods for affected edges.
    # An edge (u,v) is affected if values[u] or values[v] changes.
    
    def calculate_score(values, edge_set):
        sums = set()
        prods = set()
        for u, v in edge_set:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    # Initial state
    values, edge_set = init_state()
    current_score = calculate_score(values, edge_set)
    
    if current_score < best_score:
        best_score = current_score
        best_values = values[:]
        best_edges = list(edge_set)
        
    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    # We need to estimate steps. 
    # 58 seconds. Let's aim for ~100k-500k steps depending on speed.
    # Each step: generate neighbor, calc delta.
    
    # To speed up, we can maintain sums/prods counts?
    # No, distinct count is hard to update incrementally without storing counts.
    # Storing counts of each sum/prod in the current set allows O(1) update for one edge change.
    
    # Let's implement incremental update.
    # State:
    # values: list of 14
    # edge_set: set of 35
    # sum_counts: dict {sum_value: count}
    # prod_counts: dict {prod_value: count}
    # distinct_sums: len(sum_counts)
    # distinct_prods: len(prod_counts)
    # score: max(distinct_sums, distinct_prods)
    
    def build_state(values, edge_set):
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        for u, v in edge_set:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] += 1
            prod_counts[p] += 1
        return sum_counts, prod_counts
    
    sum_counts, prod_counts = build_state(values, edge_set)
    distinct_sums = len(sum_counts)
    distinct_prods = len(prod_counts)
    current_score = max(distinct_sums, distinct_prods)
    
    if current_score < best_score:
        best_score = current_score
        best_values = values[:]
        best_edges = list(edge_set)
        
    steps = 0
    while time.time() < deadline and T > T_min:
        # Generate Neighbor
        # 50% chance: change an edge
        # 50% chance: change a value
        
        if random.random() < 0.5:
            # Change Edge
            # Pick a random edge in set and a random edge not in set
            edges_list = list(edge_set)
            e_in = edges_list[random.randint(0, len(edges_list)-1)]
            
            # Find an e_out
            # Efficient way: pick random edge, if in set, retry (unlikely to loop long with 91 total edges, 35 selected)
            e_out = None
            for _ in range(10):
                cand = random.choice(all_edges)
                if cand not in edge_set:
                    e_out = cand
                    break
            if e_out is None:
                continue
            
            u_in, v_in = e_in
            u_out, v_out = e_out
            
            s_in = values[u_in] + values[v_in]
            p_in = values[u_in] * values[v_in]
            s_out = values[u_out] + values[v_out]
            p_out = values[u_out] * values[v_out]
            
            # Update counts
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
                
            sum_counts[s_out] += 1
            prod_counts[p_out] += 1
            
            new_distinct_sums = len(sum_counts)
            new_distinct_prods = len(prod_counts)
            new_score = max(new_distinct_sums, new_distinct_prods)
            delta = new_score - current_score
            
            # Acceptance
            if delta < 0 or (T > T_min and random.random() < pow(2.71828, -delta / T)):
                # Accept
                edge_set.remove(e_in)
                edge_set.add(e_out)
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = values[:]
                    best_edges = list(edge_set)
            else:
                # Reject: revert counts
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                sum_counts[s_in] += 1
                prod_counts[p_in] += 1
                
        else:
            # Change Value
            idx = random.randint(0, n - 1)
            old_val = values[idx]
            # New value: random small int
            new_val = random.randint(1, 100)
            if new_val == old_val:
                continue
            
            # Check distinctness constraint
            if new_val in values:
                continue
                
            # Find all edges incident to idx
            affected_edges = [e for e in edge_set if idx in e]
            
            # Store old sums/prods and new sums/prods
            changes = []
            valid = True
            for u, v in affected_edges:
                other = v if u == idx else u
                s_old = old_val + values[other]
                p_old = old_val * values[other]
                s_new = new_val + values[other]
                p_new = new_val * values[other]
                changes.append((s_old, p_old, s_new, p_new))
            
            # Apply changes to counts
            for s_old, p_old, s_new, p_new in changes:
                sum_counts[s_old] -= 1
                if sum_counts[s_old] == 0:
                    del sum_counts[s_old]
                prod_counts[p_old] -= 1
                if prod_counts[p_old] == 0:
                    del prod_counts[p_old]
                
                sum_counts[s_new] += 1
                prod_counts[p_new] += 1
            
            new_distinct_sums = len(sum_counts)
            new_distinct_prods = len(prod_counts)
            new_score = max(new_distinct_sums, new_distinct_prods)
            delta = new_score - current_score
            
            # Acceptance
            if delta < 0 or (T > T_min and random.random() < pow(2.71828, -delta / T)):
                # Accept
                values[idx] = new_val
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = values[:]
                    best_edges = list(edge_set)
            else:
                # Reject: revert counts
                for s_old, p_old, s_new, p_new in changes:
                    sum_counts[s_new] -= 1
                    if sum_counts[s_new] == 0:
                        del sum_counts[s_new]
                    prod_counts[p_new] -= 1
                    if prod_counts[p_new] == 0:
                        del prod_counts[p_new]
                    
                    sum_counts[s_old] += 1
                    prod_counts[p_old] += 1
        
        T *= cooling_rate
        steps += 1
        
        # Periodic restart if stuck? 
        # If no improvement for 5000 steps, maybe restart with current best?
        # For now, simple cooling.
        
    # Final check
    if best_values is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {'values': values, 'edges': edges}
        
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [list(e) for e in best_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
