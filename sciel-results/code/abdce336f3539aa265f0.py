"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _all_pairs(n):
    return list(combinations(range(n), 2))

def _edge_sums_prods(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return len(sums), len(prods)

def _greedy_edge_select(values, target_k):
    """Greedy: pick edges that reuse existing sum/prod labels as much as possible."""
    n = len(values)
    all_edges = _all_pairs(n)
    sums = set()
    prods = set()
    chosen = []
    # Sort edges by some heuristic: prefer edges whose sum/prod are "common"
    # We'll iterate and greedily add edges that minimize increase in distinct counts
    remaining = list(all_edges)
    random.shuffle(remaining)
    
    for e in remaining:
        if len(chosen) >= P['m']:
            break
        i, j = e
        s = values[i] + values[j]
        p = values[i] * values[j]
        # Compute how many new distinct values this edge would add
        new_s = 1 if s not in sums else 0
        new_p = 1 if p not in prods else 0
        # Add it (greedy doesn't look ahead)
        sums.add(s)
        prods.add(p)
        chosen.append(e)
    
    # If we didn't get enough (shouldn't happen since we have all 91 edges), fill
    if len(chosen) < P['m']:
        for e in all_edges:
            if e not in chosen:
                chosen.append(e)
            if len(chosen) >= P['m']:
                break
    
    return chosen

def _local_search_edges(values, edges, iters=2000):
    """Local search on edge set: swap edges to reduce max(distinct_sums, distinct_prods)."""
    n = len(values)
    all_edges = _all_pairs(n)
    edge_set = set(edges)
    best_edges = list(edges)
    best_score = _raw_value(values, best_edges)
    
    for _ in range(iters):
        cur_sums, cur_prods = _edge_sums_prods(values, best_edges)
        cur_score = max(len(cur_sums), len(cur_prods))
        
        # Try swapping one edge for another
        in_edge = random.choice(best_edges)
        # Pick an edge not in the current set
        candidates = [e for e in all_edges if e not in edge_set]
        if not candidates:
            break
        out_edge = random.choice(candidates)
        
        new_edges = [e for e in best_edges if e != in_edge]
        new_edges.append(out_edge)
        new_score = _raw_value(values, new_edges)
        
        if new_score < cur_score:
            best_edges = new_edges
            best_score = new_score
            edge_set = set(best_edges)
        # With some probability, accept equal or slightly worse for diversity
        elif new_score == cur_score and random.random() < 0.1:
            best_edges = new_edges
            edge_set = set(best_edges)
    
    return best_edges, best_score

def _local_search_values(values, edges, iters=5000):
    """Local search on values: perturb individual values to reduce objective."""
    n = len(values)
    best_values = list(values)
    best_score = _raw_value(best_values, edges)
    
    for _ in range(iters):
        idx = random.randrange(n)
        old_val = best_values[idx]
        # Random new value
        new_val = random.randint(1, P['maximum'])
        if new_val == old_val:
            continue
        # Check distinctness
        if new_val in best_values:
            continue
        best_values[idx] = new_val
        new_score = _raw_value(best_values, edges)
        if new_score < best_score:
            best_score = new_score
        else:
            best_values[idx] = old_val
    
    return best_values, best_score

def _anneal(values, edges, iters=20000, T_start=10.0, T_end=0.1):
    """Simulated annealing on both values and edges."""
    n = len(values)
    all_edges = _all_pairs(n)
    cur_values = list(values)
    cur_edges = list(edges)
    cur_score = _raw_value(cur_values, cur_edges)
    best_values = list(cur_values)
    best_edges = list(cur_edges)
    best_score = cur_score
    edge_set = set(cur_edges)
    
    for it in range(iters):
        T = T_start * (T_end / T_start) ** (it / max(1, iters - 1))
        
        if random.random() < 0.5:
            # Perturb a value
            idx = random.randrange(n)
            old_val = cur_values[idx]
            new_val = random.randint(1, P['maximum'])
            if new_val == old_val or new_val in cur_values:
                continue
            cur_values[idx] = new_val
            new_score = _raw_value(cur_values, cur_edges)
            delta = new_score - cur_score
            if delta < 0 or random.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(cur_values)
                    best_edges = list(cur_edges)
            else:
                cur_values[idx] = old_val
        else:
            # Swap an edge
            in_edge = random.choice(cur_edges)
            candidates = [e for e in all_edges if e not in edge_set]
            if not candidates:
                continue
            out_edge = random.choice(candidates)
            new_edges = [e for e in cur_edges if e != in_edge]
            new_edges.append(out_edge)
            new_score = _raw_value(cur_values, new_edges)
            delta = new_score - cur_score
            if delta < 0 or random.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                cur_edges = new_edges
                cur_score = new_score
                edge_set = set(cur_edges)
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(cur_values)
                    best_edges = list(cur_edges)
    
    return best_values, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    
    # Start with a reasonable initial configuration
    # Use values that are spread out but allow some sum/product collisions
    # Initial: try geometric-ish progression
    best_values = None
    best_edges = None
    best_score = m  # worst case
    
    # Try several random initializations
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Generate random distinct values
        vals = random.sample(range(1, P['maximum'] + 1), n)
        random.shuffle(vals)
        
        # Greedy edge selection
        edges = _greedy_edge_select(vals, m)
        score = _raw_value(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
        
        # Local search on edges
        if time.time() > deadline:
            break
        edges_ls, score_ls = _local_search_edges(vals, edges, iters=500)
        if score_ls < best_score:
            best_score = score_ls
            best_values = vals
            best_edges = edges_ls
        
        # Local search on values
        if time.time() > deadline:
            break
        vals_ls, score_vls = _local_search_values(vals, edges_ls, iters=1000)
        if score_vls < best_score:
            best_score = score_vls
            best_values = vals_ls
            best_edges = edges_ls
    
    # Final annealing phase with remaining time
    if best_values is not None and time.time() < deadline - 5.0:
        remaining_iters = int((deadline - time.time() - 2.0) * 500)  # ~500 iters/sec estimate
        remaining_iters = max(remaining_iters, 1000)
        best_values, best_edges, best_score = _anneal(
            best_values, best_edges, 
            iters=remaining_iters,
            T_start=5.0, T_end=0.05
        )
    
    # Ensure valid output
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    
    # Convert to proper types
    result_values = [int(v) for v in best_values]
    result_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': result_values, 'edges': result_edges}
# EVOLVE-BLOCK-END
