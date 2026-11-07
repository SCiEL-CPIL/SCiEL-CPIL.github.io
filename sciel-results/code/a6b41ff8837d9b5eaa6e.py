"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _all_edges(n):
    return [list(e) for e in combinations(range(n), 2)]

def _precompute(values):
    n = len(values)
    edge_data = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_data[(i,j)] = (values[i] + values[j], values[i] * values[j])
    return edge_data

def _greedy_init_edges(edge_data, m):
    """Greedy: pick edges that share sums/products with already-picked edges."""
    all_edges = list(edge_data.keys())
    random.shuffle(all_edges)
    
    # First, count frequency of each sum and product across all edges
    sum_freq = Counter()
    prod_freq = Counter()
    for e in all_edges:
        s, p = edge_data[e]
        sum_freq[s] += 1
        prod_freq[p] += 1
    
    selected = []
    selected_sums = Counter()
    selected_prods = Counter()
    
    for e in all_edges:
        if len(selected) >= m:
            break
        s, p = edge_data[e]
        # Score: prefer edges whose sum/product are already common
        score = selected_sums[s] + selected_prods[p]
        selected.append((score, e))
    
    selected.sort(key=lambda x: -x[0])
    result = []
    for score, e in selected[:m]:
        result.append(e)
        s, p = edge_data[e]
        selected_sums[s] += 1
        selected_prods[p] += 1
    
    return result

def _optimize_edges_fast(values, time_budget):
    n = len(values)
    m = P['m']
    edge_data = _precompute(values)
    all_edges = list(edge_data.keys())
    
    # Greedy initialization
    edges = set(_greedy_init_edges(edge_data, m))
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in edges:
        s, p = edge_data[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    def get_score():
        return max(len(sum_counts), len(prod_counts))
    
    current_score = get_score()
    best_score = current_score
    best_edges = set(edges)
    
    deadline = time.time() + time_budget
    no_improve = 0
    
    while time.time() < deadline and no_improve < 500:
        e_in = random.choice(list(edges))
        e_out_candidates = [e for e in all_edges if e not in edges]
        if not e_out_candidates:
            break
        e_out = random.choice(e_out_candidates)
        
        s_in, p_in = edge_data[e_in]
        s_out, p_out = edge_data[e_out]
        
        sum_counts[s_in] -= 1
        if sum_counts[s_in] == 0:
            del sum_counts[s_in]
        sum_counts[s_out] += 1
        
        prod_counts[p_in] -= 1
        if prod_counts[p_in] == 0:
            del prod_counts[p_in]
        prod_counts[p_out] += 1
        
        new_score = get_score()
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(edges)
            no_improve = 0
        else:
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
            sum_counts[s_in] += 1
            
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
            prod_counts[p_in] += 1
            no_improve += 1
        
        edges.discard(e_in)
        edges.add(e_out)
    
    return list(best_edges), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_result = None
    best_score = float('inf')
    
    initial_sets = [
        list(range(1, P['n']+1)),
        [2*(i+1) for i in range(P['n'])],
        [2**i for i in range(P['n'])],
    ]
    for start in [1, 2, 3, 5]:
        for d in [1, 2, 3, 4, 5]:
            vals = [start + i*d for i in range(P['n'])]
            if max(vals) <= P['maximum'] and len(set(vals)) == P['n']:
                initial_sets.append(vals)
    for _ in range(8):
        initial_sets.append(sorted(random.sample(range(1, P['maximum']+1), P['n'])))
    
    random.shuffle(initial_sets)
    
    for vals in initial_sets:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 4.0:
            break
        
        alloc = min(12.0, remaining * 0.85)
        edges, score = _optimize_edges_fast(vals, alloc)
        
        if score < best_score:
            best_score = score
            best_result = (list(vals), edges)
    
    while time.time() < deadline - 4.0:
        vals = sorted(random.sample(range(1, P['maximum']+1), P['n']))
        remaining = deadline - time.time()
        if remaining < 4.0:
            break
        alloc = min(10.0, remaining * 0.85)
        edges, score = _optimize_edges_fast(vals, alloc)
        if score < best_score:
            best_score = score
            best_result = (list(vals), edges)
    
    if best_result is None:
        vals = list(range(1, P['n']+1))
        edges = _all_edges(P['n'])[:P['m']]
        best_result = (vals, edges)
    
    values, edges = best_result
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
