import time
import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _optimize_edges_sa(values, m, n, time_budget=2.0, seed=42):
    """
    Simulated annealing to find m edges minimizing max(distinct sums, distinct prods).
    Uses fast incremental scoring via reference counting.
    """
    random.seed(seed)
    all_edges = list(combinations(range(n), 2))
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        u, v = e
        edge_sums[e] = values[u] + values[v]
        edge_prods[e] = values[u] * values[v]
    
    # Start with a random subset
    edges = set(random.sample(all_edges, m))
    
    # Build reference counts
    sum_counts = {}
    prod_counts = {}
    for e in edges:
        s = edge_sums[e]
        p = edge_prods[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    def current_score():
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        return max(distinct_sums, distinct_prods)
    
    current_score_val = current_score()
    best_score = current_score_val
    best_edges = set(edges)
    
    start_time = time.time()
    deadline = start_time + time_budget
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random edge to remove
        edge_to_remove = random.choice(list(edges))
        
        # Pick random edge to add (not in current set)
        new_edge = random.choice(all_edges)
        while new_edge in edges:
            new_edge = random.choice(all_edges)
        
        # Compute delta score
        s_rem = edge_sums[edge_to_remove]
        p_rem = edge_prods[edge_to_remove]
        s_add = edge_sums[new_edge]
        p_add = edge_prods[new_edge]
        
        # Remove edge_to_remove
        sum_counts[s_rem] -= 1
        if sum_counts[s_rem] == 0:
            del sum_counts[s_rem]
        prod_counts[p_rem] -= 1
        if prod_counts[p_rem] == 0:
            del prod_counts[p_rem]
        
        # Add new_edge
        sum_counts[s_add] = sum_counts.get(s_add, 0) + 1
        prod_counts[p_add] = prod_counts.get(p_add, 0) + 1
        
        new_score = current_score()
        
        # Accept or reject
        if new_score <= current_score_val or random.random() < pow(2.718281828, -(new_score - current_score_val) / max(T, 1e-9)):
            # Accept
            edges.discard(edge_to_remove)
            edges.add(new_edge)
            current_score_val = new_score
            if current_score_val < best_score:
                best_score = current_score_val
                best_edges = set(edges)
        else:
            # Reject - revert
            sum_counts[s_add] -= 1
            if sum_counts[s_add] == 0:
                del sum_counts[s_add]
            prod_counts[p_add] -= 1
            if prod_counts[p_add] == 0:
                del prod_counts[p_add]
            
            sum_counts[s_rem] = sum_counts.get(s_rem, 0) + 1
            prod_counts[p_rem] = prod_counts.get(p_rem, 0) + 1
        
        T *= cooling
    
    return best_score, [list(e) for e in best_edges]

def _optimize_values_edges_joint(values, m, n, time_budget=5.0, seed=42):
    """
    Jointly optimize values and edges using simulated annealing on both.
    """
    random.seed(seed)
    n = len(values)
    values = list(values)
    
    all_edges = list(combinations(range(n), 2))
    
    # Initial random edges
    edges = set(random.sample(all_edges, m))
    
    def compute_score(vals, edge_set):
        sums = set()
        prods = set()
        for u, v in edge_set:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))
    
    current_score = compute_score(values, edges)
    best_score = current_score
    best_values = list(values)
    best_edges = set(edges)
    
    start_time = time.time()
    deadline = start_time + time_budget
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        # Choose move type: 60% edge swap, 40% value mutation
        if random.random() < 0.6:
            # Edge swap
            edge_to_remove = random.choice(list(edges))
            new_edge = random.choice(all_edges)
            while new_edge in edges:
                new_edge = random.choice(all_edges)
            
            new_edges = edges.copy()
            new_edges.discard(edge_to_remove)
            new_edges.add(new_edge)
            
            new_score = compute_score(values, new_edges)
            
            if new_score <= current_score or random.random() < pow(2.718281828, -(new_score - current_score) / max(T, 1e-9)):
                edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(values)
                    best_edges = set(edges)
        else:
            # Value mutation: change one value
            idx = random.randint(0, n - 1)
            old_val = values[idx]
            
            # Try a few candidate new values
            candidates = []
            for _ in range(5):
                if random.random() < 0.5:
                    new_val = random.randint(1, 100)
                else:
                    new_val = random.randint(1, 500)
                if new_val != old_val:
                    candidates.append(new_val)
            
            if not candidates:
                continue
                
            new_val = random.choice(candidates)
            values[idx] = new_val
            
            # Check validity: all distinct positive <= 10000
            if len(set(values)) != n or any(v < 1 or v > 10000 for v in values):
                values[idx] = old_val
                continue
            
            new_score = compute_score(values, edges)
            
            if new_score <= current_score or random.random() < pow(2.718281828, -(new_score - current_score) / max(T, 1e-9)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(values)
                    best_edges = set(edges)
            else:
                values[idx] = old_val
        
        T *= cooling
    
    return best_score, best_values, [list(e) for e in best_edges]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 14
    m = 35
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Phase 1: Try structured value sets with SA edge optimization
    candidate_sets = [
        list(range(1, 15)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2 ** i for i in range(n)],
        [2 * i for i in range(1, 15)],
        [3 * i for i in range(1, 15)],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196],
    ]
    
    # Add some random small value sets
    random.seed(42)
    for _ in range(8):
        vals = []
        while len(vals) < n:
            if random.random() < 0.8:
                v = random.randint(1, 40)
            else:
                v = random.randint(1, 100)
            if v not in vals:
                vals.append(v)
        candidate_sets.append(vals)
    
    for i, vals in enumerate(candidate_sets):
        if time.time() > deadline - 10:
            break
        
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        t_budget = min(3.0, remaining * 0.8)
        if t_budget < 0.5:
            break
        
        # Try multiple seeds for diversity
        for seed in [42, 123, 789]:
            if time.time() > deadline - 5:
                break
            score, edges = _optimize_edges_sa(vals, m, n, time_budget=max(0.5, t_budget/3), seed=seed)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_values = list(vals)
                best_overall_edges = edges
    
    # Phase 2: Joint value+edge optimization from best found so far
    if best_overall_values is not None and time.time() < deadline - 15:
        for seed in [42, 123, 789, 1000, 2000]:
            if time.time() > deadline - 5:
                break
            t_budget = min(4.0, (deadline - time.time()) * 0.7)
            if t_budget < 1.0:
                break
            score, vals2, edges2 = _optimize_values_edges_joint(best_overall_values, m, n, time_budget=t_budget, seed=seed)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_values = list(vals2)
                best_overall_edges = edges2
    
    # Phase 3: If time remains, try more random value sets with joint optimization
    if best_overall_values is not None and time.time() < deadline - 10:
        random.seed(999)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            vals = []
            while len(vals) < n:
                r = random.random()
                if r < 0.4:
                    v = random.randint(1, 15)
                elif r < 0.7:
                    v = random.randint(1, 30)
                else:
                    v = random.randint(1, 60)
                if v not in vals:
                    vals.append(v)
            if len(set(vals)) != n:
                continue
            t_budget = min(3.0, (deadline - time.time()) * 0.5)
            if t_budget < 0.5:
                break
            score, vals2, edges2 = _optimize_values_edges_joint(vals, m, n, time_budget=t_budget, seed=random.randint(0, 10000))
            if score < best_overall_score:
                best_overall_score = score
                best_overall_values = list(vals2)
                best_overall_edges = edges2
    
    # Fallback
    if best_overall_values is None:
        best_overall_values = list(range(1, 15))
        best_overall_edges = [list(e) for e in combinations(range(n), 2)][:m]
        best_overall_score = _score(best_overall_values, best_overall_edges)
    
    # Ensure types are correct
    result_values = [int(v) for v in best_overall_values]
    result_edges = [list(map(int, e)) for e in best_overall_edges]
    
    return {'values': result_values, 'edges': result_edges}
