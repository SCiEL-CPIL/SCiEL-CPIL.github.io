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

def _optimize_edges(values, m, n, max_iter=20000):
    """
    Find a subset of m edges that minimizes the score for given values.
    Uses local search (swap moves).
    """
    all_edges = list(combinations(range(n), 2))
    # Start with a random subset
    edges = set(random.sample(all_edges, m))
    
    def get_score(edge_set):
        sums = set()
        prods = set()
        for u, v in edge_set:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    current_score = get_score(edges)
    best_score = current_score
    best_edges = set(edges)
    
    # Precompute sums and products for all possible edges to speed up evaluation?
    # With n=14, there are 91 edges.
    # sum_val[e], prod_val[e]
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        u, v = e
        edge_sums[e] = values[u] + values[v]
        edge_prods[e] = values[u] * values[v]
        
    def get_score_fast(edge_set):
        sums = set()
        prods = set()
        for e in edge_set:
            sums.add(edge_sums[e])
            prods.add(edge_prods[e])
        return max(len(sums), len(prods))

    current_score = get_score_fast(edges)
    best_score = current_score
    best_edges = set(edges)
    
    for i in range(max_iter):
        # Randomly select an edge to remove
        edge_to_remove = random.choice(list(edges))
        
        # Randomly select an edge to add
        # We try a few random candidates to find one not in the set
        added = False
        for _ in range(5):
            new_edge = random.choice(all_edges)
            if new_edge not in edges:
                # Perform swap
                new_edges = edges.copy()
                new_edges.remove(edge_to_remove)
                new_edges.add(new_edge)
                
                new_score = get_score_fast(new_edges)
                
                if new_score <= current_score:
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(edges)
                added = True
                break
        
        if not added:
            continue
            
    return best_score, [list(e) for e in best_edges]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Candidate value sets
    # 1. Small integers (many sum collisions, some product collisions)
    cand1 = list(range(1, 15))
    # 2. Highly composite numbers (many product collisions)
    cand2 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 3. Powers of 2 (baseline, unique sums/prods)
    cand3 = [2 ** i for i in range(n)]
    # 4. Even numbers (sums even, products multiples of 4)
    cand4 = [2 * i for i in range(1, 15)]
    # 5. Random small numbers
    random.seed(42)
    candidates = [cand1, cand2, cand3, cand4]
    
    for _ in range(10):
        vals = []
        while len(vals) < n:
            # Bias towards small numbers to encourage collisions
            if random.random() < 0.9:
                v = random.randint(1, 50)
            else:
                v = random.randint(1, 100)
            if v not in vals:
                vals.append(v)
        candidates.append(vals)
        
    for vals in candidates:
        if time.time() > deadline - 5:
            break
            
        # Validate
        if len(set(vals)) != n or any(v < 1 or v > max_val for v in vals):
            continue
            
        # Optimize edges
        # Use more iterations for sets that seem promising (small max value)
        max_iter = 5000 if max(vals) < 100 else 2000
        score, edges = _optimize_edges(vals, m, n, max_iter)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vals
            best_overall_edges = edges
            
    # Fallback
    if best_overall_values is None:
        best_overall_values = [2 ** i for i in range(n)]
        best_overall_edges = [list(e) for e in combinations(range(n), 2)][:m]
        best_overall_score = _score(best_overall_values, best_overall_edges)
        
    # Ensure types are correct
    result_values = [int(v) for v in best_overall_values]
    result_edges = [list(map(int, e)) for e in best_overall_edges]
    
    return {'values': result_values, 'edges': result_edges}
