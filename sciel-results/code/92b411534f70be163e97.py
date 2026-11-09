import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    best_score = float('inf')
    best_result = None
    
    # Generate candidate value sets focused on small numbers for more collisions
    candidates = []
    
    # Small integers with rich factorization - best for product collisions
    small_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 20],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
        [1, 2, 4, 8, 16, 3, 6, 12, 24, 9, 18, 36, 72, 5],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36],
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21],
        [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24, 30],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24],
    ]
    candidates.extend(small_sets)
    
    # Random small sets
    for _ in range(40):
        pool = list(range(1, 50))
        vals = random.sample(pool, n)
        candidates.append(sorted(vals))
    
    # Random sets with duplicates avoided but small range
    for _ in range(30):
        pool = list(range(1, 30))
        vals = random.sample(pool, n)
        candidates.append(sorted(vals))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def optimize_edges_sa(values, max_steps=800, restarts=8):
        """Simulated annealing for edge optimization with multiple restarts"""
        best_edges = None
        best_local_score = float('inf')
        
        for restart in range(restarts):
            if time.time() > deadline - 3:
                break
            
            current_edges = set(random.sample(all_edges, m))
            current_score = evaluate(values, current_edges)
            best_edges_restart = list(current_edges)
            best_score_restart = current_score
            
            temp = 3.0
            no_improve = 0
            steps = 0
            while steps < max_steps and time.time() < deadline - 3:
                # Pick random edge to remove
                e_remove = random.choice(list(current_edges))
                
                # Pick random edge to add (not in current)
                attempts = 0
                while attempts < 30:
                    e_add = random.choice(all_edges)
                    if e_add not in current_edges:
                        break
                    attempts += 1
                else:
                    continue
                
                # Compute new score
                new_edges = set(current_edges)
                new_edges.discard(e_remove)
                new_edges.add(e_add)
                new_score = evaluate(values, new_edges)
                
                delta = new_score - current_score
                
                if delta <= 0:
                    current_edges = new_edges
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score_restart:
                        best_score_restart = current_score
                        best_edges_restart = list(current_edges)
                else:
                    # Accept with probability
                    p = random.random()
                    if p < (1.0 / (1.0 + delta * temp)):
                        current_edges = new_edges
                        current_score = new_score
                        no_improve += 1
                    else:
                        no_improve += 1
                
                temp *= 0.99
                steps += 1
                
                if no_improve > 150:
                    break
            
            if best_score_restart < best_local_score:
                best_local_score = best_score_restart
                best_edges = best_edges_restart
        
        return best_edges, best_local_score
    
    # First pass: quick evaluation of all candidates to rank them
    quick_results = []
    for vals in candidates:
        if time.time() > deadline - 15:
            break
        for _ in range(5):
            edges = set(random.sample(all_edges, m))
            score = evaluate(vals, edges)
            quick_results.append((score, vals))
    
    # Sort by quick score and take top candidates for full optimization
    quick_results.sort(key=lambda x: x[0])
    top_candidates = []
    seen = set()
    for score, vals in quick_results:
        key = tuple(sorted(vals))
        if key not in seen:
            seen.add(key)
            top_candidates.append(vals)
        if len(top_candidates) >= 20:
            break
    
    # Full optimization on top candidates
    for vals in top_candidates:
        if time.time() > deadline - 5:
            break
        
        edges, local_score = optimize_edges_sa(vals, max_steps=500, restarts=6)
        
        if edges and local_score < best_score:
            best_score = local_score
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
        
        # If we found a very good score, do more optimization
        if best_score <= 25:
            edges2, local_score2 = optimize_edges_sa(vals, max_steps=800, restarts=10)
            if edges2 and local_score2 < best_score:
                best_score = local_score2
                best_result = {'values': list(vals), 'edges': [list(e) for e in edges2]}
    
    if best_result is None:
        # Fallback
        best_result = {'values': list(range(1, 15)), 'edges': [list(e) for e in all_edges[:35]]}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
