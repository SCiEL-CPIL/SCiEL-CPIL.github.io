import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 14
    m = 35
    
    def evaluate_9_values(vals):
        """
        vals: list of 9 distinct integers.
        Returns the minimum possible max(distinct_sums, distinct_products)
        for a graph K9 minus one edge.
        """
        # Precompute all sums and products for K9
        all_sums = {}
        all_prods = {}
        edges = []
        for i in range(9):
            for j in range(i+1, 9):
                s = vals[i] + vals[j]
                p = vals[i] * vals[j]
                edges.append((i, j, s, p))
                all_sums.setdefault(s, []).append((i, j))
                all_prods.setdefault(p, []).append((i, j))
        
        total_sums = len(all_sums)
        total_prods = len(all_prods)
        
        best_score = float('inf')
        best_edge_to_remove = None
        
        # Try removing each edge
        # Optimization: Only consider edges where the sum or product is unique, 
        # or where removing it doesn't increase the count (impossible) 
        # Actually, we just need to find the min max(count_sums_after, count_prods_after)
        
        # If we remove edge e=(i,j) with sum s_e and prod p_e:
        # new_count_sums = total_sums - (1 if len(all_sums[s_e])==1 else 0)
        # new_count_prods = total_prods - (1 if len(all_prods[p_e])==1 else 0)
        
        for i, j, s, p in edges:
            ns = total_sums - (1 if len(all_sums[s]) == 1 else 0)
            np = total_prods - (1 if len(all_prods[p]) == 1 else 0)
            score = max(ns, np)
            if score < best_score:
                best_score = score
                best_edge_to_remove = (i, j)
                
        return best_score, best_edge_to_remove

    def full_score(vals9, edge_to_remove):
        """
        Calculate the actual score for the 14-vertex graph.
        """
        # Values 0-8 are vals9.
        # Edges are all pairs in 0-8 except edge_to_remove.
        sums = set()
        prods = set()
        for i in range(9):
            for j in range(i+1, 9):
                if (i, j) == edge_to_remove:
                    continue
                sums.add(vals9[i] + vals9[j])
                prods.add(vals9[i] * vals9[j])
        return max(len(sums), len(prods))

    best_score = float('inf')
    best_vals = None
    best_edge = None

    # 1. Try consecutive integers
    for start in range(1, 100):
        if time.time() > deadline - 5:
            break
        vals = list(range(start, start+9))
        score, edge = evaluate_9_values(vals)
        if score < best_score:
            best_score = score
            best_vals = vals[:]
            best_edge = edge

    # 2. Try APs
    for start in range(1, 50):
        for d in range(1, 20):
            if time.time() > deadline - 5:
                break
            vals = [start + i*d for i in range(9)]
            if max(vals) > 10000:
                continue
            score, edge = evaluate_9_values(vals)
            if score < best_score:
                best_score = score
                best_vals = vals[:]
                best_edge = edge

    # 3. Local search around best found
    if best_vals:
        current_vals = best_vals[:]
        current_score = best_score
        current_edge = best_edge
        
        # Try swapping elements
        random.seed(42)
        iterations = 0
        max_iterations = 20000
        while time.time() < deadline - 5 and iterations < max_iterations:
            iterations += 1
            new_vals = current_vals[:]
            idx = random.randint(0, 8)
            # Try small changes
            delta = random.randint(-5, 5)
            if delta == 0:
                continue
            new_val = current_vals[idx] + delta
            if new_val <= 0 or new_val > 10000:
                continue
            if new_val in set(current_vals) - {current_vals[idx]}:
                continue
            
            new_vals[idx] = new_val
            score, edge = evaluate_9_values(new_vals)
            
            if score < current_score:
                current_score = score
                current_vals = new_vals[:]
                current_edge = edge
                best_score = score
                best_vals = current_vals[:]
                best_edge = current_edge
            elif score == current_score and random.random() < 0.1:
                # Plateau move
                current_vals = new_vals[:]
                current_edge = edge

    # 4. Random search
    random.seed(123)
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        # Sample from 1..50
        vals = random.sample(range(1, 51), 9)
        score, edge = evaluate_9_values(vals)
        if score < best_score:
            best_score = score
            best_vals = vals[:]
            best_edge = edge

    # Fallback
    if best_vals is None:
        best_vals = list(range(1, 10))
        best_score, best_edge = evaluate_9_values(best_vals)

    # Construct final result
    final_values = [0] * 14
    for i in range(9):
        final_values[i] = best_vals[i]
    
    used = set(best_vals)
    # Fill remaining 5 with distinct integers
    candidate = 100
    for i in range(9, 14):
        while candidate in used:
            candidate += 1
        final_values[i] = candidate
        used.add(candidate)
        
    final_edges = []
    for i in range(9):
        for j in range(i+1, 9):
            if (i, j) == best_edge:
                continue
            final_edges.append([i, j])
            
    # Ensure types
    final_values = [int(v) for v in final_values]
    final_edges = [[int(i), int(j)] for i, j in final_edges]
    
    return {'values': final_values, 'edges': final_edges}
