"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Candidate value sets designed to have high collision rates for sums and products
    # Set 1: Highly composite / small factors
    v1 = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
    # Set 2: Powers of 2 and multiples
    v2 = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    # Set 3: Small integers (good for sums)
    v3 = list(range(1, 15))
    # Set 4: Multiples of 6 (good for products)
    v4 = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
    # Set 5: Mix
    v5 = [2, 4, 8, 16, 32, 64, 12, 24, 48, 96, 192, 384, 768, 1536]

    best_score = float('inf')
    best_result = None

    candidate_values = [v1, v2, v3, v4, v5]
    
    # Shuffle candidates to ensure variety in early iterations
    random.shuffle(candidate_values)

    for values in candidate_values:
        if time.time() > deadline:
            break
            
        # Ensure values are distinct, positive, <= 10000
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        n = 14
        m = 35
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sums and products for all edges
        edge_sums = {}
        edge_products = {}
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_sums[(i, j)] = s
            edge_products[(i, j)] = p

        def get_score(edges):
            sums = set()
            prods = set()
            for e in edges:
                sums.add(edge_sums[e])
                prods.add(edge_products[e])
            return max(len(sums), len(prods))

        # Simulated Annealing for edge selection
        # Initialize with random edges
        current_edges = set(random.sample(all_edges, m))
        current_score = get_score(current_edges)
        
        best_edges = set(current_edges)
        best_local_score = current_score
        
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 0.01
        
        while temperature > min_temperature:
            if time.time() > deadline:
                break
                
            # Generate a neighbor: swap one edge in with one edge out
            in_edge = random.choice(list(current_edges))
            out_edge = random.choice([e for e in all_edges if e not in current_edges])
            
            new_edges = set(current_edges)
            new_edges.remove(in_edge)
            new_edges.add(out_edge)
            
            new_score = get_score(new_edges)
            
            # Acceptance probability
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_edges = set(current_edges)
            else:
                # Accept worse solutions with probability exp(-delta / T)
                # Since score is integer, delta is at least 1
                delta = new_score - current_score
                if delta == 0:
                    # Equal score, accept to explore
                    current_edges = new_edges
                elif random.random() < math.exp(-delta / temperature):
                    current_edges = new_edges
                    current_score = new_score
                    
            temperature *= cooling_rate
            
        # After SA, do a local search (hill climbing) to ensure local optimum
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for in_edge in all_edges:
                if in_edge in best_edges:
                    continue
                for out_edge in best_edges:
                    new_edges = set(best_edges)
                    new_edges.remove(out_edge)
                    new_edges.add(in_edge)
                    new_score = get_score(new_edges)
                    if new_score < best_local_score:
                        best_edges = new_edges
                        best_local_score = new_score
                        improved = True
                        break
                if improved:
                    break

        if best_local_score < best_score:
            best_score = best_local_score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [list(e) for e in best_edges]
            }
            
        # Early exit if we find a very good score
        if best_score <= 10:
            break

    if best_result is None:
        # Fallback
        values = v1
        all_edges = list(itertools.combinations(range(14), 2))
        edges = all_edges[:35]
        best_result = {
            'values': [int(v) for v in values],
            'edges': [list(e) for e in edges]
        }

    return best_result
# EVOLVE-BLOCK-END
