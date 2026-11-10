"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_pairs = list(combinations(range(n), 2))
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def _optimize_edges(values, initial_edges, time_limit):
        # initial_edges is a list of [i, j]
        best_edges = [list(e) for e in initial_edges]
        best_score = _raw_value(values, best_edges)
        
        # Precompute pair indices for fast lookup
        edge_set = set(map(tuple, best_edges))
        
        # Local search loop
        no_improve_count = 0
        while time.time() < time_limit and no_improve_count < 1000:
            # Pick a random edge to remove
            edge_to_remove = random.choice(best_edges)
            i, j = edge_to_remove
            
            # Pick a random non-edge to add
            # To speed up, pick random pair until we find a non-edge
            while True:
                new_i, new_j = random.choice(all_pairs)
                if (new_i, new_j) not in edge_set and (new_j, new_i) not in edge_set:
                    break
            
            # Tentative swap
            best_edges.remove(edge_to_remove)
            edge_set.remove(tuple(edge_to_remove))
            
            new_edge = [new_i, new_j]
            best_edges.append(new_edge)
            edge_set.add(tuple(new_edge))
            
            new_score = _raw_value(values, best_edges)
            
            if new_score < best_score:
                best_score = new_score
                no_improve_count = 0
            else:
                # Revert
                best_edges.remove(new_edge)
                edge_set.remove(tuple(new_edge))
                
                best_edges.append(edge_to_remove)
                edge_set.add(tuple(edge_to_remove))
                no_improve_count += 1
                
        return best_edges, best_score

    best_result = None
    best_overall_score = float('inf')
    
    # Candidate value sets
    candidate_values = [
        [2**i for i in range(n)],       # Powers of 2
        list(range(1, n+1)),             # Small integers
        [i*2 for i in range(1, n+1)],   # Even small integers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27], # Odds
    ]
    
    # Add some random sets
    random.seed(42)
    for _ in range(5):
        vals = random.sample(range(1, 10001), n)
        candidate_values.append(sorted(vals))
        
    # Initial edges: first 35 pairs
    initial_edges = [list(e) for e in all_pairs[:m]]
    
    for vals in candidate_values:
        # Ensure distinct and positive
        if len(set(vals)) != n or any(v <= 0 or v > maximum for v in vals):
            continue
            
        # Run local search for this value set
        # Use a portion of the time for each set
        time_per_set = (deadline - time.time()) / len(candidate_values)
        if time_per_set < 1.0:
            time_per_set = 1.0
            
        edges, score = _optimize_edges(vals, initial_edges, time.time() + time_per_set)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': vals, 'edges': edges}
            
        if time.time() > deadline - 5.0:
            break

    if best_result is None:
        # Fallback
        vals = list(range(1, n+1))
        edges = [list(e) for e in all_pairs[:m]]
        best_result = {'values': vals, 'edges': edges}
        
    # Final check and type conversion
    result = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(i), int(j)] for i, j in best_result['edges']]
    }
    
    return result

# EVOLVE-BLOCK-END
