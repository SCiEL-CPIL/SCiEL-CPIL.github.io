"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

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
        
        # Precompute sum and product for each edge to speed up updates
        # We maintain counts of distinct sums and products
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        def update_counts(edges, add=True):
            for i, j in edges:
                s = values[i] + values[j]
                p = values[i] * values[j]
                if add:
                    sum_counts[s] += 1
                    prod_counts[p] += 1
                else:
                    sum_counts[s] -= 1
                    prod_counts[p] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
                    if prod_counts[p] == 0:
                        del prod_counts[p]

        # Initialize counts
        update_counts(best_edges, add=True)
        
        def get_score():
            return max(len(sum_counts), len(prod_counts))
        
        best_score = get_score()
        
        # Local search loop
        no_improve_count = 0
        # Use a fixed number of iterations or time-based
        # Since each step is O(1) amortized, we can do many steps
        max_iterations = 200000 
        
        while time.time() < time_limit and no_improve_count < 5000 and max_iterations > 0:
            max_iterations -= 1
            
            # Pick a random edge to remove
            idx_remove = random.randint(0, m - 1)
            edge_to_remove = best_edges[idx_remove]
            i, j = edge_to_remove
            
            # Pick a random non-edge to add
            # To speed up, pick random pair until we find a non-edge
            # We need a fast way to check if an edge exists.
            # A set of tuples is O(1).
            # But we modify best_edges. Let's maintain a set of edges.
            # Actually, rebuilding the set every time is O(m).
            # Let's just keep a set `edge_set` synchronized with `best_edges`.
            
            # We didn't initialize edge_set in this version. Let's do it properly.
            # Wait, I need to define edge_set outside the loop or initialize it.
            # I'll refactor to use a set for edges.
            
            # Re-initializing for correctness in this thought block, 
            # but in the final code I will implement it correctly.
            pass

    # Corrected _optimize_edges with edge_set
    def _optimize_edges_fast(values, initial_edges, time_limit):
        best_edges = [list(e) for e in initial_edges]
        edge_set = set(map(tuple, best_edges))
        
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        def add_edge(edge):
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] += 1
            prod_counts[p] += 1
            edge_set.add(tuple(edge))
            best_edges.append(edge)
            
        def remove_edge(edge):
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] -= 1
            prod_counts[p] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            if prod_counts[p] == 0:
                del prod_counts[p]
            edge_set.remove(tuple(edge))
            # Remove from list efficiently? 
            # Removing from a list is O(m). 
            # For m=35, it's fast enough.
            best_edges.remove(edge)

        # Initialize
        for e in best_edges:
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] += 1
            prod_counts[p] += 1

        def get_score():
            return max(len(sum_counts), len(prod_counts))
        
        best_score = get_score()
        
        no_improve_count = 0
        while time.time() < time_limit and no_improve_count < 2000:
            # Pick random edge to remove
            idx = random.randint(0, len(best_edges) - 1)
            e_remove = best_edges[idx]
            
            # Pick random non-edge to add
            while True:
                i, j = random.choice(all_pairs)
                if (i, j) not in edge_set and (j, i) not in edge_set:
                    break
            
            e_add = [i, j]
            
            # Tentative swap
            remove_edge(e_remove)
            add_edge(e_add)
            
            new_score = get_score()
            
            if new_score < best_score:
                best_score = new_score
                no_improve_count = 0
            else:
                # Revert
                remove_edge(e_add)
                add_edge(e_remove)
                no_improve_count += 1
                
        return best_edges, best_score

    best_result = None
    best_overall_score = float('inf')
    
    # Candidate value sets
    # Focus on compact sets to minimize sum range
    candidate_values = [
        list(range(1, n+1)),             # 1..14
        list(range(2, n+2)),             # 2..15
        list(range(3, n+3)),             # 3..16
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        # Smooth numbers / highly composite
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    ]
    
    # Add some random compact sets
    random.seed(42)
    for _ in range(5):
        # Generate 14 distinct integers in a small range, e.g., 1..30
        vals = random.sample(range(1, 31), n)
        candidate_values.append(sorted(vals))
        
    # Initial edges: 
    # For compact values, start with edges that have middle sums
    # Or just random
    initial_edges = [list(e) for e in random.sample(all_pairs, m)]
    
    for vals in candidate_values:
        if len(set(vals)) != n or any(v <= 0 or v > maximum for v in vals):
            continue
            
        time_per_set = (deadline - time.time()) / max(1, len(candidate_values))
        if time_per_set < 2.0:
            time_per_set = 2.0
            
        # Run local search
        edges, score = _optimize_edges_fast(vals, initial_edges, time.time() + time_per_set)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': vals, 'edges': edges}
            
        if time.time() > deadline - 5.0:
            break

    if best_result is None:
        vals = list(range(1, n+1))
        edges = [list(e) for e in all_pairs[:m]]
        best_result = {'values': vals, 'edges': edges}
        
    result = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(i), int(j)] for i, j in best_result['edges']]
    }
    
    return result

# EVOLVE-BLOCK-END
