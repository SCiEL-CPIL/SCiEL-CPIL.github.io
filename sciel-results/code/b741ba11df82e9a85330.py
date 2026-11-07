import random
import itertools
import time
import heapq

def construct():
    """
    Constructs a graph with 14 vertices and 35 edges to minimize the max of distinct sums and products.
    Uses priority queue based greedy edge selection followed by local search refinement.
    """
    n = 14
    m = 35
    
    # Candidate value sets
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    candidates.append(list(range(10, n + 10)))
    
    # 2. Multiples
    candidates.append([2 * i for i in range(1, n + 1)])
    candidates.append([3 * i for i in range(1, n + 1)])
    candidates.append([4 * i for i in range(1, n + 1)])
    candidates.append([6 * i for i in range(1, n + 1)])
    
    # 3. Structured set with many factor collisions
    candidates.append([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48])
    
    # 4. Random sets with bias towards small numbers
    random.seed(42)
    for _ in range(20):
        vals = random.sample(range(1, 51), n)
        candidates.append(vals)
        
    # 5. Additional structured sets: powers of 2 and 3 mixed
    p23 = [2**i * 3**j for i in range(5) for j in range(5)]
    p23 = sorted(set(p23))[:n]
    if len(p23) == n:
        candidates.append(p23)

    best_score = float('inf')
    best_edges = None
    best_values = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_select_edges_with_heap(values):
        """
        Selects edges using a greedy approach minimizing marginal cost.
        """
        all_edges = list(itertools.combinations(range(n), 2))
        
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining_edges = set(all_edges)
        
        # Precompute sums and products for all edges
        edge_props = {}
        for i, j in all_edges:
            edge_props[(i, j)] = (values[i] + values[j], values[i] * values[j])
            
        for _ in range(m):
            best_edge = None
            best_cost = float('inf')
            
            for edge in remaining_edges:
                i, j = edge
                s, p = edge_props[edge]
                
                new_sums_count = len(current_sums) + (1 if s not in current_sums else 0)
                new_prods_count = len(current_prods) + (1 if p not in current_prods else 0)
                
                cost = max(new_sums_count, new_prods_count)
                
                if cost < best_cost:
                    best_cost = cost
                    best_edge = edge
            
            if best_edge is None:
                break
                
            i, j = best_edge
            s, p = edge_props[best_edge]
            current_sums.add(s)
            current_prods.add(p)
            selected_edges.append(best_edge)
            remaining_edges.remove(best_edge)
            
        return selected_edges, best_cost

    def local_search(values, initial_edges, time_limit=2.0):
        """
        Refines the edge set using simulated annealing.
        """
        current_edges = set(initial_edges)
        current_score = evaluate(values, current_edges)
        best_edges = list(current_edges)
        best_score = current_score
        
        all_edges = list(itertools.combinations(range(n), 2))
        
        start_time = time.time()
        temperature = 10.0
        cooling = 0.995
        
        while time.time() - start_time < time_limit:
            # Pick a random edge to remove
            edge_out = random.choice(list(current_edges))
            
            # Pick a random edge to add (not currently in graph)
            while True:
                edge_in = tuple(sorted(random.sample(range(n), 2)))
                if edge_in not in current_edges:
                    break
            
            # Calculate new score
            new_edges = set(current_edges)
            new_edges.remove(edge_out)
            new_edges.add(edge_in)
            
            new_score = evaluate(values, new_edges)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score or random.random() < pow(2.71828, (current_score - new_score) / temperature):
                current_edges = new_edges
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
            
            temperature *= cooling
            
        return best_edges, best_score

    for values in candidates:
        if len(set(values)) != n or max(values) > 10000 or min(values) < 1:
            continue
            
        # 1. Greedy Selection
        greedy_edges, greedy_score = greedy_select_edges_with_heap(values)
        
        # 2. Local Search Refinement
        ls_edges, ls_score = local_search(values, greedy_edges, time_limit=1.0)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
            best_values = values
            
        if best_score <= 12: # Stop early if we find a very good solution
            break

    if best_edges is None:
        best_values = list(range(1, n + 1))
        all_edges = list(itertools.combinations(range(n), 2))
        best_edges = all_edges[:m]

    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
