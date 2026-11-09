import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (vertices[i], vertices[j]) not in edge_set:
                    return False
        return True
        
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (vertices[i], vertices[j]) in edge_set:
                    return False
        return True

    # Find all maximal cliques and maximal independent sets
    # Using Bron-Kerbosch or simple enumeration for n=14
    all_vertices = list(range(n))
    maximal_cliques = []
    maximal_independent = []
    
    # Generate all subsets of size >= 2 (size 1 are trivially both, but we want maximal)
    # Actually, it's easier to find maximal cliques/ind sets directly.
    # For n=14, we can just check all subsets? 2^14 = 16384. Very fast.
    
    for mask in range(1, 1 << n):
        vertices = [i for i in range(n) if (mask >> i) & 1]
        if not vertices:
            continue
        # Check if maximal clique
        if is_clique(vertices):
            # Check if maximal: no vertex can be added
            is_maximal = True
            for v in all_vertices:
                if v not in vertices:
                    if is_clique(vertices + [v]):
                        is_maximal = False
                        break
            if is_maximal:
                maximal_cliques.append(frozenset(vertices))
                
        if is_independent(vertices):
            # Check if maximal: no vertex can be added
            is_maximal = True
            for v in all_vertices:
                if v not in vertices:
                    if is_independent(vertices + [v]):
                        is_maximal = False
                        break
            if is_maximal:
                maximal_independent.append(frozenset(vertices))

    # Combine and remove duplicates (a set can be both a maximal clique and maximal independent set? Only if size 1 or 2 with no edge? No, maximal clique and maximal independent set are usually distinct unless n is small. Size 1 sets are both. We should include them.)
    # Actually, size 1 sets are always maximal cliques and maximal independent sets?
    # If a vertex has no edges, {v} is a maximal independent set? No, if it has no edges, the whole graph is an independent set.
    # Let's just take the union of the two lists and remove duplicates.
    all_sets = list(set(maximal_cliques + maximal_independent))
    
    # Sort by size descending to find large sets first (greedy heuristic)
    all_sets.sort(key=lambda x: -len(x))
    
    # Greedy initial solution
    remaining = set(range(n))
    greedy_parts = []
    while remaining:
        best_set = None
        for s in all_sets:
            if s <= remaining:
                if best_set is None or len(s) > len(best_set):
                    best_set = s
        if best_set is None:
            # Should not happen, but if it does, pick any vertex
            best_set = frozenset([list(remaining)[0]])
        greedy_parts.append(list(best_set))
        remaining -= best_set
    best_solution = greedy_parts
    best_count = len(best_solution)
    
    # Backtracking search
    start_time = time.time()
    timeout = 50.0 # Leave some margin
    
    def search(remaining_set, current_parts, current_count):
        nonlocal best_solution, best_count, start_time, timeout
        
        if time.time() - start_time > timeout:
            return True # Timeout signal
            
        if not remaining_set:
            if current_count < best_count:
                best_count = current_count
                best_solution = [list(p) for p in current_parts]
            return False
            
        # Lower bound: 
        # The minimum number of parts needed is at least ceil(|remaining| / max_size)
        max_size = max(len(s) for s in all_sets if s <= remaining_set) if any(s <= remaining_set for s in all_sets) else 1
        if max_size == 0:
             max_size = 1
        lower_bound = (len(remaining_set) + max_size - 1) // max_size
        if current_count + lower_bound >= best_count:
            return False
            
        # Try each set
        for s in all_sets:
            if s <= remaining_set:
                new_remaining = remaining_set - s
                if search(new_remaining, current_parts + [s], current_count + 1):
                    return True
                    
    search(set(range(n)), [], 0)
    
    # Ensure types are correct
    final_parts = []
    for part in best_solution:
        final_parts.append([int(v) for v in part])
        
    return final_parts
# EVOLVE-BLOCK-END
