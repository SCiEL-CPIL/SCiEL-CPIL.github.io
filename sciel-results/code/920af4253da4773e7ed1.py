"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Construct 14 distinct positive integers <= 10000 and exactly 35 edges
    to minimize max(number of distinct edge sums, number of distinct edge products).
    
    Strategy:
    1. Use structured initial values (arithmetic progression) to encourage sum collisions.
    2. Initialize edges by selecting those with the smallest sums to minimize distinct sums.
    3. Perform local search to minimize the maximum of distinct sums and products.
    """
    
    n = 14
    m = 35
    max_val = 10000
    
    def score(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))

    def get_initial_solution():
        # Values: 1 to 14
        vals = list(range(1, n + 1))
        
        # Calculate all edges with their sums
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                s = vals[i] + vals[j]
                all_edges.append((s, i, j))
        
        # Sort by sum to pick edges with smallest sums first
        all_edges.sort(key=lambda x: x[0])
        
        # Pick first 35 edges
        edges = []
        for _, i, j in all_edges[:m]:
            edges.append([i, j])
            
        return vals, edges

    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Run multiple restarts with slight variations
    for restart in range(20):
        if restart == 0:
            values, edges = get_initial_solution()
        else:
            # Perturb the initial solution
            values, edges = get_initial_solution()
            # Randomly swap some values
            for _ in range(5):
                i, j = random.sample(range(n), 2)
                values[i], values[j] = values[j], values[i]
            # Randomly swap some edges
            all_possible = list(combinations(range(n), 2))
            edge_set = set(tuple(sorted(e)) for e in edges)
            available = [e for e in all_possible if tuple(sorted(e)) not in edge_set]
            if available:
                for _ in range(5):
                    if not edges: break
                    rem_e = random.choice(edges)
                    edges.remove(rem_e)
                    add_e = random.choice(available)
                    edges.append(list(add_e))
                    available.remove(add_e)
        
        current_score = score(values, edges)
        
        # Local search
        for iteration in range(2000):
            move_type = random.choice(['edge', 'swap_edge', 'swap_val'])
            
            new_values = values[:]
            new_edges = edges[:]
            
            if move_type == 'swap_val':
                i, j = random.sample(range(n), 2)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                
            elif move_type == 'edge':
                existing_set = set(tuple(sorted(e)) for e in new_edges)
                all_edges_list = list(combinations(range(n), 2))
                available = [e for e in all_edges_list if tuple(sorted(e)) not in existing_set]
                if not available:
                    continue
                new_e = random.choice(available)
                rem_e = random.choice(new_edges)
                new_edges.remove(rem_e)
                new_edges.append(list(new_e))
                
            elif move_type == 'swap_edge':
                if len(new_edges) < 2:
                    continue
                e1, e2 = random.sample(new_edges, 2)
                existing_set = set(tuple(sorted(e)) for e in new_edges)
                all_edges_list = list(combinations(range(n), 2))
                available = [e for e in all_edges_list if tuple(sorted(e)) not in existing_set]
                if len(available) < 2:
                    continue
                new_e1, new_e2 = random.sample(available, 2)
                new_edges.remove(e1)
                new_edges.remove(e2)
                new_edges.append(list(new_e1))
                new_edges.append(list(new_e2))
            
            new_score = score(new_values, new_edges)
            
            if new_score < current_score:
                values = new_values
                edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = values[:]
                    best_edges = edges[:]
            # Early stop if we hit a very low score? Min possible is roughly 11-15?
            if best_score <= 15:
                break
        
        if best_score <= 15:
            break

    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END
