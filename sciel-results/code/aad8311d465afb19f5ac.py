"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _generate_structured_values():
    """Generate structured value sets that may have good sum/product properties."""
    strategy = random.random()
    if strategy < 0.3:
        # Small consecutive integers
        start = random.randint(1, 100)
        step = random.randint(1, 3)
        values = [start + i * step for i in range(14)]
        if max(values) > 10000:
            values = list(range(1, 15))
    elif strategy < 0.5:
        # Powers of 2
        base = random.randint(1, 3)
        values = [base ** i for i in range(14)]
        if max(values) > 10000:
            values = [2 ** i for i in range(14)]
            if max(values) > 10000:
                values = list(range(1, 15))
    elif strategy < 0.7:
        # Fibonacci-like
        a, b = random.randint(1, 5), random.randint(1, 5)
        values = [a, b]
        for _ in range(12):
            a, b = b, a + b
            values.append(b)
        if max(values) > 10000:
            values = list(range(1, 15))
    else:
        # Random sample
        values = random.sample(range(1, 10001), 14)
    
    # Ensure distinct and positive
    values = sorted(set(values))
    while len(values) < 14:
        new_val = random.randint(1, 10000)
        if new_val not in values:
            values.append(new_val)
    values = values[:14]
    return values

def _generate_edges(n, m):
    """Generate m random edges from n vertices."""
    all_edges = list(combinations(range(n), 2))
    edges = [list(e) for e in random.sample(all_edges, m)]
    return edges

def _local_search(values, edges, max_iters=5000, deadline=None):
    """Perform local search by mutating values and edges."""
    n = len(values)
    m = len(edges)
    current_score = _score(values, edges)
    best_values = values[:]
    best_edges = edges[:]
    best_score = current_score
    
    all_edges = list(combinations(range(n), 2))
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Randomly choose mutation type
        mutation_type = random.random()
        
        if mutation_type < 0.5:
            # Mutate a value
            idx = random.randint(0, n - 1)
            old_val = values[idx]
            new_val = random.randint(1, 10000)
            if new_val == old_val:
                continue
            if new_val in values and new_val != old_val:
                continue
            
            values[idx] = new_val
            new_score = _score(values, edges)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = values[:]
                    best_edges = edges[:]
            else:
                values[idx] = old_val
        else:
            # Mutate an edge: remove one, add another
            if m <= 1:
                continue
            
            # Remove a random edge
            remove_idx = random.randint(0, m - 1)
            removed_edge = tuple(sorted(edges[remove_idx]))
            edges.pop(remove_idx)
            
            # Add a random new edge
            available_edges = [e for e in all_edges if e not in edge_set]
            if not available_edges:
                edges.insert(remove_idx, list(removed_edge))
                continue
            
            new_edge = random.choice(available_edges)
            edges.append(list(new_edge))
            edge_set.add(new_edge)
            edge_set.discard(removed_edge)
            
            new_score = _score(values, edges)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = values[:]
                    best_edges = edges[:]
            else:
                # Revert
                edges.remove(list(new_edge))
                edge_set.discard(new_edge)
                edges.append(list(removed_edge))
                edge_set.add(removed_edge)
                edges.sort(key=lambda e: (e[0], e[1]))
                # Restore order approximately
                edges = [list(e) for e in sorted(edges, key=lambda x: (x[0], x[1]))]
    
    return best_values, best_edges, best_score

def construct():
    n = 14
    m = 35
    deadline = time.time() + 50  # Safety margin
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Multiple restarts with different strategies
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart < 3:
            # Structured initializations
            values = _generate_structured_values()
        else:
            # Random initialization
            values = random.sample(range(1, 10001), 14)
        
        edges = _generate_edges(n, m)
        
        # Local search
        vals, eds, score = _local_search(values, edges, max_iters=3000, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = eds
        
        # Early exit if we found a very good solution
        if best_score <= 10:
            break
    
    # Final fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
