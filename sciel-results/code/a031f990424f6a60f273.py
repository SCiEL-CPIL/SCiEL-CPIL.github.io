"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
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

def _generate_candidate():
    """Generate a random candidate with 14 distinct positive integers <= 10000
    and 35 random edges."""
    values = random.sample(range(1, 10001), 14)
    all_edges = list(combinations(range(14), 2))
    edges = [list(e) for e in random.sample(all_edges, 35)]
    return values, edges

def _local_search(values, edges, max_iters=2000):
    """Perform local search by mutating values and edges to reduce score."""
    current_score = _score(values, edges)
    best_score = current_score
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    
    for it in range(max_iters):
        if random.random() < 0.7:
            # Mutate a value
            new_values = list(values)
            idx = random.randint(0, 13)
            old_val = new_values[idx]
            # Try a new random value
            new_val = random.randint(1, 10000)
            while new_val in new_values and new_val != old_val:
                new_val = random.randint(1, 10000)
            new_values[idx] = new_val
            new_score = _score(new_values, edges)
            if new_score <= current_score:
                values = new_values
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(values)
                    best_edges = [list(e) for e in edges]
        else:
            # Mutate an edge: swap one edge for another
            new_edges = [list(e) for e in edges]
            idx = random.randint(0, 34)
            old_edge = tuple(new_edges[idx])
            # Find a random edge not in current set
            all_edges = list(combinations(range(14), 2))
            current_set = set(tuple(sorted(e)) for e in new_edges)
            candidates = [e for e in all_edges if e not in current_set]
            if candidates:
                new_edge = random.choice(candidates)
                new_edges[idx] = list(new_edge)
                new_score = _score(values, new_edges)
                if new_score <= current_score:
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(values)
                        best_edges = [list(e) for e in edges]
    
    return best_values, best_edges, best_score

def construct():
    """Construct values and edges minimizing max(distinct sums, distinct products)."""
    random.seed(42)
    
    best_overall = None
    best_overall_score = float('inf')
    
    # Run multiple random restarts with local search
    for restart in range(5):
        values, edges = _generate_candidate()
        # Also try some structured initializations
        if restart == 0:
            # Use small consecutive integers
            values = list(range(1, 15))
            all_edges = list(combinations(range(14), 2))
            edges = [list(e) for e in all_edges[:35]]
        elif restart == 1:
            # Use powers of 2
            values = [2**i for i in range(1, 15)]
            values = [min(v, 10000) for v in values]
            # Ensure distinct
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen and v <= 10000:
                    seen.add(v)
                    unique_values.append(v)
            while len(unique_values) < 14:
                new_v = random.randint(1, 10000)
                if new_v not in seen:
                    seen.add(new_v)
                    unique_values.append(new_v)
            values = unique_values
            all_edges = list(combinations(range(14), 2))
            edges = [list(e) for e in random.sample(all_edges, 35)]
        elif restart == 2:
            # Use Fibonacci-like numbers
            values = [1, 2]
            while len(values) < 14:
                values.append(values[-1] + values[-2])
            values = [min(v, 10000) for v in values]
            # Ensure distinct and <= 10000
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen and 1 <= v <= 10000:
                    seen.add(v)
                    unique_values.append(v)
            while len(unique_values) < 14:
                new_v = random.randint(1, 10000)
                if new_v not in seen:
                    seen.add(new_v)
                    unique_values.append(new_v)
            values = unique_values
            all_edges = list(combinations(range(14), 2))
            edges = [list(e) for e in random.sample(all_edges, 35)]
        
        vals, eds, score = _local_search(values, edges, max_iters=3000)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall = (vals, eds)
        
        if best_overall_score <= 35:
            break
    
    values, edges = best_overall
    return {'values': list(values), 'edges': [list(e) for e in edges]}
# EVOLVE-BLOCK-END
