import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Generate all possible edges with their product and sum
    edges_info = []
    for i in range(n):
        for j in range(i + 1, n):
            prod = values[i] * values[j]
            summ = values[i] + values[j]
            edges_info.append((prod, summ, i, j))
    
    # Sort by product to minimize distinct products
    edges_info.sort(key=lambda x: x[0])
    
    # Select the first 35 edges
    selected_edges = []
    for _ in range(m):
        if time.time() > deadline:
            break
        prod, summ, i, j = edges_info[len(selected_edges)]
        selected_edges.append([i, j])
        
    # If deadline hit, return what we have (should not happen)
    # But ensure we return valid structure
    if len(selected_edges) < m:
         # Fallback to seed if something goes wrong
         return {'values': values, 'edges': [list(e) for e in itertools.combinations(range(n), 2)[:m]]}

    return {'values': values, 'edges': selected_edges}
