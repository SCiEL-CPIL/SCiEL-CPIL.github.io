import itertools

def construct():
    # Odd numbers in [1, 180]
    odds = list(range(1, 181, 2))
    # Even numbers in [1, 180]
    evens = list(range(2, 181, 2))
    
    # Build conflict graph for evens
    # Edge exists if e1, e2 form a forbidden triple with an odd number
    n = len(evens)
    conflict = [[False] * n for _ in range(n)]
    
    for i in range(n):
        e1 = evens[i]
        for j in range(i + 1, n):
            e2 = evens[j]
            # Check if 1/o = 1/e1 + 1/e2 has an odd integer solution o in [1, 180]
            # o = e1 * e2 / (e1 + e2)
            num = e1 * e2
            den = e1 + e2
            if num % den == 0:
                o = num // den
                if o % 2 == 1 and 1 <= o <= 180:
                    conflict[i][j] = True
                    conflict[j][i] = True

    # Find Maximum Independent Set
    # Simple greedy: sort by degree, pick lowest degree, remove neighbors
    # Or use a basic backtracking for better results
    # Given n=90, greedy might be enough to beat 90.
    # Let's try a simple greedy first.
    
    # Calculate degrees
    degrees = [sum(conflict[i]) for i in range(n)]
    
    # Greedy MIS
    selected = []
    remaining = set(range(n))
    
    # Sort indices by degree
    indices = sorted(range(n), key=lambda x: degrees[x])
    
    # We need to be careful: picking a node removes its neighbors.
    # Standard greedy: pick node with min degree, add to MIS, remove it and its neighbors from remaining.
    # Repeat.
    
    # To make it more effective, we can re-sort or use a priority queue.
    # For n=90, a simple loop is fine.
    
    while remaining:
        # Find node in remaining with min degree (within remaining)
        # Since degrees change, we might need to recalculate or just use original degrees as heuristic.
        # Using original degrees is a valid heuristic.
        min_node = min(remaining, key=lambda x: degrees[x])
        selected.append(min_node)
        # Remove min_node and its neighbors from remaining
        to_remove = {min_node}
        for j in range(n):
            if j in remaining and conflict[min_node][j]:
                to_remove.add(j)
        remaining -= to_remove
        
    # The set of evens to include
    included_evens = [evens[i] for i in selected]
    
    # Combine
    result = set(odds)
    result.update(included_evens)
    
    return list(result)
