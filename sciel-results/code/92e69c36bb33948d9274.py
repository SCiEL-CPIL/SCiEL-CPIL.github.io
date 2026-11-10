"""Program for the suite task: maximize cardinality of S ⊆ [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
def construct():
    import itertools
    from collections import Counter

    # New Approach:
    # Use the universe {51, ..., 100}.
    # In this universe:
    # - Min 2-sum: 51+52=103
    # - Max 2-sum: 99+100=199 (No 2-sum equals 200)
    # - Min 3-sum: 51+52+53=156
    # - Max 3-sum: 98+99+100=297 (3-sums can equal 200)
    # - Min 4-sum: 51+52+53+54=210 (No 4-sum equals 200)
    # So we only need to break 3-sums.
    
    universe = list(range(51, 101))
    
    # Generate all forbidden triples
    forbidden_triples = []
    n = len(universe)
    # Optimization: Use indices to speed up sum checks slightly, or just values
    # Since n=50, C(50,3) is 19600. Very fast.
    for i in range(n):
        for j in range(i+1, n):
            # Prune: if universe[i] + universe[j] + universe[j+1] > 200, break inner loop?
            # Actually, for fixed i, j, we can calculate required third element.
            # But iterating is fine.
            for k in range(j+1, n):
                if universe[i] + universe[j] + universe[k] == 200:
                    forbidden_triples.append((universe[i], universe[j], universe[k]))
    
    if not forbidden_triples:
        return universe

    # Greedy Hitting Set on the hypergraph of triples
    remaining = set(universe)
    current_forbidden = forbidden_triples
    
    # To speed up filtering, we can maintain a list of active triples.
    # Since the number of triples is small (< 20k), linear scan is fast enough.
    
    while current_forbidden:
        counter = Counter()
        for t in current_forbidden:
            for e in t:
                if e in remaining:
                    counter[e] += 1
        
        if not counter:
            break
            
        # Pick element with max frequency
        best_elem = max(counter, key=counter.get)
        
        # Remove it
        remaining.remove(best_elem)
        
        # Filter triples
        current_forbidden = [t for t in current_forbidden if best_elem not in t]
        
    return list(remaining)
# EVOLVE-BLOCK-END
