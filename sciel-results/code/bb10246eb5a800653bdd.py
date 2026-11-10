"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The strategy is to select a subset from {51, ..., 100} such that no subset sums to 200.
    # In this range, any subset of size >= 4 has sum >= 51+52+53+54 = 210 > 200.
    # Any subset of size 2 has sum <= 99+100 = 199 < 200.
    # Any subset of size 1 has sum <= 100 < 200.
    # Thus, we only need to prevent any subset of size 3 from summing to 200.
    
    # Generate all forbidden triples (a, b, c) with 51 <= a < b < c <= 100 and a+b+c=200
    forbidden_triples = []
    for a in range(51, 101):
        for b in range(a + 1, 101):
            c = 200 - a - b
            if c > b and c <= 100:
                forbidden_triples.append((a, b, c))
    
    # We want to find a minimum hitting set for these triples to remove from the pool {51, ..., 100}.
    # The remaining elements will form our candidate set.
    
    # Greedy Hitting Set Algorithm:
    # 1. Count the degree of each vertex (element) in the hypergraph of forbidden triples.
    # 2. Iteratively select the vertex with the highest degree, add it to the hitting set,
    #    and remove all edges (triples) containing it.
    # 3. Repeat until no edges remain.
    
    # Initialize degrees
    degrees = {i: 0 for i in range(51, 101)}
    for triple in forbidden_triples:
        for x in triple:
            degrees[x] += 1
    
    remaining_triples = list(forbidden_triples)
    hitting_set = set()
    
    while remaining_triples:
        # Find the element with the highest degree among those in remaining_triples
        # To be efficient, we can just pick the max from the current degrees, 
        # but we must ensure it's actually in a remaining triple.
        # Since degrees are static counts of the original hypergraph, 
        # an element might have high original degree but all its triples might be hit.
        # So we need to pick the element that appears in the most *remaining* triples.
        
        current_counts = {}
        for triple in remaining_triples:
            for x in triple:
                current_counts[x] = current_counts.get(x, 0) + 1
        
        if not current_counts:
            break
            
        # Pick the element with the max count in remaining triples
        best_elem = max(current_counts, key=current_counts.get)
        hitting_set.add(best_elem)
        
        # Remove all triples containing best_elem
        remaining_triples = [t for t in remaining_triples if best_elem not in t]
        
    # The candidate set is {51, ..., 100} minus the hitting_set
    candidate_set = set(range(51, 101)) - hitting_set
    
    return list(candidate_set)
# EVOLVE-BLOCK-END
