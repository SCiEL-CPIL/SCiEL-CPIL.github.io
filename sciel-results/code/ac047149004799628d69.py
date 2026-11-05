"""Program for the suite task: maximize subset of [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Candidate set: {51, ..., 100}
    # Any subset summing to 200 must have size 3 (since min 4-sum > 200 and max 2-sum < 200).
    # We model this as a Maximum Independent Set problem on a 3-uniform hypergraph.
    # Vertices: 51..100
    # Hyperedges: Triples (a,b,c) in candidates summing to 200.
    # Greedy Heuristic: Repeatedly pick the vertex with the fewest conflicts (lowest degree),
    # add it to the result, and remove it and all its conflicting neighbors from consideration.
    
    candidates = list(range(51, 101))
    
    # Generate all conflicting triples
    triples = []
    for a, b, c in itertools.combinations(candidates, 3):
        if a + b + c == 200:
            triples.append((a, b, c))
    
    # Map each vertex to the set of triples it belongs to
    # We need to update this as we remove triples.
    # To be efficient, we can just recompute degrees or maintain a counter.
    # Given the small size (50 vertices, < 1000 triples), recomputing is fine.
    
    available = set(candidates)
    active_triples = triples[:]
    result = []
    
    while active_triples:
        # Find vertex in 'available' with minimum number of occurrences in 'active_triples'
        # If a vertex is not in any active triple, its count is 0.
        
        counts = {x: 0 for x in available}
        for t in active_triples:
            for x in t:
                if x in available:
                    counts[x] += 1
        
        # Find min count
        min_count = min(counts.values())
        
        # Pick all vertices with min count? Or just one?
        # Standard greedy: pick one.
        # If min_count is 0, we can pick all of them immediately as they have no conflicts.
        if min_count == 0:
            # Add all zero-conflict vertices
            zero_vertices = [x for x, c in counts.items() if c == 0]
            result.extend(zero_vertices)
            available.difference_update(zero_vertices)
            # Active triples remain the same (none contained these vertices)
            # But we must ensure we don't infinite loop if active_triples is not empty but no zero vertices?
            # If min_count == 0, we removed them. Next iteration will have higher min_count or empty available.
            # If available becomes empty, loop ends.
            continue
        
        # If min_count > 0, pick one vertex with min_count
        # Tie-breaker: smallest value for determinism
        candidates_for_pick = [x for x, c in counts.items() if c == min_count]
        target = min(candidates_for_pick)
        
        result.append(target)
        available.remove(target)
        
        # Remove all triples containing target
        new_active_triples = []
        # Also identify neighbors to remove from 'available'
        neighbors_to_remove = set()
        for t in active_triples:
            if target in t:
                # This triple is broken
                # The other elements in this triple are neighbors and cannot be in the independent set
                for x in t:
                    if x != target:
                        neighbors_to_remove.add(x)
            else:
                new_active_triples.append(t)
                
        active_triples = new_active_triples
        available.difference_update(neighbors_to_remove)
        
    # Add any remaining available vertices (they have no active conflicts)
    result.extend(available)
    
    # Ensure nonempty
    if not result:
        return [1]
        
    return result
# EVOLVE-BLOCK-END
