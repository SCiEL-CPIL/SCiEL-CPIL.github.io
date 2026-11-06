import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    six_subsets = list(itertools.combinations(points, 6))
    num_subsets = len(six_subsets)
    
    # Map 6-subset to index
    subset_to_idx = {s: i for i, s in enumerate(six_subsets)}
    
    # Precompute for each triple, the indices of 6-subsets it covers
    # A triple covers a 6-subset if the triple is a subset of the 6-subset
    triple_to_subsets = {}
    for t in all_triples:
        t_set = set(t)
        contained_in = []
        for s in six_subsets:
            if t_set.issubset(set(s)):
                contained_in.append(subset_to_idx[s])
        triple_to_subsets[t] = contained_in

    def is_linear(triple, existing_sets):
        # existing_sets is a set of frozensets
        t_set = set(triple)
        for t in existing_sets:
            if len(t_set & t) > 1:
                return False
        return True

    best_system = []
    best_uncovered = num_subsets + 1

    # Run multiple randomized greedy constructions
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_triples = []
        current_sets = set()
        cover_count = [0] * num_subsets
        uncovered_count = num_subsets
        
        for t in all_triples:
            if is_linear(t, current_sets):
                current_triples.append(t)
                current_sets.add(frozenset(t))
                
                # Update coverage
                for idx in triple_to_subsets[t]:
                    if cover_count[idx] == 0:
                        uncovered_count -= 1
                    cover_count[idx] += 1
                
                if uncovered_count == 0:
                    # Found a system with alpha <= 5
                    return current_triples
        
        # Update best
        if uncovered_count < best_uncovered:
            best_uncovered = uncovered_count
            best_system = current_triples

    return best_system
