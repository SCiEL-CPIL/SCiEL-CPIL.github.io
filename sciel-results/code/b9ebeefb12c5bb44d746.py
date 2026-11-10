<<<<<<< SEARCH
def construct():
    """Construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
    start = time.time()
    deadline = start + 115  # safety margin
    
    _indep_cache.clear()
    
    best_triples = None
    best_indep = 16
    
    # Phase 1: Multiple greedy constructions
    for seed in range(40):
        if time.time() > deadline - 40:
            break
        triples = _greedy_construct(seed, time_limit=3)
        if triples:
            indep = _independence_number(triples)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples[:]
        
        if best_indep <= 5:
            break
    
    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    # Phase 2: Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        rng = random.Random(42)
        best_triples = _local_search(best_triples, deadline, rng)
    
    return best_triples
=======
def construct():
    """Construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
    start = time.time()
    deadline = start + 140  # safety margin (evaluator kills at 150s)
    
    _indep_cache.clear()
    
    best_triples = None
    best_indep = 16
    
    # Phase 1: Multiple greedy constructions with more restarts
    num_greedy = 80
    for seed in range(num_greedy):
        if time.time() > deadline - 60:
            break
        triples = _greedy_construct(seed, time_limit=4)
        if triples:
            indep = _independence_number(triples)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples[:]
        
        if best_indep <= 5:
            break
    
    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    # Phase 2: Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        rng = random.Random(42)
        best_triples = _local_search(best_triples, deadline, rng)
    
    # Phase 3: If we have time, try local search from a second best greedy solution
    if time.time() < deadline - 20:
        _indep_cache.clear()
        # Find a different starting point
        for seed in range(200, 220):
            if time.time() > deadline - 20:
                break
            triples = _greedy_construct(seed, time_limit=3)
            if triples:
                indep = _independence_number(triples)
                if indep < best_indep:
                    best_indep = indep
                    best_triples = triples[:]
                    break
    
    if time.time() < deadline - 10:
        rng = random.Random(123)
        best_triples = _local_search(best_triples, deadline, rng)
    
    return best_triples
>>>>>>> REPLACE
