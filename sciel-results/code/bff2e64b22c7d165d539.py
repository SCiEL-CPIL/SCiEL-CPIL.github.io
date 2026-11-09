import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # For each set of 6 points, precompute the 15 ways to partition into 3 pairs
    # and the 8 transversal triples for each such partition
    six_points_configs = {}
    for six in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        configs = []
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            configs.append(transversal)
        six_points_configs[six] = configs
    
    # Build reverse mapping: for each transversal constraint, which 6-point sets it belongs to
    # Actually, let's build a faster validity check.
    # For a given triple t, which (six, transversal) pairs does it affect?
    # A triple t = (x,y,z) is in a transversal if x,y,z are chosen one from each pair.
    # So for each 6-point set containing x,y,z, and each pairing of those 6 points,
    # check if (x,y,z) is in the transversal.
    
    # Precompute: for each triple, list of (six, config_idx) where it appears
    triple_to_constraints = {t: [] for t in all_triples}
    for six, configs in six_points_configs.items():
        six_set = set(six)
        for config_idx, transversal in enumerate(configs):
            for t in transversal:
                if t in triple_to_constraints:
                    triple_to_constraints[t].append((six, config_idx))
    
    def is_valid_fast(triples_set, candidate_triple):
        """Check if adding candidate_triple to triples_set creates a violation.
        Only need to check constraints involving candidate_triple."""
        if candidate_triple in triples_set:
            return False
        for six, config_idx in triple_to_constraints[candidate_triple]:
            transversal = six_points_configs[six][config_idx]
            # Check if all 8 transversal triples are in triples_set or candidate_triple
            count = 0
            for t in transversal:
                if t == candidate_triple:
                    count += 1
                elif t in triples_set:
                    count += 1
                else:
                    break
            if count == 8:
                return False
        return True
    
    def count_violations_fast(triples_set):
        """Count number of K(2,2,2) configurations present."""
        count = 0
        for six, configs in six_points_configs.items():
            for transversal in configs:
                c = 0
                for t in transversal:
                    if t in triples_set:
                        c += 1
                if c == 8:
                    count += 1
        return count
    
    # Greedy construction: add triples in random order, skip if it creates a violation
    def greedy_random():
        random.shuffle(all_triples)
        triples_set = set()
        for t in all_triples:
            if is_valid_fast(triples_set, t):
                triples_set.add(t)
        return triples_set
    
    # Local search: try to add a random triple, or swap one out for another
    def local_search(triples_set, max_steps=5000):
        current = set(triples_set)
        current_size = len(current)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Try adding a random triple
            t = random.choice(all_triples)
            if t not in current:
                if is_valid_fast(current, t):
                    current.add(t)
                    current_size += 1
                    continue
            
            # Try swapping: remove one, add another
            if current:
                remove_t = random.choice(tuple(current))
                current.discard(remove_t)
                
                # Try to add a different triple
                found = False
                for _ in range(20):
                    add_t = random.choice(all_triples)
                    if add_t != remove_t and add_t not in current:
                        if is_valid_fast(current, add_t):
                            current.add(add_t)
                            found = True
                            break
                if not found:
                    # Couldn't find a good swap, put it back
                    current.add(remove_t)
        
        return current
    
    # Run greedy with multiple random seeds
    best_triples = set()
    best_size = 0
    
    # First, a deterministic greedy to get a baseline
    deterministic_order = sorted(all_triples, key=lambda t: (t[0], t[1], t[2]))
    for t in deterministic_order:
        if is_valid_fast(best_triples, t):
            best_triples.add(t)
    best_size = len(best_triples)
    
    # Now try randomized greedy + local search
    random.seed(42)
    
    while time.time() < deadline:
        # Greedy random
        g_result = greedy_random()
        
        # Local search
        ls_result = local_search(g_result, max_steps=3000)
        
        if len(ls_result) > best_size:
            best_triples = ls_result
            best_size = len(ls_result)
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final local search on best
    final = local_search(best_triples, max_steps=5000)
    if len(final) > best_size:
        best_triples = final
        best_size = len(final)
    
    # Convert to list of lists
    return [list(t) for t in sorted(best_triples)]
