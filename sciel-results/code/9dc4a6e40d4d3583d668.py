import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit
    
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
    
    # Precompute: for each triple, list of (six, config_idx) where it appears
    triple_to_constraints = {t: [] for t in all_triples}
    for six, configs in six_points_configs.items():
        for config_idx, transversal in enumerate(configs):
            for t in transversal:
                if t in triple_to_constraints:
                    triple_to_constraints[t].append((six, config_idx))
    
    # Count how many constraints each triple is in
    constraint_count = {t: len(constraints) for t, constraints in triple_to_constraints.items()}
    
    def is_valid_fast(triples_set, candidate_triple):
        """Check if adding candidate_triple to triples_set creates a violation."""
        if candidate_triple in triples_set:
            return False
        for six, config_idx in triple_to_constraints[candidate_triple]:
            transversal = six_points_configs[six][config_idx]
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
    
    def greedy_order(order):
        triples_set = set()
        for t in order:
            if is_valid_fast(triples_set, t):
                triples_set.add(t)
        return triples_set
    
    def local_search_improved(triples_set, max_steps=10000):
        """Improved local search with add, swap, and 2-swap moves."""
        current = set(triples_set)
        best = set(current)
        best_size = len(current)
        current_size = len(current)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Try adding a random triple
                t = random.choice(all_triples)
                if t not in current:
                    if is_valid_fast(current, t):
                        current.add(t)
                        current_size += 1
                        if current_size > best_size:
                            best = set(current)
                            best_size = current_size
                        continue
            
            elif move_type < 0.8:
                # Try swapping: remove one, add another
                if current:
                    remove_t = random.choice(tuple(current))
                    current.discard(remove_t)
                    
                    found = False
                    for _ in range(50):
                        add_t = random.choice(all_triples)
                        if add_t != remove_t and add_t not in current:
                            if is_valid_fast(current, add_t):
                                current.add(add_t)
                                found = True
                                break
                    
                    if found:
                        if current_size > best_size:
                            best = set(current)
                            best_size = current_size
                    else:
                        current.add(remove_t)
            
            else:
                # Try 2-swap: remove two, add two
                if len(current) >= 2:
                    items = list(current)
                    r1, r2 = random.sample(items, 2)
                    current.discard(r1)
                    current.discard(r2)
                    
                    found1 = False
                    found2 = False
                    added1 = None
                    added2 = None
                    
                    for _ in range(50):
                        a1 = random.choice(all_triples)
                        if a1 not in current and a1 != r1 and a1 != r2:
                            if is_valid_fast(current, a1):
                                current.add(a1)
                                found1 = True
                                added1 = a1
                                break
                    
                    if found1:
                        for _ in range(50):
                            a2 = random.choice(all_triples)
                            if a2 not in current and a2 != r1 and a2 != r2 and a2 != added1:
                                if is_valid_fast(current, a2):
                                    current.add(a2)
                                    found2 = True
                                    added2 = a2
                                    break
                    
                    if found1 and found2:
                        if current_size > best_size:
                            best = set(current)
                            best_size = current_size
                    else:
                        # Revert
                        if found1:
                            current.discard(added1)
                        if found2:
                            current.discard(added2)
                        current.add(r1)
                        current.add(r2)
        
        return best
    
    # Run multiple strategies
    best_triples = set()
    best_size = 0
    
    # Strategy 1: Greedy with constraint-count ordering (fewest constraints first)
    order1 = sorted(all_triples, key=lambda t: constraint_count[t])
    result1 = greedy_order(order1)
    if len(result1) > best_size:
        best_triples = result1
        best_size = len(result1)
    
    # Strategy 2: Greedy with reverse constraint-count ordering
    order2 = sorted(all_triples, key=lambda t: -constraint_count[t])
    result2 = greedy_order(order2)
    if len(result2) > best_size:
        best_triples = result2
        best_size = len(result2)
    
    # Strategy 3: Deterministic sorted order
    order3 = sorted(all_triples)
    result3 = greedy_order(order3)
    if len(result3) > best_size:
        best_triples = result3
        best_size = len(result3)
    
    # Local search on best
    if time.time() < deadline:
        ls_result = local_search_improved(best_triples, max_steps=10000)
        if len(ls_result) > best_size:
            best_triples = ls_result
            best_size = len(ls_result)
    
    # Multiple random restarts
    random.seed(42)
    
    while time.time() < deadline - 10:
        # Random greedy
        random_order = all_triples[:]
        random.shuffle(random_order)
        g_result = greedy_order(random_order)
        
        # Local search
        ls_result = local_search_improved(g_result, max_steps=8000)
        
        if len(ls_result) > best_size:
            best_triples = ls_result
            best_size = len(ls_result)
    
    # Final polish
    if time.time() < deadline:
        final = local_search_improved(best_triples, max_steps=10000)
        if len(final) > best_size:
            best_triples = final
            best_size = len(final)
    
    # Convert to list of lists
    return [list(t) for t in sorted(best_triples)]
