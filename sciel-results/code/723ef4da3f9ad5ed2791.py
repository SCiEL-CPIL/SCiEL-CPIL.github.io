import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145
    n = 16
    
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute: for each triple, the list of other-transversal tuples
    triple_data = {}
    for T in all_triples:
        a, b, c = T
        entries = []
        others = [p for p in range(n) if p not in (a, b, c)]
        for x, y, z in itertools.combinations(others, 3):
            for perm in itertools.permutations([x, y, z]):
                other_trans = (
                    frozenset((a, b, perm[2])),
                    frozenset((a, perm[1], c)),
                    frozenset((a, perm[1], perm[2])),
                    frozenset((perm[0], b, c)),
                    frozenset((perm[0], b, perm[2])),
                    frozenset((perm[0], perm[1], c)),
                    frozenset((perm[0], perm[1], perm[2])),
                )
                entries.append(other_trans)
        triple_data[T] = entries
    
    def check_violation(system_set, T):
        for other_trans in triple_data[T]:
            for ot in other_trans:
                if ot not in system_set:
                    break
            else:
                return True
        return False
    
    def greedy_randomized(seed, time_limit):
        rng = random.Random(seed)
        triples = list(all_triples)
        rng.shuffle(triples)
        
        system = set()
        system_list = []
        
        for T in triples:
            if time.time() > time_limit:
                break
            T_frozen = frozenset(T)
            if T_frozen in system:
                continue
            if not check_violation(system, T):
                system.add(T_frozen)
                system_list.append(list(T))
        
        # Second pass: try again with remaining triples
        if time.time() < time_limit - 0.5:
            for T in triples:
                if time.time() > time_limit - 0.1:
                    break
                T_frozen = frozenset(T)
                if T_frozen not in system:
                    if not check_violation(system, T):
                        system.add(T_frozen)
                        system_list.append(list(T))
        
        return system_list
    
    best_system = []
    best_count = 0
    
    seed_counter = 0
    while time.time() < deadline - 2:
        seed = seed_counter
        time_limit = min(deadline - 0.5, time.time() + 10)
        system = greedy_randomized(seed, time_limit)
        count = len(system)
        if count > best_count:
            best_count = count
            best_system = system
        seed_counter += 1
    
    return best_system
