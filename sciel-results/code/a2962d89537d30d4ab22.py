import time
import random
from itertools import combinations

def construct():
    n = 16
    deadline = time.time() + 110
    
    all_triples = list(combinations(range(n), 3))
    
    def get_pair_partitions(points):
        pts = list(points)
        a = pts[0]
        remaining = pts[1:]
        partitions = []
        for i in range(5):
            b = remaining[i]
            rest = remaining[:i] + remaining[i+1:]
            for j in range(4):
                c = rest[j]
                rest2 = rest[:j] + rest[j+1:]
                d, e, f = rest2[0], rest2[1], rest2[2]
                partitions.append(frozenset([frozenset([a,b]), frozenset([c,d]), frozenset([e,f])]))
        return partitions
    
    # Precompute all K(2,2,2) configurations
    configs = []
    triple_to_configs = {t: [] for t in all_triples}
    
    for six_pts in combinations(range(n), 6):
        for partition in get_pair_partitions(six_pts):
            parts = list(partition)
            p1, p2, p3 = parts[0], parts[1], parts[2]
            a, b = list(p1)
            c, d = list(p2)
            e, f = list(p3)
            transversals = frozenset([
                frozenset([a,c,e]), frozenset([a,c,f]),
                frozenset([a,d,e]), frozenset([a,d,f]),
                frozenset([b,c,e]), frozenset([b,c,f]),
                frozenset([b,d,e]), frozenset([b,d,f])
            ])
            cid = len(configs)
            configs.append(transversals)
            for t in transversals:
                triple_to_configs[t].append(cid)
    
    num_configs = len(configs)
    
    best_set = set()
    best_count = 0
    
    # Multiple strategies
    strategies = [
        ('greedy_add', lambda: list(all_triples)),
        ('greedy_add_rev', lambda: list(reversed(all_triples))),
    ]
    
    for _ in range(20):
        strategies.append(('random', lambda: (lambda lst: (lst.__imul__(1), random.shuffle(lst), lst)[2])(list(all_triples))))
    
    while time.time() < deadline:
        # Greedy add: start empty, add triples that don't complete any config
        present = set()
        config_count = [0] * num_configs
        
        order = list(all_triples)
        random.shuffle(order)
        
        for t in order:
            can_add = True
            for cid in triple_to_configs[t]:
                if config_count[cid] >= 7:
                    can_add = False
                    break
            if can_add:
                present.add(t)
                for cid in triple_to_configs[t]:
                    config_count[cid] += 1
        
        if len(present) > best_count:
            best_count = len(present)
            best_set = set(present)
    
    return [list(t) for t in best_set]
