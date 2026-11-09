import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    triple_to_configs = defaultdict(list)
    config_to_triples = []
    
    for six_set in itertools.combinations(range(n), 6):
        a, b, c, d, e, f = six_set
        for partner_idx in range(1, 6):
            partner = six_set[partner_idx]
            remaining = [six_set[i] for i in range(6) if i != 0 and i != partner_idx]
            r0, r1, r2, r3 = remaining
            pairs_list = [
                [(a, partner), (r0, r1), (r2, r3)],
                [(a, partner), (r0, r2), (r1, r3)],
                [(a, partner), (r0, r3), (r1, r2)],
            ]
            for pairs in pairs_list:
                p1, p2, p3 = pairs
                transversals = set()
                for x in p1:
                    for y in p2:
                        for z in p3:
                            transversals.add(tuple(sorted((x, y, z))))
                config_id = len(config_to_triples)
                config_to_triples.append(list(transversals))
                for t in transversals:
                    triple_to_configs[t].append(config_id)
    
    num_configs = len(config_to_triples)
    
    def try_add(ti, present, config_count):
        if present[ti]:
            return False
        t = all_triples[ti]
        for cid in triple_to_configs[t]:
            if config_count[cid] == 7:
                return False
        present[ti] = True
        for cid in triple_to_configs[t]:
            config_count[cid] += 1
        return True
    
    def try_remove(ti, present, config_count):
        if not present[ti]:
            return False
        t = all_triples[ti]
        present[ti] = False
        for cid in triple_to_configs[t]:
            config_count[cid] -= 1
        return True
    
    def local_search_add(present, config_count):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for ti in range(num_triples):
                if not present[ti]:
                    if try_add(ti, present, config_count):
                        improved = True
    
    def solve_greedy(seed):
        random.seed(seed)
        present = [False] * num_triples
        config_count = [0] * num_configs
        order = list(range(num_triples))
        random.shuffle(order)
        for ti in order:
            if time.time() > deadline - 5:
                break
            try_add(ti, present, config_count)
        local_search_add(present, config_count)
        count = sum(1 for p in present if p)
        result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
        return result, count
    
    def solve_deletion(seed):
        random.seed(seed)
        present = [True] * num_triples
        config_count = [0] * num_configs
        for i in range(num_triples):
            for cid in triple_to_configs[all_triples[i]]:
                config_count[cid] += 1
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            violated_cid = -1
            for cid in range(num_configs):
                if config_count[cid] == 8:
                    violated_cid = cid
                    break
            if violated_cid == -1:
                break
            triples_in_config = config_to_triples[violated_cid]
            best_ti = -1
            best_score = -1
            for t in triples_in_config:
                ti = triple_index[t]
                if not present[ti]:
                    continue
                score = 0
                for c in triple_to_configs[t]:
                    if config_count[c] == 8:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_ti = ti
            if best_ti == -1:
                break
            try_remove(best_ti, present, config_count)
        local_search_add(present, config_count)
        count = sum(1 for p in present if p)
        result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
        return result, count
    
    best_result = None
    best_count = 0
    seed = 0
    while time.time() < deadline - 10:
        if seed % 2 == 0:
            result, count = solve_greedy(seed)
        else:
            result, count = solve_deletion(seed)
        if count > best_count:
            best_count = count
            best_result = result
        seed += 1
    
    if best_result is None:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    return best_result
