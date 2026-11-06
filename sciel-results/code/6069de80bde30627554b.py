import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    for six_pts in combinations(range(n), 6):
        a, b, c, d, e, f = six_pts
        pairings = [
            ((a,b),(c,d),(e,f)), ((a,b),(c,e),(d,f)), ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)), ((a,c),(b,e),(d,f)), ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)), ((a,d),(b,e),(c,f)), ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)), ((a,e),(b,d),(c,f)), ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)), ((a,f),(b,d),(c,e)), ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in pairings:
            x1, x2 = p1
            y1, y2 = p2
            z1, z2 = p3
            transversals = set()
            for x in (x1, x2):
                for y in (y1, y2):
                    for z in (z1, z2):
                        t = tuple(sorted((x, y, z)))
                        transversals.add(t)
            idx_set = frozenset(triple_to_idx[t] for t in transversals)
            k2_idx = len(k222_list)
            k222_list.append(idx_set)
            for idx in idx_set:
                triple_to_k222s[idx].append(k2_idx)
    
    num_k222s = len(k222_list)
    
    def greedy_construct(order):
        present = set()
        k222_count = [0] * num_k222s
        for idx in order:
            can_add = True
            for k_idx in triple_to_k222s[idx]:
                if k222_count[k_idx] >= 7:
                    can_add = False
                    break
            if can_add:
                present.add(idx)
                for k_idx in triple_to_k222s[idx]:
                    k222_count[k_idx] += 1
        return present, k222_count
    
    def local_search(present, k222_count, rng):
        improved = True
        while improved and (time.time() - start_time) < deadline - 2:
            improved = False
            present_list = list(present)
            rng.shuffle(present_list)
            for remove_idx in present_list:
                if (time.time() - start_time) > deadline - 2:
                    break
                present.discard(remove_idx)
                for k_idx in triple_to_k222s[remove_idx]:
                    k222_count[k_idx] -= 1
                added = False
                for add_idx in range(num_triples):
                    if add_idx in present:
                        continue
                    can_add = True
                    for k_idx in triple_to_k222s[add_idx]:
                        if k222_count[k_idx] >= 7:
                            can_add = False
                            break
                    if can_add:
                        present.add(add_idx)
                        for k_idx in triple_to_k222s[add_idx]:
                            k222_count[k_idx] += 1
                        improved = True
                        added = True
                        break
                if not added:
                    present.add(remove_idx)
                    for k_idx in triple_to_k222s[remove_idx]:
                        k222_count[k_idx] += 1
        return present, k222_count
    
    rng = random.Random(42)
    best_size = 0
    best_present = set()
    
    for round_num in range(30):
        if (time.time() - start_time) > deadline - 5:
            break
        if round_num == 0:
            order = list(range(num_triples))
            rng.shuffle(order)
        else:
            order = list(best_present)
            rng.shuffle(order)
            remaining = [i for i in range(num_triples) if i not in best_present]
            rng.shuffle(remaining)
            order.extend(remaining[:30])
        
        present, k222_count = greedy_construct(order)
        present, k222_count = local_search(present, k222_count, rng)
        
        if len(present) > best_size:
            best_size = len(present)
            best_present = set(present)
    
    return [list(all_triples[i]) for i in sorted(best_present)]
