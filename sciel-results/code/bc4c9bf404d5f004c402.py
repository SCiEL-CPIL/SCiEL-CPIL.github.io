import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    pairings_6 = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)],
    ]
    
    k222_triples = []
    triple_to_k222 = [[] for _ in range(num_triples)]
    
    for pts in combinations(range(n), 6):
        for pairing in pairings_6:
            triples_in_k = []
            p = pts
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        a = p[pairing[0][i]]
                        b = p[pairing[1][j]]
                        c = p[pairing[2][k]]
                        t = tuple(sorted((a, b, c)))
                        triples_in_k.append(triple_index[t])
            k222_triples.append(triples_in_k)
            k_idx = len(k222_triples) - 1
            for ti in triples_in_k:
                triple_to_k222[ti].append(k_idx)
    
    num_k222 = len(k222_triples)
    
    def try_greedy(order):
        selected = set()
        k222_count = [0] * num_k222
        
        for ti in order:
            can_add = True
            for k_idx in triple_to_k222[ti]:
                if k222_count[k_idx] == 7:
                    can_add = False
                    break
            if can_add:
                selected.add(ti)
                for k_idx in triple_to_k222[ti]:
                    k222_count[k_idx] += 1
        
        return selected
    
    def local_search(selected, max_iters=200):
        k222_count = [0] * num_k222
        for ti in selected:
            for k_idx in triple_to_k222[ti]:
                k222_count[k_idx] += 1
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try 1-for-1 swaps
            for ti_out in list(selected):
                for k_idx in triple_to_k222[ti_out]:
                    k222_count[k_idx] -= 1
                
                for ti_in in range(num_triples):
                    if ti_in in selected:
                        continue
                    can_add = True
                    for k_idx in triple_to_k222[ti_in]:
                        if k222_count[k_idx] == 7:
                            can_add = False
                            break
                    if can_add:
                        selected.discard(ti_out)
                        selected.add(ti_in)
                        for k_idx in triple_to_k222[ti_in]:
                            k222_count[k_idx] += 1
                        improved = True
                        break
                
                if improved:
                    break
                else:
                    for k_idx in triple_to_k222[ti_out]:
                        k222_count[k_idx] += 1
            
            if not improved:
                break
        
        return selected
    
    best_selected = set()
    best_size = 0
    
    rng = random.Random(42)
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart % 5 == 0:
            order = list(range(num_triples))
            rng.shuffle(order)
        elif restart % 5 == 1:
            order = sorted(range(num_triples), key=lambda ti: len(triple_to_k222[ti]), reverse=True)
            rng.shuffle(order)
        elif restart % 5 == 2:
            order = sorted(range(num_triples), key=lambda ti: len(triple_to_k222[ti]))
            rng.shuffle(order)
        elif restart % 5 == 3:
            # Point-based ordering
            order = []
            for pt in range(n):
                for ti in range(num_triples):
                    t = all_triples[ti]
                    if pt in t and ti not in order:
                        order.append(ti)
            rng.shuffle(order)
        else:
            # Reverse of most constrained
            order = sorted(range(num_triples), key=lambda ti: len(triple_to_k222[ti]))
            rng.shuffle(order)
        
        selected = try_greedy(order)
        selected = local_search(selected)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = set(selected)
    
    result = [list(all_triples[ti]) for ti in sorted(best_selected)]
    return result
