import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Fast lookup: triple tuple -> index
_TRIPLE_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _TRIPLE_IDX[(a, b, c)]

def _build_k222_structure():
    """Build incidence between triples and K(2,2,2) configurations."""
    k222_triples = []  # list of 8-tuples of triple indices
    triple_to_k222s = [[] for _ in range(N_TRIPLES)]
    
    seen = set()
    for a in range(V):
        for a_p in range(a+1, V):
            for b in range(V):
                if b in (a, a_p):
                    continue
                for b_p in range(V):
                    if b_p in (a, a_p, b):
                        continue
                    for c in range(V):
                        if c in (a, a_p, b, b_p):
                            continue
                        for c_p in range(V):
                            if c_p in (a, a_p, b, b_p, c):
                                continue
                            pairs = sorted([(min(a,a_p), max(a,a_p)),
                                            (min(b,b_p), max(b,b_p)),
                                            (min(c,c_p), max(c,c_p))])
                            key = (pairs[0], pairs[1], pairs[2])
                            if key in seen:
                                continue
                            seen.add(key)
                            
                            p1, p2 = pairs[0]
                            p3, p4 = pairs[1]
                            p5, p6 = pairs[2]
                            
                            triples_in_k222 = [
                                (p1, p3, p5),
                                (p1, p3, p6),
                                (p1, p4, p5),
                                (p1, p4, p6),
                                (p2, p3, p5),
                                (p2, p3, p6),
                                (p2, p4, p5),
                                (p2, p4, p6),
                            ]
                            
                            triple_indices = []
                            for t in triples_in_k222:
                                t_sorted = tuple(sorted(t))
                                triple_indices.append(_TRIPLE_IDX[t_sorted])
                            
                            k222_triples.append(triple_indices)
    
    for k222_idx, triple_indices in enumerate(k222_triples):
        for t_idx in triple_indices:
            triple_to_k222s[t_idx].append(k222_idx)
    
    return k222_triples, triple_to_k222s

K222_TRIPLES, TRIPLE_TO_K222S = _build_k222_structure()
N_K222 = len(K222_TRIPLES)

def _precompute_blockers():
    """For each triple, precompute the other 7 triples in each shared K(2,2,2)."""
    blockers = []
    for t_idx in range(N_TRIPLES):
        t_blockers = []
        for k222_idx in TRIPLE_TO_K222S[t_idx]:
            others = [t for t in K222_TRIPLES[k222_idx] if t != t_idx]
            t_blockers.append(tuple(others))
        blockers.append(t_blockers)
    return blockers

BLOCKERS = _precompute_blockers()

def _can_add(t_idx, presence):
    """Check if triple t_idx can be added. presence is a list of bools."""
    for others in BLOCKERS[t_idx]:
        all_present = True
        for o in others:
            if not presence[o]:
                all_present = False
                break
        if all_present:
            return False
    return True

def _greedy(order, deadline):
    presence = [False] * N_TRIPLES
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        if _can_add(idx, presence):
            presence[idx] = True
            count += 1
    return count, presence

def _local_search(presence, count, deadline):
    improved = True
    iter_count = 0
    max_iters = 300
    
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        absent = [i for i in range(N_TRIPLES) if not presence[i]]
        random.shuffle(absent)
        
        # Try adds
        for idx in absent:
            if time.time() >= deadline:
                return count, presence
            if _can_add(idx, presence):
                presence[idx] = True
                count += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps
        present = [i for i in range(N_TRIPLES) if presence[i]]
        random.shuffle(present)
        
        swap_done = False
        for new_idx in absent[:30]:
            if time.time() >= deadline:
                return count, presence
            if _can_add(new_idx, presence):
                presence[new_idx] = True
                count += 1
                improved = True
                swap_done = True
                break
            
            # Find blocking present triples
            blocking = set()
            for others in BLOCKERS[new_idx]:
                if all(presence[o] for o in others):
                    blocking.update(others)
            
            if not blocking:
                continue
            
            blocking_list = list(blocking)
            random.shuffle(blocking_list)
            for rem_idx in blocking_list[:15]:
                if not presence[rem_idx]:
                    continue
                presence[rem_idx] = False
                if _can_add(new_idx, presence):
                    presence[new_idx] = True
                    improved = True
                    swap_done = True
                    break
                else:
                    presence[rem_idx] = True
            
            if swap_done:
                break
        
        if not swap_done:
            break
    
    return count, presence

def construct():
    start = time.time()
    deadline = start + 110
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    
    while time.time() < deadline - 5:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        count, presence = _local_search(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
    
    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

random.seed(42)
