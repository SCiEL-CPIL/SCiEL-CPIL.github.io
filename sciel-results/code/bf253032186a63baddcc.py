"""Max triples on 16 points avoiding K(2,2,2)."""
import time
import random
import itertools

N = 16
DEADLINE = 115.0

def _build_data():
    triples = []
    tri_index = {}
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                idx = len(triples)
                triples.append((a, b, c))
                tri_index[(a, b, c)] = idx
    M = len(triples)
    
    config_ids = []
    triple_to_configs = [[] for _ in range(M)]
    
    for pts in itertools.combinations(range(N), 6):
        p = list(pts)
        matchings = []
        def gen(rem, cur):
            if not rem:
                matchings.append(tuple(cur))
                return
            first = min(rem)
            for i in range(len(rem)):
                second = rem[i]
                new_rem = [x for j, x in enumerate(rem) if j != i and x != first]
                gen(new_rem, cur + [(first, second)])
        gen(p, [])
        
        for matching in matchings:
            (a, a2), (b, b2), (c, c2) = matching
            trans = [
                (a, b, c), (a, b, c2), (a, b2, c), (a, b2, c2),
                (a2, b, c), (a2, b, c2), (a2, b2, c), (a2, b2, c2)
            ]
            cid = len(config_ids)
            tri_indices = [tri_index[tuple(sorted(t))] for t in trans]
            config_ids.append(tuple(tri_indices))
            for idx in tri_indices:
                triple_to_configs[idx].append(cid)
    
    return triples, config_ids, triple_to_configs

def construct():
    start = time.time()
    triples, config_ids, triple_to_configs = _build_data()
    M = len(triples)
    NC = len(config_ids)
    
    def tl():
        return DEADLINE - (time.time() - start)
    
    best = []
    best_size = 0
    
    for seed in range(500):
        if tl() < 2.0:
            break
        rng = random.Random(seed)
        present = [False] * M
        ccount = [0] * NC
        sel = []
        
        # Random order
        order = list(range(M))
        rng.shuffle(order)
        
        for t in order:
            ok = True
            for cid in triple_to_configs[t]:
                if ccount[cid] == 7:
                    ok = False
                    break
            if ok:
                present[t] = True
                sel.append(t)
                for cid in triple_to_configs[t]:
                    ccount[cid] += 1
        
        # Local search: add rejected
        rejected = [t for t in range(M) if not present[t]]
        rng.shuffle(rejected)
        for t in rejected:
            if tl() < 2.0:
                break
            ok = True
            for cid in triple_to_configs[t]:
                if ccount[cid] == 7:
                    ok = False
                    break
            if ok:
                present[t] = True
                sel.append(t)
                for cid in triple_to_configs[t]:
                    ccount[cid] += 1
        
        if len(sel) > best_size:
            best_size = len(sel)
            best = list(sel)
    
    return [list(triples[t]) for t in best]
