"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Try n from 34 down to 9
    for n in range(34, 8, -1):
        if time.time() >= deadline - 10:
            break
        
        triples = list(combinations(range(n), 3))
        n_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Build four-set and five-set triple lists
        four_set_triples = []
        for s in combinations(range(n), 4):
            ts = [triple_idx[(s[0],s[1],s[2])], triple_idx[(s[0],s[1],s[3])],
                  triple_idx[(s[0],s[2],s[3])], triple_idx[(s[1],s[2],s[3])]]
            four_set_triples.append(ts)
        
        five_set_triples = []
        for s in combinations(range(n), 5):
            ts = [triple_idx[t] for t in combinations(s, 3)]
            five_set_triples.append(ts)
        
        # Reverse mappings
        triple_to_four = [[] for _ in range(n_triples)]
        for i, ts in enumerate(four_set_triples):
            for t in ts:
                triple_to_four[t].append(i)
        
        triple_to_five = [[] for _ in range(n_triples)]
        for i, ts in enumerate(five_set_triples):
            for t in ts:
                triple_to_five[t].append(i)
        
        four_arr = np.array(four_set_triples) if four_set_triples else None
        five_arr = np.array(five_set_triples) if five_set_triples else None
        
        def count_viol(c):
            rv = int(np.all(c[four_arr] == 0, axis=1).sum()) if four_arr is not None else 0
            bv = int(np.all(c[five_arr] == 1, axis=1).sum()) if five_arr is not None else 0
            return rv + bv
        
        # Try initializations
        best_viol = float('inf')
        best_colors = None
        
        inits = []
        # Modular
        inits.append(np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]))
        # Random
        for _ in range(20):
            inits.append(np.random.randint(0, 2, n_triples))
        
        for c0 in inits:
            if time.time() >= deadline - 3:
                break
            c = c0.copy()
            viol = count_viol(c)
            if viol == 0:
                return {'n': n, 'colors': list(c)}
            
            # Local search
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for t in range(n_triples):
                    if time.time() >= deadline - 2:
                        break
                    oc = c[t]
                    dr = 0
                    for fi in triple_to_four[t]:
                        ts = four_set_triples[fi]
                        orr = sum(1 for ot in ts if ot != t and c[ot] == 0)
                        if oc == 0 and orr == 3: dr -= 1
                        elif oc == 1 and orr == 3: dr += 1
                    db = 0
                    for fi in triple_to_five[t]:
                        ts = five_set_triples[fi]
                        ob = sum(1 for ot in ts if ot != t and c[ot] == 1)
                        if oc == 1 and ob == 9: db -= 1
                        elif oc == 0 and ob == 9: db += 1
                    d = dr + db
                    if d < 0:
                        c[t] = 1 - oc
                        viol += d
                        improved = True
                        if viol == 0:
                            return {'n': n, 'colors': list(c)}
            
            if viol < best_viol:
                best_viol = viol
                best_colors = c
        
        # If we found a valid solution, we'd have returned already
        # Move to smaller n
    
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
