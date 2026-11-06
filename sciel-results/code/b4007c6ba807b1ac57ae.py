"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import random
import time

def construct():
    def build_lookup(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            for t in [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]:
                triple_to_quads[triple_idx[t]].append(quad)
        for quint in combinations(range(n), 5):
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (quint[i], quint[j], quint[k])
                        triple_to_quints[triple_idx[t]].append(quint)
        return triples, triple_idx, triple_to_quads, triple_to_quints

    def count_viol(n, colors, triple_idx):
        red_k4 = 0
        blue_k5 = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_idx[(a,b,c)]] == 0 and 
                colors[triple_idx[(a,b,d)]] == 0 and 
                colors[triple_idx[(a,c,d)]] == 0 and 
                colors[triple_idx[(b,c,d)]] == 0):
                red_k4 += 1
        for quint in combinations(range(n), 5):
            all_blue = True
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        if colors[triple_idx[(quint[i], quint[j], quint[k])]] != 1:
                            all_blue = False
                            break
                    if not all_blue:
                        break
                if not all_blue:
                    break
            if all_blue:
                blue_k5 += 1
        return red_k4 + blue_k5

    def sa_search(n, max_iters=10000, timeout=90):
        start = time.time()
        triples, triple_idx, t2q, t2q5 = build_lookup(n)
        nt = len(triples)
        colors = np.random.randint(0, 2, nt).astype(np.int8)
        cv = count_viol(n, colors, triple_idx)
        bv = cv
        bc = colors.copy()
        T = 5.0
        cool = 0.99995
        for it in range(max_iters):
            if time.time() - start > timeout:
                break
            idx = np.random.randint(nt)
            old = colors[idx]
            new = 1 - old
            dr = 0
            for quad in t2q[idx]:
                rc = sum(1 for t in [(quad[0],quad[1],quad[2]),(quad[0],quad[1],quad[3]),(quad[0],quad[2],quad[3]),(quad[1],quad[2],quad[3])] if colors[triple_idx[t]] == 0)
                nrc = rc - 1 if old == 0 else rc + 1
                dr += (1 if nrc == 4 else 0) - (1 if rc == 4 else 0)
            db = 0
            for quint in t2q5[idx]:
                bc5 = sum(1 for i in range(5) for j in range(i+1,5) for k in range(j+1,5) if colors[triple_idx[(quint[i],quint[j],quint[k])]] == 1)
                nbc5 = bc5 - 1 if old == 1 else bc5 + 1
                db += (1 if nbc5 == 10 else 0) - (1 if bc5 == 10 else 0)
            d = dr + db
            if d < 0 or np.random.random() < np.exp(-d / T):
                colors[idx] = new
                cv += d
                if cv < bv:
                    bv = cv
                    bc = colors.copy()
            T = max(T * cool, 0.01)
        if bv == 0:
            return bc
        return None

    for n in range(34, 8, -1):
        for attempt in range(3):
            c = sa_search(n, max_iters=8000, timeout=90)
            if c is not None:
                return {'n': n, 'colors': [int(x) for x in c]}
    n = 9
    return {'n': n, 'colors': [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(n),3)]}
# EVOLVE-BLOCK-END
