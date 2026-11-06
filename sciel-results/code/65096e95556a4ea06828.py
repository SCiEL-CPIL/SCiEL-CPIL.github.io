"""Optimized search for large sunflower-free families of 4-subsets of [1,16]."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for a in range(1, 17):
        for b in range(a + 1, 17):
            for c in range(b + 1, 17):
                for d in range(c + 1, 17):
                    all_sets.append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d-1)))
    
    n_total = len(all_sets)
    
    # Precompute intersections
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        ai = all_sets[i]
        for j in range(i + 1, n_total):
            v = ai & all_sets[j]
            inter[i][j] = v
            inter[j][i] = v
    
    def greedy_build(order):
        family = []
        in_fam = [False] * n_total
        for idx in order:
            if in_fam[idx]:
                continue
            ok = True
            for i in range(len(family)):
                a = family[i]
                ia = inter[a]
                for j in range(i + 1, len(family)):
                    b = family[j]
                    if ia[b] == ia[idx] == inter[b][idx]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                family.append(idx)
                in_fam[idx] = True
        return family, in_fam
    
    def local_improve(family, in_fam):
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            for idx in range(n_total):
                if in_fam[idx]:
                    continue
                ok = True
                for i in range(len(family)):
                    a = family[i]
                    ia = inter[a]
                    for j in range(i + 1, len(family)):
                        b = family[j]
                        if ia[b] == ia[idx] == inter[b][idx]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    family.append(idx)
                    in_fam[idx] = True
                    improved = True
                    break
        return family, in_fam
    
    best_family = []
    rng = random.Random(777)
    
    for restart in range(200):
        if time.time() > DEADLINE - 5:
            break
        order = list(range(n_total))
        rng.shuffle(order)
        family, in_fam = greedy_build(order)
        family, in_fam = local_improve(family, in_fam)
        if len(family) > len(best_family):
            best_family = family[:]
        if len(best_family) >= 250:
            break
    
    if time.time() < DEADLINE - 3:
        in_fam = [False] * n_total
        for idx in best_family:
            in_fam[idx] = True
        best_family, in_fam = local_improve(best_family, in_fam)
    
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = [i+1 for i in range(16) if (mask >> i) & 1]
        result.append(elems)
    return result
