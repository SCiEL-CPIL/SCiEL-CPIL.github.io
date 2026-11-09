import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_sets = list(combinations(range(1, 17), 4))
    n = len(all_sets)
    
    all_bitmasks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_bitmasks.append(mask)
    
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = all_bitmasks[i] & all_bitmasks[j]
            inter[i][j] = v
            inter[j][i] = v
    
    def try_add(family, k):
        for i in range(len(family)):
            a = family[i]
            ia = inter[a][k]
            for j in range(i + 1, len(family)):
                b = family[j]
                if inter[a][b] == ia == inter[b][k]:
                    return False
        return True
    
    best = []
    order = list(range(n))
    
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        random.shuffle(order)
        fam = []
        for k in order:
            if try_add(fam, k):
                fam.append(k)
        if len(fam) > len(best):
            best = fam[:]
    
    return [sorted(all_sets[i]) for i in best]
