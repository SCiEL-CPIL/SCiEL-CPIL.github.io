import random
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 17
    pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    
    # Pre-compute all C6 difference sequences in Z_17?
    # No, just check on the fly for small sets.
    
    def has_c6_in_set(S):
        elems = list(S)
        L = len(elems)
        if L < 2:
            return False
        # Check all sequences of length 6
        # Optimization: Prune if sum cannot be 0?
        for seq in product(elems, repeat=6):
            if sum(seq) % n != 0:
                continue
            s = 0
            partials = []
            ok = True
            for d in seq:
                s = (s + d) % n
                if s == 0:
                    if len(partials) < 5:
                        ok = False
                        break
                if s in partials:
                    ok = False
                    break
                partials.append(s)
            if ok and len(partials) == 6:
                return True
        return False

    def count_c6s_circulant(assignment):
        sets = [set() for _ in range(4)]
        for i, (d1, d2) in enumerate(pairs):
            c = assignment[i]
            sets[c].add(d1)
            sets[c].add(d2)
        count = 0
        for S in sets:
            if has_c6_in_set(S):
                count += 1
        return count

    # Search for balanced assignment (2 pairs per color)
    # Permutations of [0,0,1,1,2,2,3,3]
    from itertools import permutations
    # 8! / (2!^4) = 40320 / 16 = 2520 assignments.
    # We can just iterate through all of them!
    # 2520 * 4 * 4096 = ~40 million ops. Very fast.
    
    best_assign = None
    best_count = float('inf')
    
    base = [0, 0, 1, 1, 2, 2, 3, 3]
    seen = set()
    for p in permutations(base):
        if p in seen:
            continue
        seen.add(p)
        cnt = count_c6s_circulant(p)
        if cnt < best_count:
            best_count = cnt
            best_assign = p
            if best_count == 0:
                break
                
    if best_count == 0:
        colors = [0] * (n * (n - 1) // 2)
        diff_to_color = {}
        for i, (d1, d2) in enumerate(pairs):
            diff_to_color[d1] = best_assign[i]
            diff_to_color[d2] = best_assign[i]
            
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                colors[idx] = diff_to_color[d]
                idx += 1
        return {'n': n, 'colors': colors}
        
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
