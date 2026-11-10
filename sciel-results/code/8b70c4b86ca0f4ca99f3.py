import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55

    # Precompute all 3-APs
    three_aps = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # Precompute all 4-APs
    four_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Build mappings
    elem_to_3aps = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append(i)
        elem_to_3aps[b].append(i)
        elem_to_3aps[c].append(i)
    
    elem_to_4aps = [[] for _ in range(n + 1)]
    for i, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append(i)
        elem_to_4aps[b].append(i)
        elem_to_4aps[c].append(i)
        elem_to_4aps[d].append(i)
    
    # Precompute 4-AP other masks for fast conflict checking
    four_ap_other_mask = {}
    for i, (a, b, c, d) in enumerate(four_aps):
        for y in (a, b, c, d):
            others = [z for z in (a, b, c, d) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_other_mask[(y, i)] = m
    
    def count_3aps(S):
        S_set = set(S)
        count = 0
        for a, b, c in three_aps:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    def has_4ap(S):
        S_set = set(S)
        for a, b, c, d in four_aps:
            if a in S_set and b in S_set and c in S_set and d in S_set:
                return True
        return False
    
    def adds_4ap(y, mask):
        for idx in elem_to_4aps[y]:
            om = four_ap_other_mask[(y, idx)]
            if (mask & om) == om:
                return True
        return False
    
    def gain_add(y, S):
        S_set = set(S)
        cnt = 0
        for idx in elem_to_3aps[y]:
            a, b, c = three_aps[idx]
            oth = [z for z in (a, b, c) if z != y]
            if all(z in S_set for z in oth):
                cnt += 1
        return cnt
    
    def loss_rem(y, S):
        S_set = set(S)
        cnt = 0
        for idx in elem_to_3aps[y]:
            a, b, c = three_aps[idx]
            oth = [z for z in (a, b, c) if z != y]
            if all(z in S_set for z in oth):
                cnt += 1
        return cnt
    
    best_score = -1
    best_set = set()
    
    # Multiple diverse greedy starts
    strategies = []
    
    # Strategy 1: ascending order
    strategies.append(list(range(1, n + 1)))
    
    # Strategy 2: descending order
    strategies.append(list(range(n, 0, -1)))
    
    # Strategy 3: odd first, then even
    strategies.append([x for x in range(1, n + 1) if x % 2 == 1] + [x for x in range(1, n + 1) if x % 2 == 0])
    
    # Strategy 4: random shuffles
    for _ in range(6):
        order = list(range(1, n + 1))
        random.shuffle(order)
        strategies.append(order)
    
    for strat in strategies:
        if time.time() > deadline - 8:
            break
        
        S = set()
        S_mask = 0
        for x in strat:
            if not adds_4ap(x, S_mask):
                S.add(x)
                S_mask |= (1 << (x - 1))
        
        sc = count_3aps(S)
        if sc > best_score:
            best_score = sc
            best_set = set(S)
        
        # Local search on this start
        if time.time() > deadline - 5:
            break
        
        tabu = {}
        tenure = 12
        no_imp = 0
        max_no_imp = 20
        
        for it in range(200):
            if time.time() > deadline - 3:
                break
            
            best_gain = -1
            best_type = None
            best_x = None
            best_y = None
            
            # Try adds
            for y in range(1, n + 1):
                if y in S:
                    continue
                if tabu.get(y, 0) > it:
                    continue
                if not adds_4ap(y, S_mask):
                    g = gain_add(y, S)
                    if g > best_gain:
                        best_gain = g
                        best_type = 'add'
                        best_x = None
                        best_y = y
            
            # Try removes
            for x in list(S):
                if tabu.get(x, 0) > it:
                    continue
                g = -loss_rem(x, S)
                if g > best_gain:
                    best_gain = g
                    best_type = 'remove'
                    best_x = x
                    best_y = None
            
            # Try swaps (sampled)
            if time.time() < deadline - 4:
                cands = [y for y in range(1, n + 1) if y not in S]
                if cands and S:
                    for _ in range(40):
                        x = random.choice(list(S))
                        y = random.choice(cands)
                        if tabu.get(x, 0) > it or tabu.get(y, 0) > it:
                            continue
                        tm = S_mask & ~(1 << (x - 1))
                        if adds_4ap(y, tm):
                            continue
                        lx = loss_rem(x, S)
                        gy = gain_add(y, S)
                        # Subtract shared 3-APs containing both x and y
                        shared = 0
                        S_set = set(S)
                        for idx in elem_to_3aps[x]:
                            a, b, c = three_aps[idx]
                            if y in (a, b, c):
                                oth = [z for z in (a, b, c) if z not in (x, y)]
                                if all(z in S_set for z in oth):
                                    shared += 1
                        g = -lx + (gy - shared)
                        if g > best_gain:
                            best_gain = g
                            best_type = 'swap'
                            best_x = x
                            best_y = y
            
            if best_type is None or best_gain < 0:
                no_imp += 1
                if no_imp >= max_no_imp:
                    break
                continue
            
            if best_type == 'add':
                S.add(best_y)
                S_mask |= (1 << (best_y - 1))
                tabu[best_y] = it + tenure
            elif best_type == 'remove':
                S.remove(best_x)
                S_mask &= ~(1 << (best_x - 1))
                tabu[best_x] = it + tenure
            else:
                S.remove(best_x)
                S.add(best_y)
                S_mask &= ~(1 << (best_x - 1))
                S_mask |= (1 << (best_y - 1))
                tabu[best_x] = it + tenure
                tabu[best_y] = it + tenure
            
            sc += best_gain
            no_imp = 0
            
            if sc > best_score:
                best_score = sc
                best_set = set(S)
    
    # Final verification
    if best_set and has_4ap(best_set):
        S = set()
        for x in range(1, n + 1):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if all((x - j * d) in S for j in (1, 2, 3)):
                    ok = False
                    break
            if ok:
                S.add(x)
        best_set = S
        best_score = count_3aps(S)
    
    return sorted(best_set)
