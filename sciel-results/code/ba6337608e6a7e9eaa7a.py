def construct():
    import time
    from math import lcm
    from fractions import Fraction
    import random
    
    deadline = time.time() + 80
    
    # Precompute validity dictionary
    valid = {}
    for a in range(2, 257):
        for b in range(a+1, 257):
            valid[(a,b)] = valid[(b,a)] = (lcm(a,b) > 256)
    
    def reciprocal_sum(S):
        total = Fraction(0)
        for x in S:
            total += Fraction(1, x)
        return total
    
    # Greedy init: add in order of smallest value (highest reciprocal)
    S = []
    S_set = set()
    for x in range(2, 257):
        ok = True
        for y in S:
            if not valid[(x,y)]:
                ok = False
                break
        if ok:
            S.append(x)
            S_set.add(x)
    
    best_S = list(S)
    best_score = reciprocal_sum(S)
    
    def can_add(x, S_list, S_set):
        if x in S_set:
            return False
        for y in S_list:
            if not valid[(x,y)]:
                return False
        return True
    
    def local_search(S_list, S_set):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try add
            for x in range(2, 257):
                if x not in S_set and can_add(x, S_list, S_set):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-1 swap
            for i in range(len(S_list)):
                a = S_list[i]
                for x in range(2, 257):
                    if x in S_set:
                        continue
                    ok = True
                    for j in range(len(S_list)):
                        if j == i:
                            continue
                        if not valid[(x, S_list[j])]:
                            ok = False
                            break
                    if ok:
                        S_set.remove(a)
                        S_list[i] = x
                        S_set.add(x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try 1-for-2 (limited to 30 random candidates)
            for i in range(len(S_list)):
                a = S_list[i]
                found = False
                cands = [x for x in range(2, 257) if x not in S_set]
                random.shuffle(cands)
                for x in cands[:30]:
                    for y in cands:
                        if y <= x:
                            continue
                        if not valid[(x,y)]:
                            continue
                        ok = True
                        for j in range(len(S_list)):
                            if j == i:
                                continue
                            if not valid[(x,S_list[j])] or not valid[(y,S_list[j])]:
                                ok = False
                                break
                        if ok:
                            S_set.remove(a)
                            S_list.pop(i)
                            S_list.append(x)
                            S_list.append(y)
                            S_set.add(x)
                            S_set.add(y)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if improved:
                continue
            # Try 2-for-1 (limited to 30 random candidates)
            n_list = len(S_list)
            for i in range(n_list):
                for j in range(i+1, n_list):
                    a, b = S_list[i], S_list[j]
                    found = False
                    cands = [x for x in range(2, 257) if x not in S_set]
                    random.shuffle(cands)
                    for x in cands[:30]:
                        ok = True
                        for k in range(len(S_list)):
                            if k == i or k == j:
                                continue
                            if not valid[(x, S_list[k])]:
                                ok = False
                                break
                        if ok:
                            S_set.remove(a)
                            S_set.remove(b)
                            S_list.remove(a)
                            S_list.remove(b)
                            S_list.append(x)
                            S_set.add(x)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if improved:
                continue
            break
    
    local_search(S, S_set)
    score = reciprocal_sum(S)
    if score > best_score:
        best_S = list(S)
        best_score = score
    
    # Restarts with perturbation
    while time.time() < deadline - 3:
        S = list(best_S)
        S_set = set(S)
        k = random.randint(1, min(3, len(S)))
        for _ in range(k):
            if S:
                idx = random.randint(0, len(S)-1)
                S_set.remove(S[idx])
                S.pop(idx)
        local_search(S, S_set)
        score = reciprocal_sum(S)
        if score > best_score:
            best_S = list(S)
            best_score = score
    
    return best_S
