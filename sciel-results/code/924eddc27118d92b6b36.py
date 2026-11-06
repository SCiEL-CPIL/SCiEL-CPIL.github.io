import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    n = 192
    
    def is_valid_set(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        for a in S_list:
            others = [v for v in S_list if v != a]
            L = len(others)
            for i in range(L):
                for j in range(i + 1, L):
                    for k in range(j + 1, L):
                        if (others[i] + others[j] + others[k]) % a == 0:
                            return False
        return True
    
    def difficulty(x):
        if x > 96:
            if x > 1:
                prime = True
                for p in range(2, int(x**0.5) + 1):
                    if x % p == 0:
                        prime = False
                        break
                if prime:
                    return 0
            return 1
        return 2 + (n - x) / n
    
    S = set()
    candidates = sorted(range(1, n + 1), key=difficulty)
    
    for c in candidates:
        if time.time() - start_time >= time_limit:
            break
        S.add(c)
        if not is_valid_set(S):
            S.remove(c)
    
    best_S = set(S)
    best_size = len(S)
    
    for it in range(5000):
        if time.time() - start_time >= time_limit:
            break
        
        U_not_S = [x for x in range(1, n + 1) if x not in S]
        if not U_not_S:
            break
        
        added = False
        for c in sorted(U_not_S, key=difficulty):
            S.add(c)
            if is_valid_set(S):
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                added = True
                break
            else:
                S.remove(c)
        
        if added:
            continue
        
        S_list = list(S)
        if not S_list:
            break
        
        found = False
        for rem in sorted(S_list, key=lambda x: -difficulty(x))[:20]:
            for add in sorted(U_not_S, key=difficulty)[:30]:
                if add == rem:
                    continue
                new_S = (S - {rem}) | {add}
                if is_valid_set(new_S):
                    S = new_S
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    found = True
                    break
            if found:
                break
        
        if found:
            continue
        
        if len(S) > 5:
            for _ in range(20):
                if time.time() - start_time >= time_limit:
                    break
                r1, r2 = random.sample(S_list, 2)
                add = random.choice(U_not_S)
                new_S = (S - {r1, r2}) | {add}
                if is_valid_set(new_S):
                    S = new_S
                    found = True
                    break
        
        if not found and it > 1000:
            S = set()
            random.shuffle(candidates)
            for c in candidates:
                if time.time() - start_time >= time_limit:
                    break
                S.add(c)
                if not is_valid_set(S):
                    S.remove(c)
    
    U_not_S = [x for x in range(1, n + 1) if x not in S]
    for c in sorted(U_not_S, key=difficulty):
        if time.time() - start_time >= time_limit:
            break
        S.add(c)
        if not is_valid_set(S):
            S.remove(c)
    
    if len(best_S) > len(S):
        S = best_S
    
    return list(S)
