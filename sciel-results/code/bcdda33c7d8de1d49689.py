import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    n = 192

    # Precompute triple sums to speed up can_add Check 1
    # For each sum value, store the set of triples that produce it
    # Actually, for can_add we need to check if there exists a triple in S
    # whose sum is divisible by x.
    # Instead of storing all triples (memory heavy), we can just iterate
    # over triples in S directly since |S| is small (<=48).
    # O(K^3) where K<=48 is ~ 17k ops, which is fast.
    
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        S_set = set(S_list)
        
        for i, d in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            L = len(others)
            if L < 3:
                continue
            
            # Build residue count map for 'others' mod d
            counts = {}
            for val in others:
                r = val % d
                counts[r] = counts.get(r, 0) + 1
            
            found = False
            for u_idx in range(L):
                u = others[u_idx]
                ru = u % d
                for v_idx in range(u_idx + 1, L):
                    v = others[v_idx]
                    rv = v % d
                    target_r = (- (ru + rv)) % d
                    
                    cnt = counts.get(target_r, 0)
                    if cnt == 0:
                        continue
                    
                    if target_r != ru and target_r != rv:
                        found = True
                        break
                    elif target_r == ru and target_r != rv:
                        if cnt >= 2:
                            found = True
                            break
                    elif target_r != ru and target_r == rv:
                        if cnt >= 2:
                            found = True
                            break
                    elif target_r == ru and target_r == rv:
                        if cnt >= 3:
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                return False
        
        return True

    def can_add(S, x):
        S_list = list(S)
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S
        if K >= 3:
            for i in range(K):
                u = S_list[i]
                for j in range(i + 1, K):
                    v = S_list[j]
                    for k in range(j + 1, K):
                        w = S_list[k]
                        if (u + v + w) % x == 0:
                            return False
        
        # Check 2: For each y in S, y divides sum of any triple in S U {x} that includes x
        for y in S_list:
            A = [z for z in S_list if z != y]
            L = len(A)
            if L < 2:
                continue
            for i in range(L):
                u = A[i]
                for j in range(i + 1, L):
                    v = A[j]
                    if (x + u + v) % y == 0:
                        return False
        
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if time.time() - start_time > time_limit * 0.85:
                break
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_iters=2000):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit * 0.9:
                break
            
            # Try adding an element
            if not_in_S:
                random.shuffle(not_in_S)
                added = False
                for x in not_in_S:
                    if can_add(S, x):
                        S.add(x)
                        not_in_S.remove(x)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
            
            # Try 1-swap: remove one, add one
            if not_in_S and len(S) > 0:
                S_list = list(S)
                random.shuffle(S_list)
                for r in S_list[:5]:
                    new_S = (S - {r})
                    candidates = [x for x in not_in_S if x != r]
                    if not candidates:
                        continue
                    random.shuffle(candidates)
                    for x in candidates[:10]:
                        if can_add(new_S, x):
                            new_S.add(x)
                            if is_valid(new_S):
                                S = new_S
                                not_in_S = [y for y in range(1, n + 1) if y not in S]
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_size = len(S)
                                break
                    if len(S) > best_size:
                        break
            
            # Try 2-swap: remove two, add two
            if len(not_in_S) >= 2 and len(S) >= 2:
                S_list = list(S)
                random.shuffle(S_list)
                for i in range(min(3, len(S_list))):
                    r1 = S_list[i]
                    for j in range(i + 1, min(len(S_list), i + 5)):
                        r2 = S_list[j]
                        new_S = (S - {r1, r2})
                        candidates = [x for x in not_in_S if x not in (r1, r2)]
                        if len(candidates) < 2:
                            continue
                        random.shuffle(candidates)
                        for x1 in candidates[:8]:
                            if can_add(new_S, x1):
                                new_S2 = new_S | {x1}
                                remaining_cands = [x for x in candidates if x != x1]
                                for x2 in remaining_cands[:8]:
                                    if can_add(new_S2, x2):
                                        new_S3 = new_S2 | {x2}
                                        if is_valid(new_S3):
                                            S = new_S3
                                            not_in_S = [y for y in range(1, n + 1) if y not in S]
                                            if len(S) > best_size:
                                                best_S = set(S)
                                                best_size = len(S)
                                            break
                                if len(S) > best_size:
                                    break
                        if len(S) > best_size:
                            break
                    if len(S) > best_size:
                        break
        
        return best_S

    # Phase 1: Multiple greedy seeds
    best_S = set()
    best_size = 0
    
    # Greedy from high to low
    S1 = greedy_construct(range(n, 0, -1))
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Greedy from low to high
    S2 = greedy_construct(range(1, n + 1))
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Randomized greedy
    for _ in range(5):
        if time.time() - start_time > time_limit * 0.3:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy_construct(order)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
    
    # Phase 2: Local search from best seed
    best_S = local_search(best_S)
    
    # Phase 3: More local search iterations with restarts
    while time.time() - start_time < time_limit * 0.85:
        # Try local search from current best
        improved_S = local_search(best_S, max_iters=1000)
        if len(improved_S) > len(best_S):
            best_S = improved_S
        
        # Occasionally try a fresh random seed
        if random.random() < 0.1:
            order = list(range(1, n + 1))
            random.shuffle(order)
            S_new = greedy_construct(order)
            S_new = local_search(S_new, max_iters=500)
            if len(S_new) > len(best_S):
                best_S = S_new
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
