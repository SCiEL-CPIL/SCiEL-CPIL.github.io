import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0

    def can_add(S, x):
        # Check if x can be added to S without forming a 3-AP
        for a in S:
            # x is middle: a, x, b => b = 2x - a
            b = 2 * x - a
            if b in S and b != x:
                return False
            # x is end: a, b, x => b = (x + a) / 2
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if b in S and b != x and b != a:
                    return False
            # x is start: x, a, b => b = 2a - x
            b = 2 * a - x
            if b in S and b != x and b != a:
                return False
        return True

    # Initial Greedy
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_weight = sum(1.0 / x for x in S)
    
    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1-for-1 Swap
        S_list = sorted(S, reverse=True) # Remove largest (lowest weight) first
        candidates_v = [x for x in range(1, N + 1) if x not in S]
        candidates_v.sort() # Add smallest (highest weight) first
        
        for u in S_list:
            if time.time() > deadline:
                break
            S.remove(u)
            for v in candidates_v:
                # Prune: if 1/v <= 1/u, no further v can improve
                if 1.0 / v <= 1.0 / u:
                    break
                if can_add(S, v):
                    S.add(v)
                    new_weight = best_weight - 1.0 / u + 1.0 / v
                    if new_weight > best_weight:
                        best_weight = new_weight
                        best_S = set(S)
                        improved = True
                    S.remove(v)
                    break # Best v for this u found
            S.add(u)
            
        # 2-for-1 Swap
        if not improved:
            S_list = sorted(S, reverse=True)
            candidates_v = [x for x in range(1, N + 1) if x not in S]
            candidates_v.sort()
            
            for u in S_list:
                if time.time() > deadline:
                    break
                S.remove(u)
                found = False
                for i in range(len(candidates_v)):
                    v = candidates_v[i]
                    # Prune: max gain is 2/v. If 2/v <= 1/u, break
                    if 2.0 / v <= 1.0 / u:
                        break
                    if can_add(S, v):
                        S.add(v)
                        for j in range(i + 1, len(candidates_v)):
                            w = candidates_v[j]
                            if can_add(S, w):
                                S.add(w)
                                new_weight = best_weight - 1.0 / u + 1.0 / v + 1.0 / w
                                if new_weight > best_weight:
                                    best_weight = new_weight
                                    best_S = set(S)
                                    improved = True
                                    found = True
                                S.remove(w)
                                if found:
                                    break
                        S.remove(v)
                        if found:
                            break
                S.add(u)
                
        if not improved:
            # If stuck, try a random restart or perturbation
            # Remove 3 random elements and re-greedy?
            # Or just break to avoid wasting time.
            # Let's try a simple perturbation: remove one random element, try to re-add it or others.
            if S:
                u = random.choice(list(S))
                S.remove(u)
                # Try to add any candidate
                for v in candidates_v:
                    if can_add(S, v):
                        S.add(v)
                        if sum(1.0/x for x in S) > best_weight:
                            best_S = set(S)
                            best_weight = sum(1.0/x for x in S)
                        S.remove(v)
                        break
                S.add(u)
            
    return list(best_S)
