import random
import math
import time
import numpy as np

def construct():
    n = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, n):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
            
    ap4s = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Maps
    elem_to_ap3 = [[] for _ in range(n)]
    for i, ap in enumerate(ap3s):
        for idx in ap:
            elem_to_ap3[idx].append(i)
            
    elem_to_ap4 = [[] for _ in range(n)]
    for i, ap in enumerate(ap4s):
        for idx in ap:
            elem_to_ap4[idx].append(i)

    # Weights: Marginal 3-AP count
    weights = np.array([len(elem_to_ap3[i]) for i in range(n)])
    
    # Ordering for B&B: High weight first
    order = np.argsort(-weights)
    order_list = order.tolist()
    
    # Suffix sum of weights for bounding
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order_list[i]]
        
    # Fast Greedy to establish a strong incumbent
    def greedy_solve(ordering):
        included = [False] * n
        for i in ordering:
            # Check 4-AP constraint
            valid = True
            for ap_idx in elem_to_ap4[i]:
                ap = ap4s[ap_idx]
                cnt = 0
                for x in ap:
                    if x != i and included[x]:
                        cnt += 1
                if cnt == 3:
                    valid = False
                    break
            if valid:
                included[i] = True
        
        # Score
        score = 0
        for ap in ap3s:
            if all(included[x] for x in ap):
                score += 1
        return [i for i in range(n) if included[i]], score

    # Run greedy a few times with slightly perturbed orders to get a good start
    best_set, best_score = greedy_solve(order_list)
    
    # Try a few random shuffles of the tail of the order
    for _ in range(10):
        temp_order = order_list[:]
        # Shuffle the last 50 elements
        if len(temp_order) > 50:
            tail = temp_order[-50:]
            random.shuffle(tail)
            temp_order[-50:] = tail
        s_set, s_score = greedy_solve(temp_order)
        if s_score > best_score:
            best_score = s_score
            best_set = s_set

    start_time = time.time()
    time_limit = 50.0
    
    # B&B
    # State: pos, included (list of bools), current_score
    # We use a stack to avoid recursion depth issues if needed, but 256 is small enough for recursion if optimized.
    
    def can_add(i, included):
        for ap_idx in elem_to_ap4[i]:
            ap = ap4s[ap_idx]
            cnt = 0
            for x in ap:
                if x != i and included[x]:
                    cnt += 1
            if cnt == 3:
                return False
        return True

    def get_delta(i, included):
        delta = 0
        for ap_idx in elem_to_ap3[i]:
            ap = ap3s[ap_idx]
            cnt = 0
            for x in ap:
                if x != i and included[x]:
                    cnt += 1
            if cnt == 2:
                delta += 1
        return delta

    # Iterative B&B using a stack
    # Stack items: (pos, included_copy, current_score)
    # To save memory, we can't copy the list every time. We must use backtracking (DFS).
    
    included = [False] * n
    
    def dfs(pos, current_score):
        nonlocal best_score, best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Bound
        if current_score + suffix_sum[pos] <= best_score:
            return
            
        if pos == n:
            if current_score > best_score:
                best_score = current_score
                best_set = [i for i in range(n) if included[i]]
            return
            
        i = order_list[pos]
        
        # Try Include
        if can_add(i, included):
            included[i] = True
            delta = get_delta(i, included)
            dfs(pos + 1, current_score + delta)
            included[i] = False
            
        # Try Exclude
        dfs(pos + 1, current_score)

    dfs(0, 0)
    
    return best_set
