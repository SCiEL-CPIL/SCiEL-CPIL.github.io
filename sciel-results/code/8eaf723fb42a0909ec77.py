import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Keep well under 60s
    
    # Precompute LCM and weights
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 25), 1)
    W = {x: L // x for x in range(1, 25)}
    
    best_set = []
    best_size = 0
    
    def get_subset_sums_int(weights):
        sums = {0}
        for w in weights:
            current_sums = list(sums)
            for s in current_sums:
                sums.add(s + w)
        return sums

    def is_valid_int(S):
        if not S:
            return True
        # Map elements to weights
        weights = [W[x] for x in S]
        # For each element, check if its weight is a subset sum of the others
        for i in range(len(S)):
            others_weights = weights[:i] + weights[i+1:]
            sums_others = get_subset_sums_int(others_weights)
            if weights[i] in sums_others:
                return False
        return True

    def greedy_random():
        S = []
        sums = {0}
        candidates = list(range(1, 25))
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() > deadline:
                break
            w = W[b]
            if w in sums:
                continue
            S.append(b)
            current_sums = list(sums)
            for s in current_sums:
                sums.add(s + w)
        return S

    def local_search(S):
        S = list(S)
        improved = True
        visited = set()
        
        while improved and time.time() < deadline:
            improved = False
            state = frozenset(S)
            if state in visited:
                break
            visited.add(state)
            
            # 1. Try to add any element
            for j in range(1, 25):
                if j in S:
                    continue
                new_S = S + [j]
                if is_valid_int(new_S):
                    S = new_S
                    improved = True
                    break # Restart to allow more additions
            
            # 2. Try swaps: remove i, add j
            if not improved:
                for i in S:
                    if time.time() > deadline:
                        return S
                    for j in range(1, 25):
                        if j in S or j == i:
                            continue
                        new_S = [x for x in S if x != i] + [j]
                        if is_valid_int(new_S):
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
                            
        return S

    # Main Loop
    for _ in range(50): # Try multiple random starts
        if time.time() > deadline:
            break
            
        S = greedy_random()
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    if not best_set:
        # Fallback to deterministic greedy
        S = []
        sums = {0}
        for b in range(24, 0, -1):
            w = W[b]
            if w in sums:
                continue
            S.append(b)
            current_sums = list(sums)
            for s in current_sums:
                sums.add(s + w)
        best_set = S
        
    return best_set
