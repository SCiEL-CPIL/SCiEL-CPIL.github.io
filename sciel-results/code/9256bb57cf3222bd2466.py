import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin out of 90s
    
    def check_validity(x, S):
        """Check if adding x to set S maintains the property."""
        if not S:
            return True
        s_list = sorted(S)
        s_len = len(s_list)
        
        # Case 1: x is 'a' in (x, b, c), x < b <= c
        for i in range(s_len):
            b = s_list[i]
            if b <= x:
                continue
            for j in range(i, s_len):
                c = s_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' in (a, x, x), a < x
        for a in S:
            if a < x and (2 * x) % a == 0:
                return False
        
        # Case 3: x is 'c' in (a, b, x), a < b <= x
        for i in range(s_len):
            a = s_list[i]
            for j in range(s_len):
                b = s_list[j]
                if a < b and b <= x and (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy(order, S_initial=None):
        S = set(S_initial) if S_initial else set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and check_validity(x, S):
                S.add(x)
        return S
    
    def try_add_elements(S, candidates):
        """Try to add as many candidates as possible to S."""
        S_new = set(S)
        for x in candidates:
            if time.time() > deadline:
                break
            if x not in S_new and check_validity(x, S_new):
                S_new.add(x)
        return S_new
    
    def local_improve(S, max_iters=50):
        """Try to improve S by removing one element and adding others."""
        S = set(S)
        improved = True
        iteration = 0
        while improved and iteration < max_iters and time.time() < deadline:
            improved = False
            iteration += 1
            s_list = sorted(S)
            for i, elem in enumerate(s_list):
                if time.time() > deadline:
                    break
                # Remove elem
                S_trial = set(S)
                S_trial.discard(elem)
                
                # Try to add other elements
                added = 0
                for x in range(1, n + 1):
                    if x in S_trial or x == elem:
                        continue
                    if check_validity(x, S_trial):
                        S_trial.add(x)
                        added += 1
                    if time.time() > deadline:
                        break
                
                if added > 1:  # Net gain of at least 1 element
                    S = S_trial
                    improved = True
                    break
        return S
    
    best_S = set()
    
    # Strategy 1: Descending greedy
    S1 = greedy(range(n, 0, -1))
    if len(S1) > len(best_S):
        best_S = set(S1)
    
    # Strategy 2: Ascending greedy
    if time.time() < deadline:
        S2 = greedy(range(1, n + 1))
        if len(S2) > len(best_S):
            best_S = set(S2)
    
    # Strategy 3: Random shuffles
    if time.time() < deadline:
        for seed in range(100):
            if time.time() > deadline - 20:
                break
            random.seed(seed)
            order = list(range(1, n + 1))
            random.shuffle(order)
            S3 = greedy(order)
            if len(S3) > len(best_S):
                best_S = set(S3)
    
    # Strategy 4: Local improvement on best solution
    if time.time() < deadline - 10 and len(best_S) > 0:
        improved_S = local_improve(best_S, max_iters=30)
        if len(improved_S) > len(best_S):
            best_S = improved_S
    
    # Strategy 5: Try more aggressive local search with different removal patterns
    if time.time() < deadline - 5 and len(best_S) > 0:
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            S_trial = set(best_S)
            # Randomly remove 1-2 elements
            to_remove = random.sample(sorted(S_trial), min(2, len(S_trial)))
            for elem in to_remove:
                S_trial.discard(elem)
            
            # Try to fill back up
            for x in range(1, n + 1):
                if x in S_trial:
                    continue
                if check_validity(x, S_trial):
                    S_trial.add(x)
            
            if len(S_trial) > len(best_S):
                best_S = S_trial
    
    if not best_S:
        return [256]
    return list(best_S)
