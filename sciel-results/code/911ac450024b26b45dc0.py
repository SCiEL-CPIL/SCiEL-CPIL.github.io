import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    
    def is_valid(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = S[i], S[j]
                for k in range(j+1, len(S)):
                    c = S[k]
                    if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                        return False
        return True
    
    def check_triple(S_set, a, b, c):
        """Check if adding c to {a,b} creates a forbidden triple."""
        # a,b,c distinct, check if a+b, a+c, b+c all in S_set
        return (a+b) in S_set and (a+c) in S_set and (b+c) in S_set
    
    def greedy_build(order):
        """Build set greedily following given order."""
        S = set()
        for x in order:
            # Check if adding x creates any forbidden triple
            valid = True
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    if check_triple(S | {x}, a, b, x):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
        return S
    
    best_size = 0
    best_set = set()
    
    # Initial seed: top half
    seed_set = set(range(65, 129))
    best_size = len(seed_set)
    best_set = seed_set.copy()
    
    # Multi-restart greedy
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create different orderings
        if restart == 0:
            order = list(range(65, 129)) + list(range(1, 65))
        elif restart == 1:
            order = list(range(1, 129))
        elif restart == 2:
            order = list(range(128, 0, -1))
        else:
            order = list(range(1, 129))
            random.shuffle(order)
        
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Local search: try to add elements
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        improved = False
        for x in range(1, 129):
            if x in best_set:
                continue
            # Check if adding x is valid
            S_list = list(best_set)
            valid = True
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    if check_triple(best_set | {x}, a, b, x):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_set.add(x)
                improved = True
                break
        
        if not improved:
            break
    
    # Local search: try swaps
    for iteration in range(100):
        if time.time() > deadline - 2:
            break
        
        improved = False
        S_list = list(best_set)
        for i in range(len(S_list)):
            if time.time() > deadline - 2:
                break
            for x in range(1, 129):
                if x in best_set:
                    continue
                # Try swapping S_list[i] with x
                new_set = best_set - {S_list[i]} | {x}
                # Check validity
                valid = True
                nl = list(new_set)
                for a_idx in range(len(nl)):
                    for b_idx in range(a_idx+1, len(nl)):
                        a, b = nl[a_idx], nl[b_idx]
                        for c_idx in range(b_idx+1, len(nl)):
                            c = nl[c_idx]
                            if (a+b) in new_set and (a+c) in new_set and (b+c) in new_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and len(new_set) > len(best_set):
                    best_set = new_set
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    result = sorted(best_set)
    # Ensure Python ints
    return [int(x) for x in result]
