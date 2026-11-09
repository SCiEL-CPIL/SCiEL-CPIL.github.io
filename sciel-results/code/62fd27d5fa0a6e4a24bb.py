import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave margin

    def greedy_construction():
        selected = set()
        for x in range(1, N + 1):
            is_safe = True
            max_d = N // 3 + 1
            for d in range(1, max_d + 1):
                # 1. x is start
                if (x + d in selected) and (x + 2*d in selected) and (x + 3*d in selected):
                    is_safe = False
                    break
                # 2. x is 2nd
                if x - d >= 1 and (x - d in selected) and (x + 2*d in selected) and (x + 3*d in selected):
                    is_safe = False
                    break
                # 3. x is 3rd
                if x - 2*d >= 1 and (x - 2*d in selected) and (x - d in selected) and (x + d in selected):
                    is_safe = False
                    break
                # 4. x is 4th
                if x - 3*d >= 1 and (x - 3*d in selected) and (x - 2*d in selected) and (x - d in selected):
                    is_safe = False
                    break
            if is_safe:
                selected.add(x)
        return selected

    def can_add(S, x):
        max_d = N // 3 + 1
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
            if x - d >= 1 and (x - d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
            if x - 2*d >= 1 and (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
            if x - 3*d >= 1 and (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        return True

    # Start with greedy
    current_S = greedy_construction()
    best_S = set(current_S)
    best_size = len(current_S)
    
    candidates = [x for x in range(1, N + 1) if x not in current_S]
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        if not candidates:
            break
            
        random.shuffle(candidates)
        added_any = False
        
        for x in candidates:
            if can_add(current_S, x):
                current_S.add(x)
                candidates.remove(x)
                added_any = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
                    
        if not added_any:
            # Stuck
            if current_S:
                to_remove = random.choice(list(current_S))
                current_S.remove(to_remove)
                candidates.append(to_remove)
            else:
                break
                
    return sorted(best_S)

# EVOLVE-BLOCK-END
