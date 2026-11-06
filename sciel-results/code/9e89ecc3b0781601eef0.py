import random
import time

def construct():
    start_time = time.time()
    timeout = 55
    
    # Initialize with a set of large numbers
    # These are safer because their reciprocals are small
    S = set(range(12, 25))
    
    # Verify initial set
    if not _is_valid_fast(S):
        S = set(range(15, 25))
        
    best_size = len(S)
    best_S = set(S)
    
    # Candidates to add: 1 to 11
    candidates = list(range(1, 12))
    random.shuffle(candidates)
    
    # Local search loop
    while time.time() - start_time < timeout:
        progress = False
        
        # 1. Try to add a candidate
        if candidates:
            idx = random.randint(0, len(candidates) - 1)
            a = candidates[idx]
            if a not in S:
                new_S = S | {a}
                if _is_valid_fast(new_S):
                    S = new_S
                    candidates.remove(a)
                    random.shuffle(candidates)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    progress = True
        
        # 2. Try to swap: replace one element in S with one from candidates
        if candidates and S:
            a = random.choice(candidates)
            if a not in S:
                v = random.choice(list(S))
                new_S = (S - {v}) | {a}
                if _is_valid_fast(new_S):
                    S = new_S
                    candidates.remove(a)
                    random.shuffle(candidates)
                    progress = True
        
        if not candidates:
            break

    # Final validation
    if _is_valid_fast(best_S):
        return sorted(best_S)
    return sorted(S)

def _is_valid_fast(S):
    S = set(S)
    if not S:
        return True
    
    # Use floats for speed
    recip = {a: 1.0 / a for a in S}
    EPS = 1e-12
    
    for a in S:
        target = recip[a]
        others = [b for b in S if b != a]
        
        # Sort by reciprocal descending (b ascending) to prune better
        others_sorted = sorted(others, key=lambda x: x) 
        
        found = False
        
        def dfs(idx, current_sum):
            nonlocal found
            if found:
                return
            if current_sum > target + EPS:
                return
            if abs(current_sum - target) < EPS:
                found = True
                return
            
            for i in range(idx, len(others_sorted)):
                b = others_sorted[i]
                new_sum = current_sum + recip[b]
                if new_sum <= target + EPS:
                    dfs(i + 1, new_sum)
                    if found:
                        return
                    
        dfs(0, 0.0)
        if found:
            return False
            
    return True
