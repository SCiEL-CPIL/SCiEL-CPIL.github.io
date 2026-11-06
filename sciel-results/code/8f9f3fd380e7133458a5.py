import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def is_valid(x, S, sums):
        if 2 * x in sums:
            return False
        for s in S:
            if x + s in sums:
                return False
        return True

    def count_blocked(x, S, sums):
        new_sums = set(sums)
        new_sums.add(2 * x)
        for s in S:
            new_sums.add(x + s)
        
        S_new = S + [x]
        
        blocked = 0
        for y in range(1, N + 1):
            if y in S or y == x:
                continue
            if 2 * y in new_sums:
                blocked += 1
                continue
            for s in S_new:
                if y + s in new_sums:
                    blocked += 1
                    break
        return blocked

    best_S = None
    best_size = N + 1
    
    # Run multiple trials
    for trial in range(50):
        if time.time() > deadline:
            break
        
        random.seed(trial) # Deterministic but varied
        S = []
        sums = set()
        
        while True:
            if time.time() > deadline:
                break
                
            best_x = None
            max_blocks = -1
            candidates = []
            
            for x in range(1, N + 1):
                if x in S:
                    continue
                if not is_valid(x, S, sums):
                    continue
                blocks = count_blocked(x, S, sums)
                candidates.append((blocks, x))
                
            if not candidates:
                break
            
            # Sort by blocks desc, then randomize ties?
            # Or just pick max blocks, with random tie break
            max_b = max(b for b, _ in candidates)
            max_candidates = [x for b, x in candidates if b == max_b]
            
            best_x = random.choice(max_candidates)
            
            S.append(best_x)
            sums.add(2 * best_x)
            for s in S[:-1]:
                sums.add(best_x + s)
                
        if len(S) < best_size:
            best_size = len(S)
            best_S = S[:]
            
    return best_S if best_S else [1]
