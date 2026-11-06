import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_ok():
        return time.time() < deadline

    # 1. Generate initial valid Sidon set using greedy method
    S = [0]
    diffs = set()
    covered = set()
    
    while not all(d in covered for d in range(1, 31)):
        found = False
        # Search for next element
        # Limit search to avoid infinite loop, though it should find one quickly
        for x in range(max(S) + 1, max(S) + 500):
            new_diffs = [x - y for y in S]
            # Check Sidon property (distinct new diffs)
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Check conflict with existing diffs
            conflict = False
            for d in new_diffs:
                if d in diffs:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Accept
            S.append(x)
            diffs.update(new_diffs)
            covered.update(new_diffs)
            found = True
            break
        if not found:
            # Fallback: add a large number
            x = max(S) + 100
            while True:
                new_diffs = [x - y for y in S]
                if len(set(new_diffs)) == len(new_diffs) and not any(d in diffs for d in new_diffs):
                    S.append(x)
                    diffs.update(new_diffs)
                    covered.update(new_diffs)
                    break
                x += 1

    S.sort()
    best_S = list(S)
    best_max = max(S)
    best_sum = sum(S)
    
    # 2. Local Search Optimization
    # Try to reduce max element or sum
    
    while time_ok():
        idx = random.randint(1, len(S) - 1)
        old_val = S[idx]
        
        # Try to move to a smaller value
        if old_val > 1:
            # Try a few random smaller values
            for _ in range(10):
                new_val = random.randint(1, old_val - 1)
                if new_val in S:
                    continue
                
                # Check validity
                temp_diffs = set(diffs)
                for other in S:
                    if other != old_val:
                        d = abs(old_val - other)
                        if d in temp_diffs:
                            temp_diffs.remove(d)
                
                valid = True
                for other in S:
                    if other != old_val:
                        d = abs(new_val - other)
                        if d in temp_diffs:
                            valid = False
                            break
                        temp_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check coverage
                if not all(d in temp_diffs for d in range(1, 31)):
                    continue
                
                new_S = S.copy()
                new_S[idx] = new_val
                new_S.sort()
                new_max = new_S[-1]
                new_sum = sum(new_S)
                
                if new_max < best_max or (new_max == best_max and new_sum < best_sum):
                    S = new_S
                    diffs = temp_diffs
                    best_S = list(S)
                    best_max = new_max
                    best_sum = new_sum
                    break
                elif new_sum < best_sum:
                    # Accept if it improves sum, even if max is same
                    # This helps compact the set
                    if new_max <= best_max:
                        S = new_S
                        diffs = temp_diffs
                        best_S = list(S)
                        best_max = new_max
                        best_sum = new_sum
                        break

    return best_S
