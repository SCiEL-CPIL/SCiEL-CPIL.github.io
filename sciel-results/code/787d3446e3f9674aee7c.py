import time
import random

def construct():
    start_time = time.time()
    deadline = 45.0
    n = 256
    
    best_set = set(range(129, 257))
    best_size = len(best_set)
    
    def try_start(k):
        S = set(range(k, n + 1))
        current_elements = list(range(k, n + 1))
        
        sums = set()
        for i in range(len(current_elements)):
            b = current_elements[i]
            for j in range(i, len(current_elements)):
                c = current_elements[j]
                sums.add(b + c)
        
        for a in range(k - 1, 0, -1):
            compatible = True
            for s in sums:
                if s % a == 0:
                    compatible = False
                    break
            
            if compatible:
                S.add(a)
                sums.add(2 * a)
                for b in current_elements:
                    sums.add(a + b)
                current_elements.append(a)
        
        return S

    # Deterministic starts
    for k in range(256, 127, -1):
        if time.time() - start_time > deadline:
            break
        S_candidate = try_start(k)
        if len(S_candidate) > best_size:
            best_size = len(S_candidate)
            best_set = S_candidate

    # Random starts
    while time.time() - start_time < deadline:
        k = random.randint(1, 256)
        S_candidate = try_start(k)
        if len(S_candidate) > best_size:
            best_size = len(S_candidate)
            best_set = S_candidate

    return list(best_set)
