import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def is_valid(S):
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(i + 1, len(L)):
                b = L[j]
                for k in range(j, len(L)):
                    c = L[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # ... (rest of the original functions) ...

    best_S = set()
    best_size = 0
    
    # ... (original search loop) ...

    # Safety check
    if not best_S or not is_valid(best_S):
        # Fallback to a known valid set: [171, 256]
        # This is based on the hint that 3a > max_sum for a>=171
        # Max sum for b,c <= 256 is 512. 3*171 = 513. So a>=171 is safe.
        best_S = set(range(171, 257))
        
    return sorted(best_S)
