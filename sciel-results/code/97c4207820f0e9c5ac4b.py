import time
import random
from itertools import combinations

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin
    
    n = 192
    universe = list(range(1, n + 1))
    
    def can_add(x, S):
        """
        Check if x can be added to set S without violating constraints.
        S is a list of distinct integers.
        """
        # 1. Check if x divides sum of any 3 distinct elements in S
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Check if any y in S divides sum of any 3 distinct elements in S U {x}
        # New triples: {x, a, b} with a, b in S, a != b
        if len(S) >= 2:
            for y in S:
                for i in range(len(S)):
                    a = S[i]
                    for j in range(i + 1, len(S)):
                        b = S[j]
                        if (x + a + b) % y == 0:
                            return False
                            
        return True

    def greedy(order):
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
                if len(S) == 48: # Max possible
                    break
        return S

    best_set = []
    
    # Try descending first
    best_set = greedy(list(range(n, 0, -1)))
    
    # Try ascending
    s = greedy(list(range(1, n + 1)))
    if len(s) > len(best_set):
        best_set = s
        
    if len(best_set) == 48:
        return sorted(best_set)

    # Random restarts
    # Estimate time for one greedy run:
    # ~192 candidates.
    # Average S size ~20.
    # can_add cost ~ 20^3 = 8000 ops.
    # 192 * 8000 = 1.5M ops.
    # Python speed ~10M-50M ops/sec?
    # So ~0.1s - 0.5s per run.
    # We have ~80s. Can do ~100-500 runs.
    
    while time.time() < deadline - 1:
        random_order = universe[:]
        random.shuffle(random_order)
        s = greedy(random_order)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                break

    # Final polish: try to add any remaining elements to best_set
    S = set(best_set)
    S_list = list(S)
    # Sort remaining candidates descending (larger are safer)
    remaining = [x for x in range(n, 0, -1) if x not in S]
    for x in remaining:
        if can_add(x, S_list):
            S.add(x)
            S_list.append(x)
            if len(S) == 48:
                break

    best_set = list(S)
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
