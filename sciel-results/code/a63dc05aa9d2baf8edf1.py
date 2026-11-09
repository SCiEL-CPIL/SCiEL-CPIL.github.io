import random
import time

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    def has_4ap_with(x, S):
        """Check if adding x to S creates a 4-AP."""
        # x as 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (x+d in S) and (x+2*d in S) and (x+3*d in S):
                return True
        # x as 2nd term: x-d, x, x+d, x+2d
        for d in range(1, min(x-1, (N-x)//2) + 1):
            if (x-d in S) and (x+d in S) and (x+2*d in S):
                return True
        # x as 3rd term: x-2d, x-d, x, x+d
        for d in range(1, min((x-1)//2, N-x) + 1):
            if (x-2*d in S) and (x-d in S) and (x+d in S):
                return True
        # x as 4th term: x-3d, x-2d, x-d, x
        for d in range(1, (x-1)//3 + 1):
            if (x-3*d in S) and (x-2*d in S) and (x-d in S):
                return True
        return False

    def base4_construction():
        """Numbers in [1,N] with only 0/1 in base-4 representation."""
        S = set()
        for mask in range(1, 1 << 9):  # up to 9 bits to cover 512
            val = 0
            for i in range(9):
                if mask & (1 << i):
                    val += 4 ** i
            if 1 <= val <= N:
                S.add(val)
        return S

    def greedy_random(S_init=None):
        S = set(S_init) if S_init else set()
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if x not in S and not has_4ap_with(x, S):
                S.add(x)
        return S

    def local_search(S, max_rounds=100):
        """Swap-based local search: try removing one and adding others."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline:
            improved = False
            rounds += 1
            elements = list(S)
            random.shuffle(elements)
            for u in elements:
                if time.time() > deadline:
                    return S
                S_temp = S.copy()
                S_temp.remove(u)
                # Try to add any element not in S_temp
                added = 0
                for v in range(1, N+1):
                    if v not in S_temp and not has_4ap_with(v, S_temp):
                        S_temp.add(v)
                        added += 1
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
                    break
        return S

    # Phase 1: Get a good initial set from base-4 construction
    best_set = base4_construction()
    best_size = len(best_set)
    
    # Phase 2: Randomized greedy with multiple restarts
    num_greedy = 50
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        # Mix: sometimes start from base4, sometimes from empty
        if i % 3 == 0:
            S = greedy_random(best_set.copy())
        else:
            S = greedy_random()
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 3: Local search to improve
    if time.time() < deadline:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Phase 4: Final polish - try to add any remaining element
    if time.time() < deadline:
        for v in range(1, N+1):
            if v not in best_set and not has_4ap_with(v, best_set):
                best_set.add(v)
    
    return sorted(best_set)
