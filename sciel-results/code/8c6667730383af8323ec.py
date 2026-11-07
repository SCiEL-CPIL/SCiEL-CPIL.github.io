import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_S = set(range(N // 2 + 1, N + 1))
    best_size = len(best_S)
    
    # Helper to check and add, operating on mutable state passed as args
    # to avoid closure issues and allow reuse
    def try_greedy():
        S = set(range(N // 2 + 1, N + 1))
        in_S = [False] * (N + 1)
        for x in S:
            in_S[x] = True
            
        sum_pairs = []
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                u, v = S_list[i], S_list[j]
                if in_S[u + v]:
                    sum_pairs.append((u, v))

        def can_add(x):
            for u, v in sum_pairs:
                if in_S[x + u] and in_S[x + v]:
                    return False
            return True

        def add(x):
            S.add(x)
            in_S[x] = True
            for u in S:
                if u == x: continue
                a, b = (x, u) if x < u else (u, x)
                if in_S[a + b]:
                    sum_pairs.append((a, b))

        # Pass 1: Random order of smalls
        candidates = list(range(1, N // 2 + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x):
                add(x)

        # Pass 2: Random order of all, try to add any missing
        for _ in range(20):
            progress = False
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if x not in S and can_add(x):
                    add(x)
                    progress = True
            if not progress:
                break
                
        return S

    # Run multiple restarts
    while time.time() < deadline:
        S = try_greedy()
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        # If we hit a very high size, we might want to break or continue?
        # Continue to try for even better.
        
    return list(best_S)
