import time
import random

def construct():
    n = 512
    best_size = 0
    best_set = set()
    start_time = time.time()
    deadline = start_time + 55.0
    
    def can_add(x, S):
        for d in range(1, (n - x) // 3 + 1):
            if x+d in S and x+2*d in S and x+3*d in S:
                return False
        for d in range(1, min(x-1, (n-x)//2) + 1):
            if x-d in S and x+d in S and x+2*d in S:
                return False
        for d in range(1, min(x-2, n-x) // 2 + 1):
            if x-2*d in S and x-d in S and x+d in S:
                return False
        for d in range(1, x // 3 + 1):
            if x-3*d in S and x-2*d in S and x-d in S:
                return False
        return True

    def greedy_random():
        S = set()
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        S = S.copy()
        for _ in range(20): # 20 rounds
            improved = False
            ys = list(S)
            random.shuffle(ys)
            for y in ys:
                if time.time() > deadline:
                    return S
                S.remove(y)
                added = 0
                comp = [x for x in range(1, n+1) if x not in S]
                random.shuffle(comp)
                for x in comp[:100]:
                    if can_add(x, S):
                        S.add(x)
                        added += 1
                if added > 1:
                    improved = True
                    break
                else:
                    S.add(y)
            if not improved:
                break
        return S

    # Phase 1: Greedy restarts
    top_sets = []
    for i in range(500):
        if time.time() > deadline:
            break
        S = greedy_random()
        top_sets.append(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Sort top sets by size
    top_sets.sort(key=lambda s: len(s), reverse=True)
    
    # Phase 2: Local search on top sets
    for S in top_sets[:20]:
        if time.time() > deadline:
            break
        S_improved = local_search(S)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved.copy()
            
    return sorted(best_set)
