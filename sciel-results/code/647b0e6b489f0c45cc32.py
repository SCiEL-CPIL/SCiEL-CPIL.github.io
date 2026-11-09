import random
import time

N = 256

def is_sidon(S):
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    # Compute blocked set B
    # y is blocked if y+a in sums for some a in S, or 2y in sums
    blocked = set()
    for s in sums:
        for a in S:
            y = s - a
            if 1 <= y <= N:
                blocked.add(y)
        if s % 2 == 0:
            y = s // 2
            if 1 <= y <= N:
                blocked.add(y)
                
    # Check if all y not in S are blocked
    for y in range(1, N + 1):
        if y in S_set:
            continue
        if y not in blocked:
            return False
    return True

def greedy_sidon(order, N):
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def reduce_maximal(S, N):
    S = list(S)
    improved = True
    while improved:
        improved = False
        S_set = set(S)
        sums = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                sums.add(a + S[j])
        
        # Try removals
        candidates = list(S)
        random.shuffle(candidates)
        for x in candidates:
            candidate_S = [s for s in S if s != x]
            if is_maximal(candidate_S, N):
                S = candidate_S
                improved = True
                break
        if improved:
            continue
            
        # Try swaps
        # Limit swaps to avoid slowness
        for x in S:
            # Try 5 random swaps
            for _ in range(5):
                y = random.randint(1, N)
                if y in S_set:
                    continue
                candidate_S = [s for s in S if s != x] + [y]
                if not is_sidon(candidate_S):
                    continue
                if is_maximal(candidate_S, N):
                    S = candidate_S
                    S_set = set(S)
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 25  # Safety margin
    
    best = None
    best_size = N + 1
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = reduce_maximal(S, N)
    if len(S) < best_size:
        best = S
        best_size = len(S)
        
    random.seed(42)
    
    # Try multiple random orderings
    for trial in range(5000):
        if time.time() > deadline:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Reduce
        if time.time() < deadline - 2:
            S = reduce_maximal(S, N)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
    if best is None:
        best = greedy_sidon(list(range(1, N + 1)), N)
        
    return best
