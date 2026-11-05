import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  => a = bc / (b+c)
    # We store for each a, the list of pairs (b, c) that form a triple with a.
    triples_by_a = {i: [] for i in range(1, N+1)}
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triples_by_a[a].append((b, c))
    
    # Function to check if adding x to S is valid
    def is_valid_add(x, S):
        for (b, c) in triples_by_a[x]:
            if b in S and c in S:
                return False
        return True

    # Initial solution: {91..180}
    best_S = set(range(91, N+1))
    best_size = len(best_S)
    
    # Heuristic: Sort by number of conflicts
    conf_counts = {x: len(triples_by_a[x]) for x in range(1, N+1)}
    sorted_candidates = sorted(range(1, N+1), key=lambda x: conf_counts[x])
    
    # Try greedy with sorted order
    S = set()
    for x in sorted_candidates:
        if is_valid_add(x, S):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_S = S
        
    # Try greedy with reverse sorted order
    S = set()
    for x in reversed(sorted_candidates):
        if is_valid_add(x, S):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_S = S
        
    # Randomized Greedy
    # Try multiple times
    while time.time() < deadline:
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        S = set()
        for x in candidates:
            if is_valid_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return list(best_S)
