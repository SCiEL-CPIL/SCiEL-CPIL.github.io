import math

def construct():
    N = 512
    # Precompute compatibility
    # compatible[i][j] = True if lcm(i, j) <= N
    # We can just check on the fly or precompute a bitmask.
    # 512 nodes. Bitmask is 512 bits = 64 bytes. Very fast.
    
    # Let's use a greedy approach.
    # Try to build a set S.
    # Sort candidates by some heuristic.
    # Heuristic 1: Ascending value.
    # Heuristic 2: Descending "conflict count".
    
    # Let's try a simple local search starting from [1..23].
    # Or just run a greedy with multiple restarts/orderings.
    
    def is_compatible(a, b):
        return (a * b) // math.gcd(a, b) <= N

    # Try greedy with ascending order
    S1 = []
    for x in range(1, N + 1):
        ok = True
        for y in S1:
            if not is_compatible(x, y):
                ok = False
                break
        if ok:
            S1.append(x)
            
    # Try greedy with descending order
    S2 = []
    for x in range(N, 0, -1):
        ok = True
        for y in S2:
            if not is_compatible(x, y):
                ok = False
                break
        if ok:
            S2.append(x)
            
    # Try greedy with random order (multiple times)
    import random
    best_set = max(S1, S2, key=len)
    
    for _ in range(100):
        random.seed(_)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S3 = []
        for x in candidates:
            ok = True
            for y in S3:
                if not is_compatible(x, y):
                    ok = False
                    break
            if ok:
                S3.append(x)
        if len(S3) > len(best_set):
            best_set = S3
            
    # Local search: try to swap elements or add/remove
    # If we have time, do a simple hill climbing.
    # But greedy might be enough.
    
    return best_set
