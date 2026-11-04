def _find_sidon_set(N):
    """Find a large subset of {0..N-1} with distinct sums."""
    best_S = []
    
    # Heuristic 1: Greedy from 0
    S1 = []
    used1 = set()
    for i in range(N):
        ok = True
        for j in S1:
            s1 = i + i
            s2 = i + j
            if s1 in used1 or s2 in used1:
                ok = False
                break
        if ok:
            used1.add(i+i)
            for j in S1:
                used1.add(i+j)
            S1.append(i)
    if len(S1) > len(best_S):
        best_S = S1

    # Heuristic 2: Greedy from N-1 downwards
    S2 = []
    used2 = set()
    for i in range(N-1, -1, -1):
        ok = True
        for j in S2:
            s1 = i + i
            s2 = i + j
            if s1 in used2 or s2 in used2:
                ok = False
                break
        if ok:
            used2.add(i+i)
            for j in S2:
                used2.add(i+j)
            S2.append(i)
    if len(S2) > len(best_S):
        best_S = S2
        
    # Heuristic 3: Randomized
    import random
    random.seed(42)
    for _ in range(200):
        S3 = []
        used3 = set()
        indices = list(range(N))
        random.shuffle(indices)
        for i in indices:
            ok = True
            for j in S3:
                s1 = i + i
                s2 = i + j
                if s1 in used3 or s2 in used3:
                    ok = False
                    break
            if ok:
                used3.add(i+i)
                for j in S3:
                    used3.add(i+j)
                S3.append(i)
        if len(S3) > len(best_S):
            best_S = S3
            
    return best_S

def construct():
    S = _find_sidon_set(N)
    K = len(S)
    if K > 0:
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0 / K
        return [float(x) for x in h]
    
    # Fallback
    h = np.ones(N)
    return [float(x) for x in h]
