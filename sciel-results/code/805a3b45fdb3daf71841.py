import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    N = 180

    # Precompute forbidden triples and neighbors
    # triples_with[x] = list of (y, z) such that (x, y, z) is a forbidden triple.
    triples_with = {i: [] for i in range(1, N+1)}
    
    for a in range(1, N+1):
        a2 = a*a
        for d in range(1, a):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    # Triple (a, b, c)
                    # For a: neighbors are b, c
                    # For b: neighbors are a, c
                    # For c: neighbors are a, b
                    triples_with[a].append((b, c))
                    triples_with[b].append((a, c))
                    triples_with[c].append((a, b))

    def is_safe(S, x):
        """Check if x can be added to S without forming a triple."""
        Sset = set(S)
        for y, z in triples_with[x]:
            if y in Sset and z in Sset:
                return False
        return True

    # Identify roots (smallest element in any triple)
    root_set = set()
    for a in range(1, N+1):
        a2 = a*a
        for d in range(1, a):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    root_set.add(a)
                    break

    base = set(range(1, N+1)) - root_set
    S = set(base)

    # Try to add roots greedily
    for x in sorted(root_set):
        if time.time() > deadline:
            break
        if is_safe(S, x):
            S.add(x)

    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Add safe elements
        for x in range(1, N+1):
            if time.time() > deadline:
                break
            if x not in S and is_safe(S, x):
                S.add(x)
                improved = True
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Pass 2: Swap (Remove u, Add v)
        # Try to remove one element and add more than one.
        for u in list(S):
            if time.time() > deadline:
                break
            S_temp = S - {u}
            added_count = 0
            for v in range(1, N+1):
                if v in S_temp:
                    continue
                if is_safe(S_temp, v):
                    S_temp.add(v)
                    added_count += 1
            
            if added_count > 1:
                S = S_temp
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break # Restart loop after change

    # Random Perturbation
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        S_pert = set(best_S)
        k = random.randint(1, 3)
        for _ in range(k):
            if S_pert:
                S_pert.discard(random.choice(list(S_pert)))
        
        # Re-add safe elements
        for v in range(1, N+1):
            if time.time() > deadline:
                break
            if v not in S_pert and is_safe(S_pert, v):
                S_pert.add(v)
                
        # Local search on S_pert
        improved_pert = True
        while improved_pert and time.time() < deadline:
            improved_pert = False
            for v in range(1, N+1):
                if time.time() > deadline:
                    break
                if v not in S_pert and is_safe(S_pert, v):
                    S_pert.add(v)
                    improved_pert = True
            
            # Swap
            for u in list(S_pert):
                if time.time() > deadline:
                    break
                S_temp = S_pert - {u}
                added_count = 0
                for w in range(1, N+1):
                    if w in S_temp:
                        continue
                    if is_safe(S_temp, w):
                        S_temp.add(w)
                        added_count += 1
                if added_count > 1:
                    S_pert = S_temp
                    improved_pert = True
                    break
        
        if len(S_pert) > best_size:
            best_S = S_pert
            best_size = len(S_pert)

    return list(best_S)
