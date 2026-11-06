import time
import random

def _precompute_triples(N):
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, min(2 * a, N + 1)):
            denom = b - a
            if (a * b) % denom == 0:
                c = (a * b) // denom
                if c > b and c <= N:
                    triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 180
    triples = _precompute_triples(N)
    
    element_triples = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(triples):
        element_triples[a].append(i)
        element_triples[b].append(i)
        element_triples[c].append(i)
        
    best_size = 0
    best_S = set()
    
    if not triples:
        return list(range(1, N + 1))
        
    iter_count = 0
    while time.time() < deadline:
        rng = random.Random(iter_count)
        
        S = set(range(1, N + 1))
        active = [True] * len(triples)
        active_count = len(triples)
        
        conflict_count = [0] * (N + 1)
        for i, (a, b, c) in enumerate(triples):
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
            
        while active_count > 0:
            max_count = -1
            candidates = []
            for x in S:
                if conflict_count[x] > max_count:
                    max_count = conflict_count[x]
                    candidates = [x]
                elif conflict_count[x] == max_count:
                    candidates.append(x)
                    
            if max_count == 0:
                break
                
            x = rng.choice(candidates)
            S.remove(x)
            
            for i in element_triples[x]:
                if active[i]:
                    active[i] = False
                    active_count -= 1
                    a, b, c = triples[i]
                    if a != x and a in S:
                        conflict_count[a] -= 1
                    if b != x and b in S:
                        conflict_count[b] -= 1
                    if c != x and c in S:
                        conflict_count[c] -= 1
                            
        # Polish Step
        R = set(range(1, N + 1)) - S
        # Try to add elements back
        # We can iterate R in random order to break symmetries
        r_list = list(R)
        rng.shuffle(r_list)
        
        for x in r_list:
            # Check if x can be added
            can_add = True
            for i in element_triples[x]:
                a, b, c = triples[i]
                others = [y for y in (a, b, c) if y != x]
                # If both others are in S, conflict
                if others[0] in S and others[1] in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                R.remove(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
        iter_count += 1
        if iter_count % 1000 == 0:
            pass

    return list(best_S)
