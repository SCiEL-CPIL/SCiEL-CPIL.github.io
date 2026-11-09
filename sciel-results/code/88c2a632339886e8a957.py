import random
import time

def construct():
    n = 180
    
    # 1. Generate all forbidden triples
    triples = []
    element_to_triples = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                if d >= e:
                    continue
                b = a + d
                c = a + e
                if b < c and b <= n and c <= n:
                    t = (a, b, c)
                    triples.append(t)
                    element_to_triples[a].append(t)
                    element_to_triples[b].append(t)
                    element_to_triples[c].append(t)
                    
    if not triples:
        return list(range(1, n + 1))
        
    # Precompute list of triples for faster access
    triple_list = triples
    
    best_solution = set(range(1, n + 1))
    best_size = len(best_solution)
    
    # Greedy Repair with Randomization
    num_iterations = 200
    start_time = time.time()
    time_limit = 55
    
    for _ in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        S = set(range(1, n + 1))
        
        # We will repeatedly find a violated triple and remove an element
        # To avoid infinite loops or slow scans, we just scan all triples to find one violation
        # Since we remove elements, the set of violations shrinks.
        
        while True:
            violated_triple = None
            # Randomize the order of triples to explore different paths
            # We don't want to shuffle the whole list every time, it's expensive.
            # Instead, we can just iterate and pick the first one found.
            # But to add randomness, we can start from a random index and wrap around.
            start_idx = random.randint(0, len(triple_list) - 1)
            found = False
            for i in range(len(triple_list)):
                idx = (start_idx + i) % len(triple_list)
                t = triple_list[idx]
                a, b, c = t
                if a in S and b in S and c in S:
                    violated_triple = t
                    found = True
                    break
            
            if not found:
                break
            
            a, b, c = violated_triple
            candidates = [a, b, c]
            random.shuffle(candidates)
            
            # Select candidate with highest "violation degree"
            # i.e., involved in the most other currently violated triples
            best_candidate = None
            max_count = -1
            
            for x in candidates:
                count = 0
                for t in element_to_triples[x]:
                    ta, tb, tc = t
                    if ta in S and tb in S and tc in S:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_candidate = x
            
            S.remove(best_candidate)
        
        if len(S) > best_size:
            best_size = len(S)
            best_solution = S.copy()
            
    return sorted(list(best_solution))
