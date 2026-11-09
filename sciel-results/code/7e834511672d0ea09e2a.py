import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 90.0
    
    # 1. Generate all forbidden triples
    # a, b, c distinct, 1/a = 1/b + 1/c
    # a < b < c assumed for generation to avoid duplicates in logic, 
    # but the set check is symmetric.
    # Equation: c = a*b / (b-a)
    # Let b = a + k. c = a(a+k)/k = a^2/k + a.
    # k must divide a^2. k < a.
    
    triples = []
    for a in range(1, 181):
        for k in range(1, a):
            if (a * a) % k == 0:
                b = a + k
                c = a + (a * a) // k
                if b <= 180 and c <= 180 and b != c and a != b and a != c:
                    # Store as sorted tuple for consistency
                    triples.append(tuple(sorted((a, b, c))))
    
    # Remove duplicate triples if any (shouldn't be many, but safe)
    triples = list(set(triples))
    
    # 2. Calculate initial degrees
    degree = {i: 0 for i in range(1, 181)}
    for t in triples:
        for v in t:
            degree[v] += 1
            
    # 3. Greedy removal
    S = set(range(1, 181))
    
    # We will iteratively find a conflict and remove a vertex
    # To be efficient, we can just loop through triples
    # If a triple is fully in S, we have a conflict.
    
    # Optimization: Instead of checking all triples every time, 
    # we can just pick one and resolve.
    # Given small N, checking all is fine.
    
    max_iterations = 1000
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
        conflict_found = False
        for t in triples:
            if all(v in S for v in t):
                conflict_found = True
                # Find vertex with max degree in this triple
                # Use current degree or static? Static is faster.
                # Dynamic is better. Let's use static first, it's simpler.
                v_to_remove = max(t, key=lambda v: degree[v])
                S.remove(v_to_remove)
                # Update degree? 
                # If we update degree, we must decrement degree of neighbors.
                # Let's stick to static degree for simplicity, 
                # it's a good heuristic.
                break
        
        if not conflict_found:
            break
            
    # 4. Repair: Try to add back vertices
    # Sort vertices by degree (ascending) to add "safer" ones first?
    # Or just iterate 1..180.
    for v in range(1, 181):
        if time.time() > deadline - 2:
            break
        if v not in S:
            # Check if adding v creates a conflict
            # A conflict exists if there is a triple (v, u, w) where u, w in S
            safe = True
            for t in triples:
                if v in t:
                    # Check other two
                    others = [x for x in t if x != v]
                    if all(x in S for x in others):
                        safe = False
                        break
            if safe:
                S.add(v)
                
    # 5. Local Search / Hill Climbing
    # Try to swap: remove v from S, add u not in S if it results in net gain?
    # Or just try to add any missing vertex again?
    # The repair step above adds vertices one by one.
    # It might be possible that adding u requires removing v, but adding v allows adding w.
    # A simple local search:
    # While improvement possible:
    #   Try to add a vertex. If conflict, try to remove a conflicting vertex 
    #   if the removed vertex has lower "value" (degree) than the added one?
    #   This is complex.
    
    # Let's try a few random restarts of the greedy phase if time permits.
    # But the static degree greedy is deterministic.
    # Let's try randomizing the tie-breaking in removal.
    
    best_S = S
    best_size = len(S)
    
    # Try a few random perturbations if time allows
    # Actually, let's just run the greedy with random tie-breaking a few times.
    
    for _ in range(10): # Max 10 restarts
        if time.time() > deadline - 10:
            break
            
        S = set(range(1, 181))
        # Shuffle triples to change order of conflict resolution?
        # Or shuffle tie-breaking.
        shuffled_triples = triples[:]
        random.shuffle(shuffled_triples)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            conflict_found = False
            for t in shuffled_triples:
                if all(v in S for v in t):
                    conflict_found = True
                    # Random tie-break among max degree nodes?
                    # Or just pick one randomly from the triple?
                    # Picking max degree is better.
                    # If multiple have same max degree, pick random.
                    max_deg = max(degree[v] for v in t)
                    candidates = [v for v in t if degree[v] == max_deg]
                    v_to_remove = random.choice(candidates)
                    S.remove(v_to_remove)
                    break
            
            if not conflict_found:
                break
        
        # Repair
        for v in range(1, 181):
            if v not in S:
                safe = True
                for t in triples:
                    if v in t:
                        others = [x for x in t if x != v]
                        if all(x in S for x in others):
                            safe = False
                            break
                if safe:
                    S.add(v)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return list(best_S)
