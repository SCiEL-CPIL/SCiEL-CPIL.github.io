import time
import random
import math

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    forbidden_triples = []
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    forbidden_triples.append((a, b, c))
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Bitset representation
    def make_bitset(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        return bits
    
    def bitset_has(bits, x):
        return (bits >> x) & 1
    
    def can_add_bitset(x, bits):
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                return False
        return True

    def init_solution(seed):
        random.seed(seed)
        S = set(range(91, N + 1))
        for x in range(1, 91):
            if can_add_bitset(x, make_bitset(S)):
                S.add(x)
        return S

    best_S = None
    best_size = 0
    
    for seed in [42, 123, 456, 789, 321]:
        S = init_solution(seed)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    all_nums = set(range(1, N + 1))
    random.seed(42)
    
    S = set(best_S)
    bits = make_bitset(S)
    
    # Phase 1: Quick local search to polish
    for iteration in range(50000):
        if time.time() > deadline - 20:
            break
        excluded = all_nums - S
        if not excluded:
            break
        x = random.choice(tuple(excluded))
        conflicts = []
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                conflicts.append((y, z))
        if not conflicts:
            S.add(x)
            bits |= (1 << x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        candidates = set()
        for y, z in conflicts:
            candidates.add(y)
            candidates.add(z)
        cand_list = list(candidates)
        random.shuffle(cand_list)
        found_swap = False
        for u in cand_list:
            S.remove(u)
            bits ^= (1 << u)
            if can_add_bitset(x, bits):
                S.add(x)
                bits |= (1 << x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                found_swap = True
                break
            else:
                S.add(u)
                bits |= (1 << u)
        if found_swap:
            continue

    # Phase 2: Weighted Max-SAT with dynamic clause weighting
    # Each variable x in [1, N] has a "value" of 1 if in set, 0 otherwise
    # Each forbidden triple (a,b,c) is a clause: NOT (a AND b AND c)
    # We maximize sum(values) - sum(weights * violations)
    # Dynamic weights: increase weight for triples that are frequently violated
    
    # Precompute for each element, which triples it participates in
    element_triples = [list() for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(forbidden_triples):
        element_triples[a].append(idx)
        element_triples[b].append(idx)
        element_triples[c].append(idx)
    
    # Dynamic weights for each clause
    clause_weights = [1.0] * len(forbidden_triples)
    # Track how often each clause is "active" (all three elements in set)
    clause_activity = [0] * len(forbidden_triples)
    
    # Current state
    in_set = [False] * (N + 1)
    for x in S:
        in_set[x] = True
    
    # Compute current score: |S| - sum(weights * violation)
    def compute_score():
        size = sum(in_set)
        violations = 0.0
        for idx, (a, b, c) in enumerate(forbidden_triples):
            if in_set[a] and in_set[b] and in_set[c]:
                violations += clause_weights[idx]
        return size - violations
    
    current_score = compute_score()
    best_score = current_score
    best_S_copy = set(S)
    
    # Simulated annealing with dynamic weights
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    weight_update_rate = 0.05
    weight_decay = 0.99
    
    for iteration in range(500000):
        if time.time() > deadline - 2:
            break
        if T < T_min:
            break
            
        # Choose a move
        r = random.random()
        if r < 0.5:
            # Try to add an element
            x = random.randint(1, N)
            if not in_set[x]:
                # Check if adding x creates any violations
                new_violations = 0.0
                would_violate = False
                for tidx in element_triples[x]:
                    a, b, c = forbidden_triples[tidx]
                    other1, other2 = (a, b) if x == c else ((a, c) if x == b else (b, c))
                    if in_set[other1] and in_set[other2]:
                        new_violations += clause_weights[tidx]
                        would_violate = True
                
                if not would_violate:
                    in_set[x] = True
                    bits |= (1 << x)
                    S.add(x)
                    new_score = current_score + 1.0
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S_copy = set(S)
                else:
                    # Accept with probability exp(-new_violations / T)
                    if random.random() < math.exp(-new_violations / T):
                        in_set[x] = True
                        bits |= (1 << x)
                        S.add(x)
                        current_score += 1.0 - new_violations
                        if current_score > best_score:
                            best_score = current_score
                            best_S_copy = set(S)
        elif r < 0.8:
            # Try to remove an element
            x = random.randint(1, N)
            if in_set[x]:
                # Compute how many violations are removed
                removed_violations = 0.0
                for tidx in element_triples[x]:
                    a, b, c = forbidden_triples[tidx]
                    other1, other2 = (a, b) if x == c else ((a, c) if x == b else (b, c))
                    if in_set[other1] and in_set[other2]:
                        removed_violations += clause_weights[tidx]
                
                in_set[x] = False
                bits ^= (1 << x)
                S.remove(x)
                new_score = current_score - 1.0 + removed_violations
                # Accept with probability 1 if improvement, else exp(-(new_score - current_score)/T)
                if new_score >= current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S_copy = set(S)
                else:
                    if random.random() < math.exp(-(current_score - new_score) / T):
                        current_score = new_score
                    else:
                        # Revert
                        in_set[x] = True
                        bits |= (1 << x)
                        S.add(x)
        else:
            # Try a swap: remove u, add v
            u = random.randint(1, N)
            v = random.randint(1, N)
            if in_set[u] and not in_set[v]:
                # Compute score change
                # Remove u: lose 1 from size, gain back violations that u was part of
                removed_violations_u = 0.0
                for tidx in element_triples[u]:
                    a, b, c = forbidden_triples[tidx]
                    other1, other2 = (a, b) if u == c else ((a, c) if u == b else (b, c))
                    if in_set[other1] and in_set[other2]:
                        removed_violations_u += clause_weights[tidx]
                
                # Add v: gain 1 from size, may create new violations
                new_violations_v = 0.0
                for tidx in element_triples[v]:
                    a, b, c = forbidden_triples[tidx]
                    other1, other2 = (a, b) if v == c else ((a, c) if v == b else (b, c))
                    if in_set[other1] and in_set[other2]:
                        new_violations_v += clause_weights[tidx]
                
                delta = 1.0 - 1.0 + removed_violations_u - new_violations_v
                if delta >= 0:
                    in_set[u] = False
                    in_set[v] = True
                    bits ^= (1 << u)
                    bits |= (1 << v)
                    S.remove(u)
                    S.add(v)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S_copy = set(S)
                else:
                    if random.random() < math.exp(-delta / T):
                        in_set[u] = False
                        in_set[v] = True
                        bits ^= (1 << u)
                        bits |= (1 << v)
                        S.remove(u)
                        S.add(v)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S_copy = set(S)
        
        # Update dynamic weights every 100 iterations
        if iteration % 100 == 0:
            # Decay weights
            for i in range(len(clause_weights)):
                clause_weights[i] *= weight_decay
            
            # Increase weights for clauses that are currently violated
            for idx, (a, b, c) in enumerate(forbidden_triples):
                if in_set[a] and in_set[b] and in_set[c]:
                    clause_weights[idx] += weight_update_rate
                    clause_activity[idx] += 1
                else:
                    clause_activity[idx] = max(0, clause_activity[idx] - 1)
        
        T *= cooling
    
    # Phase 3: Final greedy polish on the best solution found
    S = set(best_S_copy)
    bits = make_bitset(S)
    excluded = all_nums - S
    for x in sorted(excluded, key=lambda x: -len(conflict_pairs[x])):
        if can_add_bitset(x, bits):
            S.add(x)
            bits |= (1 << x)
    
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
    
    return sorted(best_S)
