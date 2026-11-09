"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all conflict triples: (a, b, c) where 1/a = 1/b + 1/c
    # a = k*m*n, b = k*m*(m+n), c = k*n*(m+n), gcd(m,n)=1
    triples = set()
    for k in range(1, 181):
        for m in range(1, 181):
            for n in range(m + 1, 181):
                if _gcd(m, n) != 1:
                    continue
                a = k * m * n
                b = k * m * (m + n)
                c = k * n * (m + n)
                if a <= N and b <= N and c <= N:
                    if len({a, b, c}) == 3:
                        triples.add((min(a,b,c), min(max(a,b), max(b,c)), max(a,b,c)))
    
    triples_list = list(triples)
    num_triples = len(triples_list)
    
    # Build element_triples: for each element, list of triple indices it appears in
    element_triples = {i: [] for i in range(1, N+1)}
    for idx, (a, b, c) in enumerate(triples_list):
        element_triples[a].append(idx)
        element_triples[b].append(idx)
        element_triples[c].append(idx)
    
    # Precompute for fast violation checking
    # For each triple, store the three elements
    triple_a = [t[0] for t in triples_list]
    triple_b = [t[1] for t in triples_list]
    triple_c = [t[2] for t in triples_list]
    
    # Initial seed: upper half
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Verify initial
    for (a, b, c) in triples_list:
        if a in best_set and b in best_set and c in best_set:
            best_set = set()
            best_size = 0
            break
    
    # Greedy construction with multiple orderings
    random.seed(42)
    
    def greedy_build_fast(order):
        result = set()
        for x in order:
            can_add = True
            for idx in element_triples[x]:
                a, b, c = triple_a[idx], triple_b[idx], triple_c[idx]
                if x == a:
                    if b in result and c in result:
                        can_add = False
                        break
                elif x == b:
                    if a in result and c in result:
                        can_add = False
                        break
                else:  # x == c
                    if a in result and b in result:
                        can_add = False
                        break
            if can_add:
                result.add(x)
        return result
    
    # Try different orderings
    orderings = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        list(range(91, N+1)) + list(range(1, 91)),
        list(range(121, N+1)) + list(range(61, 121)) + list(range(1, 61)),
    ]
    
    for _ in range(30):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        s = greedy_build_fast(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add elements
    def try_add(s, x):
        for idx in element_triples[x]:
            a, b, c = triple_a[idx], triple_b[idx], triple_c[idx]
            if x == a:
                if b in s and c in s:
                    return False
            elif x == b:
                if a in s and c in s:
                    return False
            else:
                if a in s and b in s:
                    return False
        return True
    
    # Add all possible elements
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for x in range(1, N+1):
            if x not in best_set and try_add(best_set, x):
                best_set.add(x)
                best_size = len(best_set)
                improved = True
    
    # Simulated annealing on Max-SAT formulation
    # Objective: maximize number of elements in set (equivalently, minimize violations)
    # Use membership array for speed
    membership = [False] * (N + 1)
    for x in best_set:
        membership[x] = True
    
    current_size = len(best_set)
    
    # Count violations
    def count_violations(memb):
        count = 0
        for i in range(num_triples):
            if memb[triple_a[i]] and memb[triple_b[i]] and memb[triple_c[i]]:
                count += 1
        return count
    
    current_violations = count_violations(membership)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    max_iterations = 500000
    
    best_sa_size = current_size
    best_sa_membership = membership[:]
    best_sa_violations = current_violations
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Pick a random element to flip
        x = random.randint(1, N)
        
        # Compute delta in violations
        old_viol = 0
        new_viol = 0
        for idx in element_triples[x]:
            a, b, c = triple_a[idx], triple_b[idx], triple_c[idx]
            # Current state
            if membership[a] and membership[b] and membership[c]:
                old_viol += 1
            # New state (flip x)
            if (not membership[a] if x == a else membership[a]) and \
               (not membership[b] if x == b else membership[b]) and \
               (not membership[c] if x == c else membership[c]):
                new_viol += 1
        
        delta_viol = new_viol - old_viol
        delta_size = 1 if not membership[x] else -1
        
        # Acceptance: prefer fewer violations, then more size
        # Energy = violations * 100 - size (penalize violations heavily)
        old_energy = old_viol * 100 - current_size
        new_energy = new_viol * 100 - (current_size + delta_size)
        delta_energy = new_energy - old_energy
        
        if delta_energy <= 0 or random.random() < math.exp(-delta_energy / T):
            membership[x] = not membership[x]
            current_violations += delta_viol
            current_size += delta_size
            
            if current_violations == 0 and current_size > best_sa_size:
                best_sa_size = current_size
                best_sa_membership = membership[:]
                best_sa_violations = 0
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Update best if SA found better
    if best_sa_violations == 0 and best_sa_size > best_size:
        best_size = best_sa_size
        best_set = set(i for i in range(1, N+1) if best_sa_membership[i])
    
    # Final local search: try to add any element that doesn't create violations
    final_membership = best_sa_membership[:]
    final_size = best_sa_size
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for x in range(1, N+1):
            if not final_membership[x]:
                can_add = True
                for idx in element_triples[x]:
                    a, b, c = triple_a[idx], triple_b[idx], triple_c[idx]
                    if final_membership[a] and final_membership[b] and final_membership[c]:
                        can_add = False
                        break
                if can_add:
                    final_membership[x] = True
                    final_size += 1
                    improved = True
    
    if final_size > best_size:
        best_size = final_size
        best_set = set(i for i in range(1, N+1) if final_membership[i])
    
    # Return the best set found
    result = list(best_set)
    
    # Verify validity
    for (a, b, c) in triples_list:
        if a in result and b in result and c in result:
            result.remove(a)
            break
    
    return result if result else [180]

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# EVOLVE-BLOCK-END
