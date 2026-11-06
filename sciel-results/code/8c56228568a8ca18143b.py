import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    # Equation: 1/a = 1/b + 1/c => c = ab / (b-a)
    # Assume a < b. Then c = a + a^2 / (b-a).
    # Let d = b-a. d must divide a^2.
    # b = a + d, c = a + a^2/d.
    # Conditions: b <= N, c <= N, b != c, a, b, c distinct.
    
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a2 = a * a
        # d must be a divisor of a^2
        # b = a + d <= N => d <= N - a
        max_d = N - a
        for d in range(1, max_d + 1):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if c <= N and b != c:
                    # a, b, c are distinct because c > a and b > a, and b != c
                    t = tuple(sorted((a, b, c)))
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)
                        
    num_triples = len(triples)
    
    # 2. Smart Initialization: Start with [87, 180] and try to add smaller elements
    def run_smart_greedy():
        # Base set: [87, 180] is known to be valid (no distinct triples)
        active = set(range(87, N + 1))
        
        # Try to add elements from 86 down to 1
        for x in range(86, 0, -1):
            # Check if x can be added
            conflict = False
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                # If the other two elements are in active, we have a conflict
                others = [y for y in (a, b, c) if y != x]
                if all(y in active for y in others):
                    conflict = True
                    break
            
            if not conflict:
                active.add(x)
            else:
                # Try to resolve conflict by removing one of the conflicting elements
                # We prefer to remove the smaller element to keep the set "heavy"
                # Or we can just skip x for now
                pass
        
        return active

    # 3. Simulated Annealing
    def simulate_annealing(initial_set, time_budget):
        present = [False] * N
        for x in initial_set:
            present[x-1] = True
            
        triple_violation = [0] * num_triples
        total_violations = 0
        
        for idx, (a, b, c) in enumerate(triples):
            if present[a-1] and present[b-1] and present[c-1]:
                triple_violation[idx] = 1
                total_violations += 1
                
        def calc_score():
            return sum(present) - 1000 * total_violations
            
        current_score = calc_score()
        best_score = current_score
        best_present = present[:]
        
        T = 10.0
        cooling = 0.999
        start_sa = time.time()
        
        while time.time() - start_sa < time_budget:
            idx = random.randint(0, N - 1)
            elem = idx + 1
            old_val = present[idx]
            present[idx] = not old_val
            
            old_violations = total_violations
            for t_idx in elem_to_triples[elem]:
                a, b, c = triples[t_idx]
                was_violation = triple_violation[t_idx]
                is_violation = present[a-1] and present[b-1] and present[c-1]
                if was_violation and not is_violation:
                    total_violations -= 1
                    triple_violation[t_idx] = 0
                elif not was_violation and is_violation:
                    total_violations += 1
                    triple_violation[t_idx] = 1
            
            new_score = calc_score()
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_present = present[:]
            else:
                if T > 0.01 and random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    present[idx] = old_val
                    for t_idx in elem_to_triples[elem]:
                        a, b, c = triples[t_idx]
                        was_violation = triple_violation[t_idx]
                        is_violation = present[a-1] and present[b-1] and present[c-1]
                        if was_violation and not is_violation:
                            total_violations -= 1
                            triple_violation[t_idx] = 0
                        elif not was_violation and is_violation:
                            total_violations += 1
                            triple_violation[t_idx] = 1
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_present

    # 4. Cleanup
    def smart_cleanup(initial_set):
        active = set(initial_set)
        violations = []
        for idx, (a, b, c) in enumerate(triples):
            if a in active and b in active and c in active:
                violations.append((a, b, c))
        
        while violations:
            # Find element in max violations
            counts = {x: 0 for x in active}
            for (a, b, c) in violations:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            
            max_count = max(counts.values())
            candidates = [x for x in active if counts[x] == max_count]
            victim = min(candidates)
            active.remove(victim)
            
            new_violations = []
            for (a, b, c) in violations:
                if a != victim and b != victim and c != victim:
                    new_violations.append((a, b, c))
            violations = new_violations
            
        return active

    best_set = set()
    best_size = 0
    
    # Strategy: Run smart greedy, then SA on top of it
    seed_set = run_smart_greedy()
    best_size = len(seed_set)
    best_set = seed_set
    
    # Run SA
    if time.time() < deadline - 5:
        sa_budget = deadline - time.time() - 5
        best_present = simulate_annealing(seed_set, sa_budget)
        sa_set = {i+1 for i, val in enumerate(best_present) if val}
        sa_set = smart_cleanup(sa_set)
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = sa_set

    # Final cleanup
    final_set = smart_cleanup(best_set)
    return list(final_set)
