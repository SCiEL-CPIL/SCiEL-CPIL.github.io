import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    raw_triples = set()
    for a in range(1, N + 1):
        a2 = a * a
        limit = int(math.isqrt(a2))
        for u in range(1, limit + 1):
            if a2 % u == 0:
                v = a2 // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b <= N and c <= N:
                    t = tuple(sorted((a, b, c)))
                    raw_triples.add(t)

    triples = list(raw_triples)
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(idx)
            
    num_triples = len(triples)
    
    # 2. Greedy Initialization with better tie-breaking
    def run_greedy(strategy='random'):
        active_elements = set(elements)
        active_triples = set(range(num_triples))
        conflict_count = {i: 0 for i in range(1, N + 1)}
        
        for idx in active_triples:
            for x in triples[idx]:
                conflict_count[x] += 1
        
        while True:
            # Find element with max conflict count
            max_conflict = -1
            candidates = []
            for x in active_elements:
                if conflict_count[x] > max_conflict:
                    max_conflict = conflict_count[x]
                    candidates = [x]
                elif conflict_count[x] == max_conflict:
                    candidates.append(x)
            
            if max_conflict == 0:
                break
            
            # Select victim based on strategy
            if strategy == 'random' and len(candidates) > 1:
                victim = random.choice(candidates)
            elif strategy == 'min':
                victim = min(candidates)
            elif strategy == 'max':
                victim = max(candidates)
            elif strategy == 'degree':
                # Pick element with highest degree in conflict graph
                # Approximate by conflict_count * (number of active neighbors)
                best_victim = None
                best_score = -1
                for x in candidates:
                    # Count how many active elements are in conflicts with x
                    deg = 0
                    for idx in elem_to_triples[x]:
                        if idx in active_triples:
                            a, b, c = triples[idx]
                            for y in (a, b, c):
                                if y != x and y in active_elements:
                                    deg += 1
                    score = conflict_count[x] * 10 + deg
                    if score > best_score:
                        best_score = score
                        best_victim = x
                victim = best_victim
            else:
                victim = min(candidates)

            active_elements.remove(victim)
            
            # Update active_triples and conflict_counts
            for idx in elem_to_triples[victim]:
                if idx in active_triples:
                    active_triples.remove(idx)
                    for x in triples[idx]:
                        if x != victim and x in active_elements:
                            conflict_count[x] -= 1
        
        return active_elements

    # 3. Simulated Annealing with incremental violation tracking
    def simulate_annealing(initial_set, time_budget):
        # State: list of booleans, index i corresponds to element i+1
        present = [False] * N
        for x in initial_set:
            present[x-1] = True
        
        # Precompute for each element, which triples it participates in
        elem_triples = {i: [] for i in range(1, N + 1)}
        for idx, (a, b, c) in enumerate(triples):
            elem_triples[a].append(idx)
            elem_triples[b].append(idx)
            elem_triples[c].append(idx)
        
        # Precompute for each triple, the indices of its three elements (0-based)
        triple_elems = [(a-1, b-1, c-1) for (a, b, c) in triples]
        
        # Track violation count for each triple
        triple_violation = [0] * num_triples  # 1 if all three present, 0 otherwise
        total_violations = 0
        
        # Initialize violation counts
        for idx in range(num_triples):
            i1, i2, i3 = triple_elems[idx]
            if present[i1] and present[i2] and present[i3]:
                triple_violation[idx] = 1
                total_violations += 1
        
        current_size = sum(present)
        current_score = current_size - 1000 * total_violations
        best_score = current_score
        best_present = present[:]
        
        T = 15.0
        cooling = 0.9998
        start_sa = time.time()
        
        # Fast violation check for a single element flip
        def check_flip_delta(elem_idx):
            """Returns delta_violations if we flip elem_idx (0-based)"""
            elem = elem_idx + 1
            new_val = not present[elem_idx]
            delta = 0
            for t_idx in elem_triples[elem]:
                i1, i2, i3 = triple_elems[t_idx]
                was_violation = triple_violation[t_idx]
                # Check if flipping this element changes violation status
                if new_val:
                    # Adding: violation if other two are present
                    others_present = (i1 != elem_idx and present[i1]) and (i2 != elem_idx and present[i2]) and (i3 != elem_idx and present[i3])
                    is_violation = others_present
                else:
                    # Removing: violation was broken
                    is_violation = False
                if was_violation and not is_violation:
                    delta -= 1
                elif not was_violation and is_violation:
                    delta += 1
            return delta, new_val

        # Fast apply for a single element flip
        def apply_single_flip(elem_idx):
            """Apply a single flip and update state. Returns delta_violations"""
            nonlocal total_violations, current_size
            elem = elem_idx + 1
            new_val = not present[elem_idx]
            present[elem_idx] = new_val
            if new_val:
                current_size += 1
            else:
                current_size -= 1
            
            delta = 0
            for t_idx in elem_triples[elem]:
                i1, i2, i3 = triple_elems[t_idx]
                was_violation = triple_violation[t_idx]
                if new_val:
                    others_present = (i1 != elem_idx and present[i1]) and (i2 != elem_idx and present[i2]) and (i3 != elem_idx and present[i3])
                    is_violation = others_present
                else:
                    is_violation = False
                if was_violation and not is_violation:
                    total_violations -= 1
                    triple_violation[t_idx] = 0
                    delta -= 1
                elif not was_violation and is_violation:
                    total_violations += 1
                    triple_violation[t_idx] = 1
                    delta += 1
            return delta

        # Revert a single flip
        def revert_single_flip(elem_idx):
            """Revert a single flip and update state"""
            nonlocal total_violations, current_size
            elem = elem_idx + 1
            old_val = not present[elem_idx]
            present[elem_idx] = old_val
            if old_val:
                current_size += 1
            else:
                current_size -= 1
            
            for t_idx in elem_triples[elem]:
                i1, i2, i3 = triple_elems[t_idx]
                was_violation = triple_violation[t_idx]
                if old_val:
                    others_present = (i1 != elem_idx and present[i1]) and (i2 != elem_idx and present[i2]) and (i3 != elem_idx and present[i3])
                    is_violation = others_present
                else:
                    is_violation = False
                if was_violation and not is_violation:
                    total_violations -= 1
                    triple_violation[t_idx] = 0
                elif not was_violation and is_violation:
                    total_violations += 1
                    triple_violation[t_idx] = 1

        while time.time() - start_sa < time_budget:
            # Choose move type: 1-flip (85%), swap (10%), 2-flip (5%)
            move_type = random.random()
            
            if move_type < 0.85:
                # 1-flip
                idx = random.randint(0, N - 1)
                delta_v = apply_single_flip(idx)
                new_score = current_size - 1000 * total_violations
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
                        revert_single_flip(idx)
                        
            elif move_type < 0.95:
                # Swap: remove one present, add one absent
                present_idx = [i for i in range(N) if present[i]]
                absent_idx = [i for i in range(N) if not present[i]]
                if present_idx and absent_idx:
                    i1 = random.choice(present_idx)
                    i2 = random.choice(absent_idx)
                    
                    # Apply both flips
                    apply_single_flip(i1)
                    apply_single_flip(i2)
                    
                    new_score = current_size - 1000 * total_violations
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
                            revert_single_flip(i2)
                            revert_single_flip(i1)
            else:
                # 2-flip add: pick two absent elements
                absent = [i for i in range(N) if not present[i]]
                if len(absent) >= 2:
                    i1, i2 = random.sample(absent, 2)
                    apply_single_flip(i1)
                    apply_single_flip(i2)
                    
                    new_score = current_size - 1000 * total_violations
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
                            revert_single_flip(i2)
                            revert_single_flip(i1)
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_present

    # 4. Smart cleanup: remove minimum elements to fix all violations
    def smart_cleanup(initial_set):
        active = set(initial_set)
        
        # Find all violations
        violations = []
        for idx, (a, b, c) in enumerate(triples):
            if a in active and b in active and c in active:
                violations.append((a, b, c))
        
        if not violations:
            return active
        
        # Greedily remove elements to fix violations
        # For each element, count how many violations it participates in
        elem_viol_count = {x: 0 for x in active}
        for (a, b, c) in violations:
            elem_viol_count[a] += 1
            elem_viol_count[b] += 1
            elem_viol_count[c] += 1
        
        while violations:
            # Find element with max violation count
            max_count = -1
            candidates = []
            for x in active:
                if elem_viol_count[x] > max_count:
                    max_count = elem_viol_count[x]
                    candidates = [x]
                elif elem_viol_count[x] == max_count:
                    candidates.append(x)
            
            if max_count == 0:
                break
            
            # Remove element with max violation count (tie-break: smallest)
            victim = min(candidates)
            active.remove(victim)
            
            # Update violations
            new_violations = []
            for (a, b, c) in violations:
                if a != victim and b != victim and c != victim:
                    new_violations.append((a, b, c))
            violations = new_violations
            
            # Update violation counts
            for x in active:
                elem_viol_count[x] = 0
            for (a, b, c) in violations:
                elem_viol_count[a] += 1
                elem_viol_count[b] += 1
                elem_viol_count[c] += 1
        
        return active

    # 5. Local Search Polish: Try to add elements back
    def local_polish(initial_set):
        active = set(initial_set)
        # Build a fast check for "can I add x?"
        # For each x, check if adding x creates any violation.
        # A violation is created if there exists a triple (x, y, z) where y, z are in active.
        
        # Precompute for each x, the pairs (y, z) such that (x, y, z) is a triple.
        # Actually, we have elem_to_triples.
        # If we add x, we must ensure no triple (x, y, z) has y, z in active.
        
        changed = True
        while changed:
            changed = False
            # Try to add any element not in active
            for x in range(1, N + 1):
                if x in active:
                    continue
                # Check if adding x is safe
                safe = True
                for t_idx in elem_to_triples[x]:
                    a, b, c = triples[t_idx]
                    # The other two elements in the triple
                    others = [y for y in (a, b, c) if y != x]
                    if others[0] in active and others[1] in active:
                        safe = False
                        break
                if safe:
                    active.add(x)
                    changed = True
        return active

    # Strategy:
    # 1. Run multiple greedy passes with different strategies
    # 2. Run SA on top few seeds
    # 3. Local polish at the end
    
    best_set = set()
    best_size = 0
    greedy_results = []
    
    # Run greedy with different strategies - use less time for greedy
    strategies = ['random', 'min', 'max', 'degree']
    for strat in strategies:
        for i in range(10): # Reduced to allow more SA time
            if time.time() > deadline - 40:
                break
            s = run_greedy(strategy=strat)
            greedy_results.append((len(s), s))
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
    
    # Sort by size, take top 6 for SA
    greedy_results.sort(key=lambda x: -x[0])
    top_seeds = [s for _, s in greedy_results[:6]]
    
    # Run SA on top seeds with more time
    time_elapsed = time.time() - start_time
    if time_elapsed < deadline - 5:
        sa_budget = (deadline - time_elapsed - 5) / len(top_seeds)
        for seed in top_seeds:
            if time.time() > deadline - 5:
                break
            best_present = simulate_annealing(seed, sa_budget)
            sa_set = {i+1 for i, val in enumerate(best_present) if val}
            # Clean up if needed
            sa_set = smart_cleanup(sa_set)
            # Polish
            sa_set = local_polish(sa_set)
            if len(sa_set) > best_size:
                best_size = len(sa_set)
                best_set = sa_set
    
    # Final smart cleanup and polish
    final_set = smart_cleanup(best_set)
    final_set = local_polish(final_set)

    return list(final_set)

# EVOLVE-BLOCK-END
