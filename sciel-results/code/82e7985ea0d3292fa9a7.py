import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples using divisor method (Inspiration 1)
    # 1/a = 1/b + 1/c  =>  c = ab/(b-a)
    # Let b = a + u, c = a + v. Then uv = a^2.
    # We iterate a from 1 to N, find divisors of a^2, check bounds.
    
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a2 = a * a
        limit = int(math.isqrt(a2))
        divisors = []
        for d in range(1, limit + 1):
            if a2 % d == 0:
                divisors.append(d)
                if d != a2 // d:
                    divisors.append(a2 // d)
        
        for u in divisors:
            v = a2 // u
            if u == v:
                continue # b == c, not distinct
            b = a + u
            c = a + v
            if b <= N and c <= N:
                # Check distinctness: a, b, c are distinct because u, v > 0 and u != v
                # Also need to ensure we don't add duplicates.
                # Since we iterate a, and u,v are symmetric in the equation uv=a^2,
                # the pair (b,c) is the same as (c,b).
                # The triple is (a, b, c).
                # We should sort (b,c) to avoid duplicate triples if we were using sets,
                # but here we just need the constraint.
                # To be safe against duplicates from different (u,v) pairs generating same (b,c)?
                # No, u and v are uniquely determined by b-a and c-a.
                # So (a,b,c) is unique.
                
                # However, the equation is symmetric in b and c.
                # My loop generates (a, a+u, a+v).
                # If I swap u and v, I get (a, a+v, a+u) which is the same set.
                # I should only add if b < c to avoid duplicates?
                # Or just add all, duplicates in triples list just mean higher conflict count, which is fine for greedy, 
                # but for SA violations count, duplicates would overcount.
                # Let's deduplicate by sorting the triple.
                
                t = tuple(sorted((a, b, c)))
                # We need to check if this exact triple was already added.
                # Since N is small, we can use a set of tuples for precomputation.
                pass

    # Let's redo precomputation cleanly with a set to avoid duplicates
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
                # Also consider if u > v? 
                # The loop u from 1 to sqrt(a2) covers all factor pairs (u, v) with u <= v.
                # Since the equation is symmetric in b and c, we only need to generate each pair {b,c} once.
                # So this is sufficient.

    triples = list(raw_triples)
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(idx)
            
    num_triples = len(triples)
    
    # 2. Greedy Initialization
    def run_greedy(randomize_ties=False):
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
                break # No more conflicts
            
            # Select victim
            if randomize_ties and len(candidates) > 1:
                victim = random.choice(candidates)
            else:
                # Pick smallest to keep larger numbers? Or random?
                # Deterministic: pick smallest
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

    # 3. Simulated Annealing Refinement (Inspiration 2)
    def simulate_annealing(initial_set, time_budget):
        # State: list of booleans, index i corresponds to element i+1
        present = [False] * N
        for x in initial_set:
            present[x-1] = True
            
        # Precompute neighbors for fast violation check?
        # adj[i] = list of (j, k)
        adj = [[] for _ in range(N)]
        for (i, j, k) in triples:
            # Convert to 0-based
            i0, j0, k0 = i-1, j-1, k-1
            adj[i0].append((j0, k0))
            adj[j0].append((i0, k0))
            adj[k0].append((i0, j0))
            
        def calc_violations():
            v = 0
            for (i, j, k) in triples:
                if present[i-1] and present[j-1] and present[k-1]:
                    v += 1
            return v

        def calc_score():
            size = sum(present)
            viol = calc_violations()
            return size - 1000 * viol

        current_score = calc_score()
        best_score = current_score
        best_present = present[:]
        
        T = 10.0
        cooling = 0.999
        start_sa = time.time()
        
        while time.time() - start_sa < time_budget:
            # Pick random index to flip
            idx = random.randint(0, N - 1)
            old_val = present[idx]
            present[idx] = not old_val
            
            new_score = calc_score()
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_present = present[:]
            else:
                # Accept with probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    present[idx] = old_val # Revert
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_present

    # Strategy:
    # 1. Run a few greedy passes to get good initial seeds.
    # 2. Run SA on the best seed with remaining time.
    
    best_set = set()
    best_size = 0
    
    # Run 100 greedy passes to find a good starting point
    for i in range(100):
        if time.time() > deadline - 30: # Leave 30s for SA
            break
        s = run_greedy(randomize_ties=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # If time remains, run SA on the best greedy result
    time_elapsed = time.time() - start_time
    if time_elapsed < deadline - 5:
        sa_budget = deadline - time_elapsed - 5
        best_present = simulate_annealing(best_set, sa_budget)
        sa_set = {i+1 for i, val in enumerate(best_present) if val}
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = sa_set
            
    # Final check: Ensure validity (SA might leave violations if penalty wasn't high enough or T too high)
    # If violations exist, we must remove elements to fix them.
    # Let's do a final greedy cleanup on the best_set to ensure validity.
    # Actually, the SA objective penalizes violations heavily, so it should be valid.
    # But to be safe:
    final_set = set(best_set)
    # Check for violations
    valid = True
    for (a, b, c) in triples:
        if a in final_set and b in final_set and c in final_set:
            valid = False
            break
            
    if not valid:
        # Fallback to greedy cleanup
        active_elements = set(best_set)
        active_triples = set(range(num_triples))
        conflict_count = {i: 0 for i in range(1, N + 1)}
        for idx in active_triples:
            a,b,c = triples[idx]
            if a in active_elements and b in active_elements and c in active_elements:
                for x in (a,b,c):
                    conflict_count[x] += 1
        while True:
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
            victim = min(candidates)
            active_elements.remove(victim)
            for idx in elem_to_triples[victim]:
                if idx in active_triples:
                    a,b,c = triples[idx]
                    if a in active_elements and b in active_elements and c in active_elements:
                        active_triples.remove(idx)
                        for x in (a,b,c):
                            if x != victim and x in active_elements:
                                conflict_count[x] -= 1
        final_set = active_elements

    return list(final_set)

# EVOLVE-BLOCK-END
