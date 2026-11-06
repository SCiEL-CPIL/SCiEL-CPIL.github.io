import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit=40.0):
        n = len(vals)
        start_time = time.time()
        
        # Map values to indices
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # Precompute conflict structure
        # A triple (i, j, k) is a conflict if vals[i] + vals[j] == vals[k]
        # We need to forbid (i, j, k) all being True.
        # This is a 3-CNF clause: ~(x_i & x_j & x_k).
        # In Max-2-SAT style, we can handle this via local search on the objective:
        # Maximize sum(x_i) subject to no triple fully active.
        # Or, more directly, use a local search that minimizes violations + (n - sum(x_i)).
        
        # For efficiency in local search, we precompute for each index k,
        # the list of pairs (i, j) such that vals[i] + vals[j] == vals[k].
        # If x_k is True, then for all such pairs, we cannot have both x_i and x_j True.
        
        # Alternatively, for each pair (i, j), precompute k = idx(i+j).
        # If x_i and x_j are True, then x_k must be False.
        
        # Let's use the pair-based view for local search moves.
        # State: x (bool array of size n)
        # Objective: maximize sum(x)
        # Constraint: for all i, j, k with i+j=k, not (x_i and x_j and x_k)
        
        # We can define a "violation score" for a state:
        # V(x) = sum over all triples (i,j,k) of [x_i * x_j * x_k]
        # We want to maximize sum(x) - lambda * V(x) with large lambda, or just minimize V(x) then maximize sum(x).
        # Given the problem size (n<=240), the number of triples is small.
        
        # Let's precompute the list of triples.
        triples = []
        for i in range(n):
            for j in range(i, n): # i, j can be same
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # We have a triple (i, j, k). 
                    # Note: if i != j, (j, i, k) is the same constraint.
                    # If i == j, it's (i, i, k).
                    triples.append((i, j, k))
        
        # To speed up local search, we can precompute for each variable k,
        # the pairs (i, j) that form a triple with k.
        # pairs_for_k[k] = list of (i, j)
        pairs_for_k = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_for_k[k].append((i, j))
            
        # Also, for each variable i, which other variables j does it interact with?
        # This is implicitly handled by the triples.
        
        # Local Search Strategy:
        # 1. Start with a random or greedy initialization.
        # 2. Calculate current sum and violations.
        # 3. Try flipping a random variable.
        #    - If flip reduces violations, accept.
        #    - If flip keeps violations same but increases sum, accept.
        #    - If flip keeps violations same and sum same, accept with small prob (to escape plateaus).
        #    - If flip increases violations, accept with small prob (simulated annealing style) or only if it allows future gains.
        
        # A better approach for "Maximize Sum subject to No Violations":
        # Since violations are "hard" constraints in the final answer, we must end with 0 violations.
        # We can use a two-phase approach:
        # Phase 1: Minimize violations.
        # Phase 2: Maximize sum (given 0 violations).
        
        # Or, use a weighted objective: Score = Sum(x) - 1000 * Violations(x).
        # We want to maximize Score.
        
        # Let's implement a fast local search on this score.
        
        def calc_violations(x):
            v = 0
            for i, j, k in triples:
                if x[i] and x[j] and x[k]:
                    v += 1
            return v

        def calc_score(x):
            return np.sum(x) - 1000 * calc_violations(x)

        # To make local search fast, we need to update score incrementally.
        # Flipping x[i] affects:
        # 1. The sum term: +/- 1
        # 2. The violation term: 
        #    - Triples where i is the "sum" (k): (a, b, i). Violation changes if x[a] and x[b] are True.
        #    - Triples where i is an "addend" (i or j): (i, b, k) or (a, i, k). Violation changes if x[b] and x[k] are True.
        
        # Let's precompute lists for incremental updates.
        # triples_where_k_is_sum[i] = list of (a, b) such that a+b=i
        # triples_where_i_is_addend[i] = list of (j, k) such that i+j=k
        
        triples_where_k_is_sum = [[] for _ in range(n)]
        triples_where_i_is_addend = [[] for _ in range(n)]
        
        for a, b, k in triples:
            triples_where_k_is_sum[k].append((a, b))
            if a != b:
                triples_where_i_is_addend[a].append((b, k))
                triples_where_i_is_addend[b].append((a, k))
            else:
                # If a==b, the pair is (a,a). Flipping a affects the triple (a,a,k).
                # We handle this carefully.
                # Actually, the list above adds (b,k) to addend[a] and (a,k) to addend[b].
                # If a==b, it adds (a,k) twice? No, the loop is `if a!=b`.
                # So for a==b, we need to add it once to addend[a].
                triples_where_i_is_addend[a].append((a, k))

        # Wait, the logic for addends needs to be precise.
        # Triple (i, j, k).
        # If we flip i:
        #   - It is the sum in triples (a, b, i). Check x[a] & x[b].
        #   - It is the first addend in triples (i, j, k). Check x[j] & x[k].
        #   - It is the second addend in triples (j, i, k). Check x[j] & x[k].
        # Note: (i, j, k) and (j, i, k) are distinct entries in `triples` if i != j?
        # My loop: `for i in range(n): for j in range(i, n)`.
        # So if i < j, we have (i, j, k). We do NOT have (j, i, k).
        # But the constraint is symmetric in i and j.
        # So when flipping i, we must consider:
        # 1. Triples where i is the sum: (a, b, i).
        # 2. Triples where i is the smaller addend: (i, j, k) with i < j.
        # 3. Triples where i is the larger addend: (j, i, k) with j < i.
        
        # Let's rebuild the lists to be explicit about roles.
        # For each index i:
        # as_sum[i] = list of (a, b) such that a+b=i (from triples)
        # as_addend[i] = list of (other, k) such that i+other=k (from triples)
        # Note: if i+other=k, the triple is (min(i,other), max(i,other), k).
        # We just need to know if x[other] and x[k] are True.
        
        as_sum = [[] for _ in range(n)]
        as_addend = [[] for _ in range(n)]
        
        for a, b, k in triples:
            as_sum[k].append((a, b))
            as_addend[a].append((b, k))
            if a != b:
                as_addend[b].append((a, k))
        
        # Now, let's implement the local search.
        
        best_x = None
        best_score = -np.inf
        best_sum = 0
        
        rng = random.Random(42)
        
        # Initialization strategies
        inits = []
        
        # 1. All True (will have many violations, but high sum)
        inits.append(np.ones(n, dtype=bool))
        
        # 2. Random
        for _ in range(10):
            inits.append(np.random.random(n) > 0.5)
            
        # 3. Greedy: Start empty, add if no violation?
        # This is slow to init but gives a valid start.
        x_greedy = np.zeros(n, dtype=bool)
        # We need a fast way to check if adding i creates a violation.
        # Adding i creates violation if:
        # - There exists (a, b) in as_sum[i] such that x[a] and x[b] are True.
        # - There exists (j, k) in as_addend[i] such that x[j] and x[k] are True.
        
        def can_add(x, i):
            # Check if adding i (setting x[i]=True) creates a new violation
            # Current x[i] is False.
            for a, b in as_sum[i]:
                if x[a] and x[b]:
                    return False
            for j, k in as_addend[i]:
                if x[j] and x[k]:
                    return False
            return True

        # Greedy init
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if can_add(x_greedy, i):
                x_greedy[i] = True
        inits.append(x_greedy)
        
        # Odds init
        x_odds = np.array([v % 2 != 0 for v in vals])
        inits.append(x_odds)
        
        # Positives init
        x_pos = np.array([v > 0 for v in vals])
        inits.append(x_pos)
        
        # Negatives init
        x_neg = np.array([v < 0 for v in vals])
        inits.append(x_neg)
        
        for x_init in inits:
            if time.time() > start_time + time_limit:
                break
            
            x = x_init.copy()
            
            # Calculate initial state
            current_violations = calc_violations(x)
            current_sum = np.sum(x)
            current_score = current_sum - 1000 * current_violations
            
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                best_sum = current_sum
            
            # Local Search
            improved = True
            no_improve_count = 0
            T = 1.0 # Temperature for simulated annealing-like acceptance
            
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                # Try flipping a random variable
                # We can pick a variable involved in violations first, or random.
                # Let's pick random.
                i = rng.randint(0, n - 1)
                
                # Calculate delta score if we flip i
                # New sum = current_sum + (1 if x[i] is False else -1)
                delta_sum = 1 if not x[i] else -1
                
                # Delta violations
                # If x[i] becomes True:
                #   New violations are created if:
                #   - (a, b) in as_sum[i] and x[a] and x[b] are True.
                #   - (j, k) in as_addend[i] and x[j] and x[k] are True.
                # If x[i] becomes False:
                #   Violations are removed if:
                #   - (a, b) in as_sum[i] and x[a] and x[b] are True.
                #   - (j, k) in as_addend[i] and x[j] and x[k] are True.
                
                # Note: The condition "x[a] and x[b] are True" is independent of x[i] in the current state.
                # So the number of violations created/removed is the same magnitude, just sign change.
                
                num_conflicts = 0
                for a, b in as_sum[i]:
                    if x[a] and x[b]:
                        num_conflicts += 1
                for j, k in as_addend[i]:
                    if x[j] and x[k]:
                        num_conflicts += 1
                
                # If x[i] is False -> True:
                #   Violations increase by num_conflicts.
                # If x[i] is True -> False:
                #   Violations decrease by num_conflicts.
                
                if x[i]:
                    delta_violations = -num_conflicts
                else:
                    delta_violations = num_conflicts
                    
                delta_score = delta_sum - 1000 * delta_violations
                
                # Acceptance criteria
                accept = False
                if delta_score > 0:
                    accept = True
                elif delta_score == 0:
                    # Accept to move around
                    accept = True
                else:
                    # Accept with probability exp(delta_score / T)
                    # Since delta_score is negative, this is small.
                    # T decays over time.
                    if rng.random() < np.exp(delta_score / T):
                        accept = True
                
                if accept:
                    x[i] = not x[i]
                    current_sum += delta_sum
                    current_violations += delta_violations
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_x = x.copy()
                        best_sum = current_sum
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    improved = True
                    T = max(0.1, T * 0.99) # Cool down
                else:
                    no_improve_count += 1
                    T = max(0.1, T * 0.99)
                    
                # If we have 0 violations, we can stop improving if sum is max?
                # No, we can still try to add more.
                # But if we have 0 violations, delta_score for adding a valid element is +1.
                # So it will always be accepted.
                # The search will naturally find the max sum valid set if it explores enough.
                
                # Safety break if stuck for too long
                if no_improve_count > 1000:
                    break

        # Final validation
        if best_x is None:
            best_x = np.zeros(n, dtype=bool)
            
        S = {vals[i] for i in range(n) if best_x[i]}
        
        # Ensure validity
        if not is_sum_free(S):
            # Fallback: Greedy valid set
            S = set()
            x_greedy = np.zeros(n, dtype=bool)
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                if can_add(x_greedy, i):
                    x_greedy[i] = True
                    S.add(vals[i])
                    
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        if size == 60:
            t_lim = 25.0
        elif size == 120:
            t_lim = 45.0
        else:
            t_lim = 65.0
            
        S = solve_instance(vals, t_lim)
        
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
