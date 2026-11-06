import time
import random
import math

def construct():
    n = 20
    max_val = 10000
    
    def score(A):
        A = list(A)
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    # Generate candidate pool: focus on highly composite/smooth numbers
    # and small integers to maximize collision opportunities
    candidates = set()
    
    # Small integers
    for i in range(1, 150):
        candidates.add(i)
    
    # Highly composite numbers
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    # Products of up to 4 primes
    for p1 in primes:
        for p2 in primes:
            v1 = p1 * p2
            if v1 <= max_val:
                candidates.add(v1)
            for p3 in primes:
                v2 = v1 * p3
                if v2 <= max_val:
                    candidates.add(v2)
                for p4 in primes:
                    v3 = v2 * p4
                    if v3 <= max_val:
                        candidates.add(v3)
    
    # Multiples of small numbers
    for i in range(1, 200):
        for mult in [2, 3, 4, 5, 6, 8, 10, 12]:
            v = i * mult
            if v <= max_val:
                candidates.add(v)
    
    candidate_list = sorted([x for x in candidates if 1 <= x <= max_val])
    
    # Use a graph-based approach: build a bipartite-like structure where
    # edges represent "collision potential" between pairs of candidates.
    # For each pair (a,b), count how many sums/prods would collide with
    # a "reference" set. Then use greedy modularity-like selection.
    
    # Precompute pair-level collision potential
    # For each pair (i, j), we want to know: if we had a set containing both,
    # how many internal sum/product collisions would result?
    # This is expensive for all pairs, so we use a heuristic:
    # score_pair(a, b) = number of c in candidate_list such that
    #   (a+b) == (c+c) or (a+b) == (c+d) for some d, or similar
    # Simplified: just count direct collisions a+b with 2c, a*b with c*c
    
    # Instead, use a greedy approach with a scoring function that estimates
    # the marginal contribution to sum/product set sizes
    
    def marginal_cost(A, cand):
        """Estimate how many NEW sums and prods cand adds to set A"""
        current_sums = set()
        current_prods = set()
        for a in A:
            for b in A:
                current_sums.add(a + b)
                current_prods.add(a * b)
        
        new_sums = 0
        new_prods = 0
        for a in A:
            if (a + cand) not in current_sums:
                new_sums += 1
            if (a * cand) not in current_prods:
                new_prods += 1
        # Also self-interactions
        if (cand + cand) not in current_sums:
            new_sums += 1
        if (cand * cand) not in current_prods:
            new_prods += 1
        return new_sums + new_prods
    
    # Greedy construction with multiple restarts
    best_A = None
    best_score = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Try several different starting strategies
    start_strategies = []
    
    # Strategy 1: Start with 1
    start_strategies.append([1])
    # Strategy 2: Start with 2
    start_strategies.append([2])
    # Strategy 3: Start with 6
    start_strategies.append([6])
    # Strategy 4: Start with a random small number
    for _ in range(5):
        start_strategies.append([random.randint(1, 30)])
    
    for start in start_strategies:
        if time.time() > deadline - 10:
            break
        
        current = start[:]
        current_set = set(current)
        
        while len(current) < n and time.time() < deadline - 10:
            best_cand = None
            best_cost = float('inf')
            
            for cand in candidate_list:
                if cand in current_set:
                    continue
                
                cost = marginal_cost(current, cand)
                if cost < best_cost:
                    best_cost = cost
                    best_cand = cand
            
            if best_cand is None:
                break
            
            current.append(best_cand)
            current_set.add(best_cand)
        
        if len(current) == n:
            sc = score(current)
            if sc < best_score:
                best_score = sc
                best_A = current[:]
    
    # Local search: try to improve by swapping elements
    if best_A is not None:
        current_A = best_A[:]
        current_score = best_score
        
        # Simulated annealing-style local search
        temp = 100
        cooling = 0.95
        no_improve_count = 0
        
        while time.time() < deadline and temp > 1:
            improved_this_round = False
            random.shuffle(current_A)
            
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = current_A[i]
                best_new = None
                best_new_score = current_score
                
                # Try swapping with candidates
                for cand in candidate_list:
                    if cand in set(current_A):
                        continue
                    
                    temp_A = current_A[:]
                    temp_A[i] = cand
                    temp_score = score(temp_A)
                    
                    if temp_score < best_new_score:
                        best_new_score = temp_score
                        best_new = cand
                
                if best_new is not None and best_new != old_val:
                    current_A[i] = best_new
                    current_score = best_new_score
                    improved_this_round = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if not improved_this_round:
                temp *= cooling
            else:
                temp *= cooling
            
            if no_improve_count > n * 2:
                break
    
    return best_A
