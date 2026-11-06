import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_size = 0
    best_set = set()
    
    def evaluate_candidate(x, S, counts):
        """
        Returns (is_valid, cost) for adding x to S.
        cost = number of sums that will become tight (count goes 1 -> 2).
        """
        cost = 0
        # Check 2*x
        c_2x = counts.get(2 * x, 0)
        if c_2x >= 2:
            return False, 0
        if c_2x == 1:
            cost += 1
            
        # Check x + y for y in S
        for y in S:
            c_xy = counts.get(x + y, 0)
            if c_xy >= 2:
                return False, 0
            if c_xy == 1:
                cost += 1
        return True, cost

    def greedy_with_cost(seed=0):
        random.seed(seed)
        S = set()
        counts = {}
        
        # Optional: Start with a random element to break symmetry?
        # Usually starting with 1 is good for density in low range, 
        # but for B2[2] in 1..N, distribution matters.
        # Let's just iterate through candidates and pick best.
        
        for _ in range(N):
            best_x = None
            best_cost = float('inf')
            valid_candidates = []
            
            # Evaluate all candidates not in S
            # Optimization: We can iterate 1..N
            for x in range(1, N + 1):
                if x in S:
                    continue
                is_valid, cost = evaluate_candidate(x, S, counts)
                if is_valid:
                    # Tie-breaking: random among best costs to diversify
                    if cost < best_cost:
                        best_cost = cost
                        valid_candidates = [x]
                    elif cost == best_cost:
                        valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick randomly from the best cost candidates
            # This allows different paths in different restarts
            x = random.choice(valid_candidates)
            S.add(x)
            
            # Update counts
            counts[2 * x] = counts.get(2 * x, 0) + 1
            for y in S:
                if y == x:
                    continue
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        
        return S

    # Run multiple restarts with different seeds
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        S = greedy_with_cost(seed=i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Local search improvement on the best set found so far?
        # Or just rely on the diversity of the greedy restarts.
        
    # Try to add more to the best set found (greedy completion)
    # The greedy_with_cost already tries to add until no more can be added.
    # But maybe the order of evaluation in the final steps matters?
    # The greedy_with_cost picks the globally best cost at each step.
    # This is a strong heuristic.
    
    # One more optimization: 
    # Sometimes starting with a specific small set helps.
    # But random tie-breaking in the greedy should cover this.
    
    return list(best_set)
