import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    N = 256
    
    def is_valid_addition(S, x):
        # Check if adding x to S creates a 3-AP
        # 1. x as middle: a + c = 2x for a, c in S
        #    c = 2x - a. If c in S, conflict.
        # 2. x as endpoint: a + x = 2c => c = (a+x)/2.
        #    If (a+x) even and c in S, conflict.
        
        for a in S:
            # x is middle
            c = 2 * x - a
            if c in S:
                return False
            # x is endpoint
            s = a + x
            if s % 2 == 0:
                c = s // 2
                if c in S:
                    return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def hill_climb(S):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to swap x_out (in S) with x_in (not in S)
            # We want x_in < x_out to increase reciprocal sum.
            x_outs = sorted([x for x in S], reverse=True)
            x_ins = sorted([x for x in range(1, N + 1) if x not in S])
            
            for x_out in x_outs:
                if time.time() > deadline:
                    break
                S_temp = S.copy()
                S_temp.remove(x_out)
                for x_in in x_ins:
                    if x_in >= x_out:
                        break
                    if is_valid_addition(S_temp, x_in):
                        S = S_temp
                        S.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        return S

    # Initial Greedy
    S = greedy_construct(range(1, N + 1))
    S = hill_climb(S)
    
    # Try a few randomized restarts if time permits
    best_S = S
    best_score = sum(1.0 / x for x in S)
    
    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 10:
        # Perturb the order or start from a random valid set?
        # Simple: Shuffle the range, do greedy, then hill climb.
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Bias towards small numbers?
        # Actually, pure shuffle might yield worse greedy.
        # Let's stick to the standard greedy but maybe try different starting points?
        # Or just run hill climb on the current best? It's already converged.
        # Let's try a "2-swap" move if 1-swap is stuck?
        # Or just break. The hill climber is usually very effective for this size.
        
        # Let's try one more thing: Randomly remove an element and re-add?
        # This is essentially what hill climb does.
        
        # If we have time, try a few random restarts of the whole process
        # with a slightly biased order (e.g., random permutation of 1..N but keeping small ones likely earlier)
        # Actually, the standard 1..N order is optimal for greedy.
        # Random restarts of greedy are likely worse.
        # Random restarts of hill climb from different initial states?
        # We can generate a random valid set by randomly adding elements.
        
        S_rand = set()
        # Randomly add elements
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(S_rand, x):
                S_rand.add(x)
        
        S_rand = hill_climb(S_rand)
        score_rand = sum(1.0 / x for x in S_rand)
        
        if score_rand > best_score:
            best_score = score_rand
            best_S = S_rand
        attempts += 1

    return list(best_S)
