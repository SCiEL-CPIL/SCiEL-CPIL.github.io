import random
import time

def construct():
    n = 128
    # Precompute all possible pairs and their sums
    # We will use a set S for membership checks
    
    def get_violation_count(S):
        count = 0
        S_list = sorted(S)
        # To speed up, we can iterate pairs (a, b)
        # If a+b is in S, check for c.
        # Optimization: For a fixed pair (a,b) with a+b in S,
        # we need c in S such that a+c in S and b+c in S.
        # This is equivalent to c in (S - a) and c in (S - b).
        # Let's just iterate c for now, but break early if found?
        # No, we need the COUNT for the score, or just a boolean for validity?
        # For SA, a penalty term is good. But counting all violations is O(N^3).
        # N=128, N^3 = 2M. 5000 iterations * 2M = 10^10 ops. Too slow in Python.
        
        # We need a faster violation check or count.
        # Alternative: Just check if ANY violation exists (Boolean).
        # If we use Boolean penalty (0 if valid, large constant if invalid),
        # we might get stuck.
        # Better: Use a heuristic score.
        
        # Let's try to optimize the Boolean check first.
        # If we can check validity in O(N^2) or O(N^2 * log N), it's feasible.
        # N=128, N^2 = 16k. 5000 * 16k = 80M. Feasible in Python if optimized.
        
        # Check:
        # For each pair (a, b) in S with a < b:
        #   if a+b in S:
        #      for each c in S:
        #         if c != a and c != b and (a+c in S) and (b+c in S):
        #            return True
        # This is O(N^3) worst case.
        
        # Can we do better?
        # For fixed a, b:
        # We need c in S such that a+c in S and b+c in S.
        # Let S_a = {x-a | x in S}. We need c in S_a.
        # Let S_b = {x-b | x in S}. We need c in S_b.
        # So c in S \cap S_a \cap S_b.
        # We can precompute S_a and S_b? No, S changes.
        
        # However, we can iterate c in S and check if (c-a) in S and (c-b) in S?
        # No, the condition is a+c in S. So c-a is not the right transform.
        # a+c = s1 => c = s1 - a.
        # So for fixed a, b, we want to know if there is a c in S such that
        # (a+c) in S and (b+c) in S.
        
        # Let's stick to the O(N^3) check but prune aggressively.
        # If N is small (e.g., < 80), it's fast.
        # If N is large, it's slow.
        # Start with small N, grow it.
        
        S_list = list(S)
        n_curr = len(S_list)
        # Sort to ensure a < b < c logic if needed, but set membership is unordered.
        # We just need distinct a,b,c.
        
        for i in range(n_curr):
            a = S_list[i]
            for j in range(i+1, n_curr):
                b = S_list[j]
                if a + b not in S:
                    continue
                # Check for c
                # c must be in S, c != a, c != b
                # a+c in S, b+c in S
                for k in range(n_curr):
                    c = S_list[k]
                    if c == a or c == b:
                        continue
                    if (a + c) in S and (b + c) in S:
                        return 1 # Just return 1 if any violation found
        return 0

    # SA Parameters
    T = 100.0
    T_min = 0.01
    cooling = 0.9995
    max_iters = 20000
    start_time = time.time()
    
    # Initial state: Random subset of size ~50
    # Or start with Odds (64) and see if we can improve?
    # Odds has 0 violations.
    # Let's start with a random set of size 50 to explore.
    
    best_S = set()
    best_score = -1
    
    # Helper to get score
    def get_score(S):
        # We want to maximize size, penalize violations.
        # If valid, score = len(S)
        # If invalid, score = len(S) - 100 * violations? 
        # Since we only return 1 for violation, we don't know count.
        # Let's change get_violation_count to return count?
        # Counting is expensive.
        # Let's use a boolean penalty.
        # Score = len(S) - (100 if invalid else 0)
        # This drives the search to find valid sets.
        # Among valid sets, it maximizes len(S).
        
        if get_violation_count(S) > 0:
            return len(S) - 100
        return len(S)

    # Initialize with Odds
    S = set(range(1, n + 1, 2))
    current_score = get_score(S)
    
    for i in range(max_iters):
        if time.time() - start_time > 50:
            break
            
        # Generate neighbor
        S_new = set(S)
        move_type = random.random()
        if move_type < 0.4:
            # Add
            candidates = [x for x in range(1, n+1) if x not in S]
            if candidates:
                x = random.choice(candidates)
                S_new.add(x)
        elif move_type < 0.7:
            # Remove
            if S:
                x = random.choice(list(S))
                S_new.remove(x)
        else:
            # Swap
            if S:
                in_s = list(S)
                out_s = [x for x in range(1, n+1) if x not in S]
                if out_s:
                    x_in = random.choice(in_s)
                    x_out = random.choice(out_s)
                    S_new.remove(x_in)
                    S_new.add(x_out)
        
        new_score = get_score(S_new)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score >= current_score:
            S = S_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
        else:
            delta = new_score - current_score
            prob = pow(2.71828, delta / T)
            if random.random() < prob:
                S = S_new
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final check and cleanup
    # Ensure best_S is valid
    if get_violation_count(best_S) > 0:
        # Fallback to a known valid set if SA failed badly
        # Try greedy from empty
        S_fallback = set()
        for x in range(1, n+1):
            S_test = S_fallback | {x}
            if get_violation_count(S_test) == 0:
                S_fallback = S_test
        if len(S_fallback) > len(best_S):
            best_S = S_fallback
        else:
            # If best_S is invalid, we must return a valid one.
            # The problem says "Any broken rule makes the whole result invalid".
            # So we MUST return a valid set.
            # If best_S is invalid, we need to repair it or use fallback.
            # Let's use the greedy fallback always if best_S is invalid.
            pass # Handled above

    return sorted(best_S)
