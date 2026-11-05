import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = 45.0  # Safety margin
    
    # Precompute 3-APs for fast checking? 
    # Or just check on the fly.
    # On-the-fly check for adding v to S:
    # v conflicts if exists a in S such that 2a-v in S (and 2a-v != v)
    # or exists a in S such that 2v-a in S (and 2v-a != v)
    
    def is_valid_addition(S, v):
        # S is a set
        # Check if v forms a 3-AP with any two elements in S
        # Case 1: v is the middle. a, v, c. a+c=2v.
        # If a in S, c = 2v - a. If c in S and c != v, invalid.
        # Case 2: v is the end. a, b, v. a+v=2b => b=(a+v)/2.
        # If a in S, b = (a+v)/2. If b is integer, b in S, b != v, invalid.
        # Case 3: v is the start. v, b, c. v+c=2b => b=(v+c)/2.
        # If c in S, b = (v+c)/2. If b is integer, b in S, b != v, invalid.
        
        # Simpler: Iterate through S.
        # For each a in S:
        #   1. Check if (2*a - v) is in S. If yes and 2*a - v != v and 2*a - v > 0, conflict.
        #      (This covers v being middle or end/start depending on order)
        #      Actually, if a, v, c is AP, then a+c=2v.
        #      If a, b, v is AP, then a+v=2b.
        #      If v, b, c is AP, then v+c=2b.
        
        # Let's just check all pairs (a, b) in S? No, that's O(N^2).
        # Check against each a in S:
        # 1. Does a, v, c exist? c = 2v - a. If c in S and c != v, conflict.
        # 2. Does a, b, v exist? b = (a+v)/2. If (a+v) % 2 == 0 and b in S and b != v, conflict.
        # 3. Does v, b, c exist? b = (v+c)/2. This is symmetric to 2 if we iterate c?
        #    Actually, if we check for each a in S:
        #    - Is there c in S such that a, v, c is AP? (v is middle) -> c = 2v - a.
        #    - Is there b in S such that a, b, v is AP? (v is end) -> b = (a+v)/2.
        #    - Is there b in S such that v, b, a is AP? (v is start) -> b = (v+a)/2.
        #    Note: Case 2 and 3 are the same check: b = (a+v)/2.
        #    So for each a in S:
        #    if (2*v - a) in S and 2*v - a != v: return False
        #    if (a + v) % 2 == 0:
        #       b = (a + v) // 2
        #       if b in S and b != v: return False
        #    return True
        
        for a in S:
            # v is middle: a, v, c
            c = 2 * v - a
            if c != v and c in S:
                return False
            # v is end: a, b, v OR v is start: v, b, a
            # In both cases, the middle term b is (a+v)/2
            if (a + v) % 2 == 0:
                b = (a + v) // 2
                if b != v and b in S:
                    return False
        return True

    def get_score(S):
        return sum(1.0 / x for x in S)

    # 1. Greedy Initialization
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = get_score(best_S)
    
    # 2. Local Search: 1-Swap
    # Try to replace u in S with v not in S (v < u) to increase score.
    # We iterate u in S, v in not S.
    # To be fast, sort candidates.
    
    current_S = set(S)
    current_score = get_score(current_S)
    
    no_improvement_count = 0
    
    while time.time() - start_time < deadline:
        improved = False
        # Get lists
        in_S = list(current_S)
        not_in_S = [x for x in range(1, n + 1) if x not in current_S]
        
        # Sort not_in_S by weight descending (smaller x first) to find good swaps early?
        # Or just iterate all.
        
        # Try all swaps
        # Optimization: Only consider v < u because 1/v > 1/u.
        for u in in_S:
            # We want v < u
            # Iterate v in not_in_S where v < u
            # Since not_in_S is unsorted, we can filter.
            # To speed up, we can pre-sort not_in_S once per iteration?
            # Let's just filter.
            for v in not_in_S:
                if v >= u:
                    continue
                
                # Check if swap is valid
                # Remove u, add v
                # Check if v is valid in current_S - {u}
                
                # Efficient check:
                # v must not form AP with any pair in current_S - {u}
                # is_valid_addition(current_S - {u}, v)
                
                # To avoid creating a new set, we can modify the check function
                # or just create the set (N=256, set copy is fast).
                
                # Let's try the set copy approach first for clarity and speed in Python
                test_S = current_S.copy()
                test_S.remove(u)
                
                if is_valid_addition(test_S, v):
                    new_score = current_score - 1.0/u + 1.0/v
                    if new_score > current_score + 1e-9:
                        current_S = test_S
                        current_S.add(v)
                        current_score = new_score
                        improved = True
                        no_improvement_count = 0
                        # Restart loops? Or just continue?
                        # If we continue, in_S and not_in_S are stale.
                        # Break out to refresh lists.
                        break
            if improved:
                break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(current_S)
        else:
            no_improvement_count += 1
            # If stuck, try a perturbation:
            # Remove a random element from S, then re-greedy fill?
            # Or just try a 2-swap?
            # Let's try a simple restart with a different seed or perturbation.
            if no_improvement_count > 5:
                # Perturbation: Remove 1-2 random elements from current_S
                # and re-greedy
                perturbed_S = set(current_S)
                if len(perturbed_S) > 2:
                    # Remove a large element (low weight) to allow smaller ones in?
                    # Or just random.
                    # Let's remove the largest element (lowest weight)
                    max_u = max(perturbed_S)
                    perturbed_S.remove(max_u)
                    
                    # Re-greedy from perturbed_S
                    # We can just try to add more elements to perturbed_S
                    for x in range(1, n + 1):
                        if x not in perturbed_S:
                            if is_valid_addition(perturbed_S, x):
                                perturbed_S.add(x)
                    
                    if get_score(perturbed_S) > current_score:
                        current_S = perturbed_S
                        current_score = get_score(current_S)
                        no_improvement_count = 0
                    else:
                        # Try another perturbation strategy
                        # Remove a random small element? No, that hurts score.
                        # Remove a random element and re-greedy
                        if perturbed_S:
                            rand_u = random.choice(list(perturbed_S))
                            perturbed_S2 = set(current_S)
                            perturbed_S2.remove(rand_u)
                            for x in range(1, n + 1):
                                if x not in perturbed_S2:
                                    if is_valid_addition(perturbed_S2, x):
                                        perturbed_S2.add(x)
                            if get_score(perturbed_S2) > current_score:
                                current_S = perturbed_S2
                                current_score = get_score(current_S)
                                no_improvement_count = 0
                else:
                    no_improvement_count += 10 # Force exit or bigger change

    return list(best_S)
