import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 256
    
    def can_add(S_set, x):
        # Check if x can be added to S_set without forming a 3-AP
        # x is not in S_set
        for u in S_set:
            # Case 1: x is middle. u, x, v. v = 2x - u.
            v = 2*x - u
            if v in S_set:
                return False
            # Case 2: x is end. v, u, x. v = 2u - x.
            w = 2*u - x
            if w in S_set:
                return False
        return True

    def is_valid(S_set):
        # Verify the whole set is 3-AP free
        # Used for final check or debugging, not in the loop for speed
        for a in S_set:
            for b in S_set:
                if a == b: continue
                c = 2*b - a
                if c in S_set and c != a and c != b:
                    return False
        return True

    # 1. Greedy Construction
    # Iterate from 1 to N, add if possible
    S = []
    S_set = set()
    for i in range(1, N + 1):
        if can_add(S_set, i):
            S.append(i)
            S_set.add(i)
            
    best_S = list(S)
    best_sum = sum(1.0 / x for x in S)
    
    # 2. Local Search (Hill Climbing)
    # We try to swap an element y in S with an element x not in S, where x < y
    # to increase the sum of reciprocals.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_x = None
        best_y = None
        
        # To speed up, we can precompute candidates or just iterate.
        # Size of S is small (~40). Size of complement is ~216.
        # Pairs ~ 40 * 216 = 8640.
        # Each check is O(|S|) ~ 40.
        # Total ops ~ 350,000 per pass. Very fast.
        
        # Iterate x from 1 to N (smaller x is better)
        # If x is in S, skip.
        # Iterate y in S.
        # If x >= y, skip (gain would be <= 0)
        
        # Optimization: We can break early if 1/x - 1/max(S) <= 0? 
        # No, we need to find the best swap.
        
        # We can sort S to make it easier? Not strictly necessary.
        
        for x in range(1, N + 1):
            if x in S_set:
                continue
            
            # Potential max gain is 1/x - 1/max_y
            # If 1/x is very small, it's unlikely to be the best.
            
            for y in S:
                if y <= x:
                    continue
                
                gain = 1.0 / x - 1.0 / y
                if gain <= best_gain:
                    continue
                
                # Check if swap is valid
                # Remove y, add x
                # We need to check if x forms 3-AP with (S \ {y})
                
                # Efficient check:
                # x is in (S \ {y}) + x.
                # For each u in S \ {y}:
                #   v = 2x - u. If v in S \ {y}, invalid.
                #   w = 2u - x. If w in S \ {y}, invalid.
                
                # Note: v or w could be y. If v == y, then y = 2x - u => u = 2x - y.
                # If the only conflict involves y, then the swap is valid.
                
                valid = True
                for u in S:
                    if u == y:
                        continue
                    
                    v = 2 * x - u
                    if v in S_set:
                        # Conflict if v is in S and v != x (x not in S yet)
                        # and v != y (y is being removed)
                        if v != y:
                            valid = False
                            break
                        
                    w = 2 * u - x
                    if w in S_set:
                        if w != y:
                            valid = False
                            break
                
                if valid:
                    best_gain = gain
                    best_x = x
                    best_y = y
        
        if best_x is not None:
            S_set.remove(best_y)
            S_set.add(best_x)
            S = list(S_set)
            current_sum = sum(1.0 / x for x in S)
            improved = True
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_S = list(S)
                
        else:
            # No improvement found in this pass.
            # Try a kick?
            # A simple kick: remove a random element, then greedily fill again?
            # Or just break, as the landscape is likely convex enough for this small N.
            # Given the time limit is generous, we can try a few random restarts of the greedy phase?
            # No, greedy is deterministic. 
            # Let's try a "shuffle" kick: remove 2 random elements, then try to re-add them or others?
            # Actually, the local search is quite powerful. If it stalls, it's likely near optimal.
            # We can break to save time, or do a limited random perturbation.
            
            # Let's do a limited perturbation:
            # Pick a random y in S, remove it.
            # Try to add the smallest x < y that fits.
            # This is essentially what the hill climbing does, but it might be stuck in a specific order.
            # However, the hill climbing checks ALL pairs, so it's a global best-improvement step.
            # If no improving swap exists, we are at a local optimum.
            
            # To escape, we can allow a "neutral" or "slightly negative" move?
            # Or random restart from scratch?
            # Given N=256, the structure is rigid. The greedy solution is usually the global optimum for this type of problem (max weight independent set on a graph with specific structure, but here the graph is sparse and weights are monotonic).
            
            # Let's just break. The gain from further search is likely negligible.
            break

    return best_S
