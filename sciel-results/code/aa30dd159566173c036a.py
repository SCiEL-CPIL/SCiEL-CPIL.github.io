import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflict bitmasks: for each x, a bitmask of all y such that
    # there exists z where {x,y,z} is a 3-AP. Actually we need pairs.
    # Better: for each x, store a list of (y,z) pairs.
    # But for fast can_add, we can precompute: for each x, a list of pairs (y,z)
    # where y < z and {x,y,z} is a 3-AP.
    
    # conflict_pairs[x] = list of (y, z) such that x, y, z form a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            # 3-AP: (a, b, c) with a < b < c
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))
    
    def can_add(x, S_mask):
        """Check if x can be added to set represented by S_mask."""
        if S_mask & (1 << x):
            return False
        for y, z in conflict_pairs[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True
    
    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total
    
    def greedy_fill(forbidden_mask, S_mask):
        """Fill S by adding elements in ascending order."""
        S = S_mask
        for x in range(1, N + 1):
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S
    
    def greedy_fill_descending(forbidden_mask, S_mask):
        """Fill S by adding elements in descending order."""
        S = S_mask
        for x in range(N, 0, -1):
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S
    
    def local_search(S_mask, max_time):
        """Run local search starting from S_mask, return best mask found."""
        S = S_mask
        sum_float = get_float_sum(S)
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Phase 1: Try to add any element that can be added
            for x in range(1, N + 1):
                if can_add(x, S):
                    S |= (1 << x)
                    sum_float += recip[x]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: 1-swap local search
            # Find best swap: remove x, add y, with recip[y] > recip[x]
            best_gain = 0
            best_x = -1
            best_y = -1
            
            for y in range(1, N + 1):
                if S & (1 << y):
                    continue
                for x in range(N, 0, -1):
                    if not (S & (1 << x)):
                        continue
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        break  # No more gain for this y since x is decreasing
                    # Check if swap is valid: remove x, then can we add y?
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
                        break  # Only need to check if x can be removed and y added;
                               # the best x for a given y is the largest x (smallest recip[x])
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                sum_float += best_gain
                improved = True
                continue
            
            # Phase 3: 1-remove + re-greedy (ascending)
            best_remove_gain = -1
            best_remove_x = -1
            best_remove_S = None
            
            for x in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if not (S & (1 << x)):
                    continue
                
                S_no_x = S & ~(1 << x)
                # Re-greedy with x forbidden
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                
                gain = sum_new - sum_float
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_x = x
                    best_remove_S = S_new
            
            if best_remove_x != -1:
                S = best_remove_S
                sum_float += best_remove_gain
                improved = True
                continue
            
            # Phase 4: 2-swap (remove two, add two) - limited random tries
            if (time.time() - start_time) < max_time - 2:
                S_list = [x for x in range(1, N + 1) if S & (1 << x)]
                not_S_list = [x for x in range(1, N + 1) if not (S & (1 << x))]
                
                if len(S_list) >= 2 and len(not_S_list) >= 2:
                    # Try a limited number of 2-swap combinations
                    num_tries = min(50, int((max_time - (time.time() - start_time)) * 5))
                    for _ in range(num_tries):
                        if (time.time() - start_time) > max_time:
                            break
                        x1, x2 = random.sample(S_list, 2)
                        y1, y2 = random.sample(not_S_list, 2)
                        
                        gain = recip[y1] + recip[y2] - recip[x1] - recip[x2]
                        if gain <= 1e-15:
                            continue
                        
                        # Check validity
                        S_new = S & ~(1 << x1) & ~(1 << x2)
                        if can_add(y1, S_new):
                            S_temp = S_new | (1 << y1)
                            if can_add(y2, S_temp):
                                # Valid swap
                                S = S_temp | (1 << y2)
                                sum_float += gain
                                improved = True
                                break
            
            if not improved:
                # Random perturbation: remove a random element and re-greedy
                if (time.time() - start_time) > max_time:
                    break
                S_list = [x for x in range(1, N + 1) if S & (1 << x)]
                if not S_list:
                    break
                x = random.choice(S_list)
                
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                
                if sum_new > sum_float:
                    S = S_new
                    sum_float = sum_new
                    improved = True
        
        return S, sum_float
    
    # Step 1: Initializations
    initializations = []
    
    # Greedy ascending
    S_asc = greedy_fill(0, 0)
    initializations.append(S_asc)
    
    # Greedy descending
    S_desc = greedy_fill_descending(0, 0)
    initializations.append(S_desc)
    
    # Randomized greedy - more trials with varied skip probabilities
    for trial in range(20):
        if (time.time() - start_time) > deadline - 20:
            break
        random.seed(trial * 12345 + 67890)
        skip_prob = 0.05 + 0.1 * (trial % 5)  # Vary skip probability
        S = 0
        for x in range(1, N + 1):
            if random.random() < skip_prob and x > 3:
                continue
            if can_add(x, S):
                S |= (1 << x)
        initializations.append(S)
    
    # Step 2: Local Search on each initialization
    best_S_mask = 0
    best_sum = 0.0
    
    remaining_init_time = deadline - 8 - (time.time() - start_time)
    if remaining_init_time < 5:
        remaining_init_time = 5
    
    time_per_ls = max(2.0, remaining_init_time / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 4:
            break
        
        remaining_time = deadline - 4 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        S_time = min(time_per_ls, remaining_time)
        S_result, sum_result = local_search(S_init, start_time + S_time)
        
        if sum_result > best_sum:
            best_sum = sum_result
            best_S_mask = S_result
    
    # Step 3: Final Polish
    remaining = deadline - 2 - (time.time() - start_time)
    if remaining > 3:
        S_final, sum_final = local_search(best_S_mask, start_time + remaining)
        if sum_final > best_sum:
            best_sum = sum_final
            best_S_mask = S_final
    
    # Convert to list
    result = [x for x in range(1, N + 1) if best_S_mask & (1 << x)]
    
    # Validate
    if len(result) > 0:
        S_set = set(result)
        valid = True
        for a in S_set:
            for b in S_set:
                if a >= b:
                    continue
                s = a + b
                if s % 2 == 0:
                    mid = s // 2
                    if mid in S_set and mid != a and mid != b:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            # Fallback to greedy
            S_fallback = greedy_fill(0, 0)
            result = [x for x in range(1, N + 1) if S_fallback & (1 << x)]
    
    return result
