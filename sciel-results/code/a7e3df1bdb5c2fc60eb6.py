import time
import random

def construct():
    """
    Finds the largest Sidon set (distinct pairwise sums) in [1, 284]
    containing {1, 2, 5, 11}.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    prescribed = [1, 2, 5, 11]
    pset = set(prescribed)
    MAXV = 284
    
    def get_sums(S):
        sums = set()
        sl = sorted(S)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def is_valid_add(x, S, used_sums):
        if x in S:
            return False
        for y in S:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True

    # Initial state
    S0 = set(prescribed)
    sums0 = get_sums(S0)
    
    best_S = list(S0)
    best_size = len(S0)
    
    # Beam Search Phase
    beam_width = 100
    max_depth = 25
    rng = random.Random(123)
    
    beam = [(S0, sums0)]
    
    for depth in range(max_depth):
        if time.time() > deadline - 30: # Leave 30s for local search
            break
            
        next_beam = []
        seen_states = set()
        
        for S, used_sums in beam:
            valid_candidates = []
            for x in range(1, MAXV + 1):
                if x in S:
                    continue
                if is_valid_add(x, S, used_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                continue
                
            # Sample candidates to expand
            # If many, sample to keep beam diverse and small
            if len(valid_candidates) > 10:
                candidates_to_try = rng.sample(valid_candidates, 10)
            else:
                candidates_to_try = valid_candidates
                
            for x in candidates_to_try:
                new_S = S | {x}
                new_sums = used_sums | {x + y for y in S} | {2 * x}
                
                key = tuple(sorted(new_S))
                if key in seen_states:
                    continue
                seen_states.add(key)
                
                next_beam.append((new_S, new_sums))
                
                if len(new_S) > best_size:
                    best_size = len(new_S)
                    best_S = list(new_S)
        
        if not next_beam:
            break
            
        if len(next_beam) > beam_width:
            # Score by number of remaining options
            def score_state(state):
                S, used_sums = state
                count = 0
                for x in range(1, MAXV + 1):
                    if x not in S and is_valid_add(x, S, used_sums):
                        count += 1
                return count
            
            next_beam.sort(key=score_state, reverse=True)
            beam = next_beam[:beam_width]
        else:
            beam = next_beam

    # Local Search Phase
    # Start from the best set found so far
    cur_S = set(best_S)
    cur_sums = get_sums(cur_S)
    cur_size = len(cur_S)
    
    ls_rng = random.Random(456)
    
    # Time budget for local search
    ls_deadline = deadline - 2.0
    
    while time.time() < ls_deadline:
        # Try to add
        valid_adds = [x for x in range(1, MAXV + 1) if x not in cur_S and is_valid_add(x, cur_S, cur_sums)]
        
        if valid_adds:
            # Pick a random valid addition
            x = ls_rng.choice(valid_adds)
            cur_S.add(x)
            cur_sums = get_sums(cur_S) # Rebuild for simplicity, fast enough
            cur_size += 1
            if cur_size > best_size:
                best_size = cur_size
                best_S = list(cur_S)
            continue
            
        # No valid additions, try swaps
        # Pick a random non-prescribed element to remove
        removable = [x for x in cur_S if x not in pset]
        if not removable:
            break # Can't improve if only prescribed elements left (and no adds)
            
        y = ls_rng.choice(removable)
        temp_S = cur_S - {y}
        temp_sums = get_sums(temp_S)
        
        valid_adds_temp = [x for x in range(1, MAXV + 1) if x not in temp_S and is_valid_add(x, temp_S, temp_sums)]
        
        if valid_adds_temp:
            # Perform swap
            x = ls_rng.choice(valid_adds_temp)
            cur_S = temp_S | {x}
            cur_sums = get_sums(cur_S)
            # Size remains same, but configuration changed
            # We don't update best_size here as it didn't increase
        else:
            # If we can't swap with y, try another y? 
            # To avoid getting stuck in a loop of failing swaps, 
            # we can just restart the local search from best_S with a new random seed
            # or break if we've tried many times.
            # For simplicity, just continue the while loop, picking a new y.
            pass

    return best_S
