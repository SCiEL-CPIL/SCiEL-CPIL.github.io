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

    def count_valid_adds(S, used_sums):
        count = 0
        for x in range(1, MAXV + 1):
            if x not in S and is_valid_add(x, S, used_sums):
                count += 1
        return count

    # Initial state
    S0 = set(prescribed)
    sums0 = get_sums(S0)
    
    best_S = list(S0)
    best_size = len(S0)
    
    # Backtracking with forward checking
    def backtrack_search(S, used_sums, depth, max_depth, rng):
        nonlocal best_S, best_size
        
        if time.time() > deadline - 30:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        
        if depth >= max_depth:
            return
        
        # Get valid candidates
        valid_candidates = []
        for x in range(1, MAXV + 1):
            if x not in S and is_valid_add(x, S, used_sums):
                valid_candidates.append(x)
        
        if not valid_candidates:
            return
        
        # Order candidates by how many options they leave (forward checking heuristic)
        # Sample to keep it fast
        if len(valid_candidates) > 15:
            sampled = rng.sample(valid_candidates, 15)
        else:
            sampled = valid_candidates
        
        # Score candidates by remaining options
        def candidate_score(x):
            new_S = S | {x}
            new_sums = used_sums | {x + y for y in S} | {2 * x}
            return count_valid_adds(new_S, new_sums)
        
        scored = [(candidate_score(x), x) for x in sampled]
        scored.sort(reverse=True)
        
        # Try top candidates
        for _, x in scored[:8]:
            new_S = S | {x}
            new_sums = used_sums | {x + y for y in S} | {2 * x}
            backtrack_search(new_S, new_sums, depth + 1, max_depth, rng)
            
            if time.time() > deadline - 30:
                return
    
    # Run multiple backtracking attempts with different random seeds
    rng = random.Random(123)
    
    for attempt in range(5):
        if time.time() > deadline - 40:
            break
        
        # Try different initial orderings
        rng_seed = 123 + attempt * 7
        attempt_rng = random.Random(rng_seed)
        
        backtrack_search(S0, sums0, 0, 20, attempt_rng)
        
        if best_size >= 19:
            break
    
    # Local Search Phase
    # Start from the best set found so far
    cur_S = set(best_S)
    cur_sums = get_sums(cur_S)
    cur_size = len(cur_S)
    
    ls_rng = random.Random(456)
    
    # Time budget for local search
    ls_deadline = deadline - 2.0
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < ls_deadline and no_improve_count < max_no_improve:
        # Try to add
        valid_adds = [x for x in range(1, MAXV + 1) if x not in cur_S and is_valid_add(x, cur_S, cur_sums)]
        
        if valid_adds:
            # Pick a random valid addition
            x = ls_rng.choice(valid_adds)
            cur_S.add(x)
            cur_sums = get_sums(cur_S)
            cur_size += 1
            no_improve_count = 0
            if cur_size > best_size:
                best_size = cur_size
                best_S = list(cur_S)
            continue
        
        # No valid additions, try swaps
        # Pick a random non-prescribed element to remove
        removable = [x for x in cur_S if x not in pset]
        if not removable:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Restart from best with new seed
                cur_S = set(best_S)
                cur_sums = get_sums(cur_S)
                cur_size = len(cur_S)
                ls_rng = random.Random(456 + int(time.time() * 1000) % 10000)
                no_improve_count = 0
            continue
            
        y = ls_rng.choice(removable)
        temp_S = cur_S - {y}
        temp_sums = get_sums(temp_S)
        
        valid_adds_temp = [x for x in range(1, MAXV + 1) if x not in temp_S and is_valid_add(x, temp_S, temp_sums)]
        
        if valid_adds_temp:
            # Perform swap - prefer swaps that might enable future additions
            # Score by remaining options after swap
            def swap_score(x):
                new_S = temp_S | {x}
                new_sums = get_sums(new_S)
                return count_valid_adds(new_S, new_sums)
            
            scored_swaps = [(swap_score(x), x) for x in valid_adds_temp[:20]]
            scored_swaps.sort(reverse=True)
            x = scored_swaps[0][1] if scored_swaps else ls_rng.choice(valid_adds_temp)
            
            cur_S = temp_S | {x}
            cur_sums = get_sums(cur_S)
            # Size remains same, but configuration changed
            no_improve_count += 1
        else:
            no_improve_count += 1

    return best_S
