import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Precompute divisibility table for speed?
    # div[a][s] is True if a divides s
    # Max sum is 256+256=512
    # Max a is 256
    # div = [[False]*(N+1) for _ in range(N+1)] # Not quite, sum goes to 512
    # We can just use % which is fast enough for N=256.
    
    def is_valid_addition(s_sorted, candidate):
        """
        Check if adding 'candidate' to 's_sorted' (sorted list) keeps validity.
        Conditions:
        1. candidate as a: a < b <= c. a=cand. b,c in s_sorted. b > cand, c >= b.
           Check: cand does not divide b+c.
        2. candidate as b: a < b <= c. b=cand. a in s_sorted, c in s_sorted U {cand}.
           a < cand, c >= cand.
           Check: a does not divide cand+c.
        3. candidate as c: a < b <= c. c=cand. a,b in s_sorted.
           a < b <= cand.
           Check: a does not divide b+cand.
        """
        n = len(s_sorted)
        pos = bisect.bisect_left(s_sorted, candidate)
        
        # 1. candidate as 'a' (smallest)
        # b, c are from s_sorted[pos:] (since s_sorted[pos] > candidate)
        # b > candidate, c >= b
        for i in range(pos, n):
            b = s_sorted[i]
            # b must be > candidate, which is true since s_sorted is sorted and candidate not in it
            target = (-b) % candidate
            # Check if any c in s_sorted[i:] has c % candidate == target
            # c >= b
            for k in range(i, n):
                if s_sorted[k] % candidate == target:
                    return False
        
        # 2. candidate as 'b' (middle)
        # a from s_sorted[:pos] (a < candidate)
        # c from s_sorted[pos:] U {candidate} (c >= candidate)
        for i in range(pos):
            a = s_sorted[i]
            # c can be candidate
            if (candidate + candidate) % a == 0:
                return False
            # c can be s_sorted[k] for k >= pos
            target = (-candidate) % a
            for k in range(pos, n):
                if s_sorted[k] % a == target:
                    return False
        
        # 3. candidate as 'c' (largest)
        # a, b from s_sorted such that a < b <= candidate
        # b <= candidate implies b is in s_sorted[:pos] (since s_sorted[pos] > candidate)
        # Wait, s_sorted[:pos] contains elements < candidate.
        # So b must be in s_sorted[:pos].
        # a must be < b.
        for i in range(pos):
            a = s_sorted[i]
            for j in range(i + 1, pos):
                b = s_sorted[j]
                # b <= candidate is guaranteed since b < candidate
                if (b + candidate) % a == 0:
                    return False
        
        return True

    def try_add(s_sorted, candidate):
        if is_valid_addition(s_sorted, candidate):
            pos = bisect.bisect_left(s_sorted, candidate)
            new_s = s_sorted[:]
            new_s.insert(pos, candidate)
            return True, new_s
        return False, s_sorted

    def greedy_high_to_low():
        s = []
        for i in range(N, 0, -1):
            ok, new_s = try_add(s, i)
            if ok:
                s = new_s
        return s

    best_set = greedy_high_to_low()
    best_len = len(best_set)
    
    # Local Search / Perturbation
    # Strategy: 
    # 1. Try to add any missing element to best_set.
    # 2. If stuck, perturb: remove k random elements, then greedily add back.
    
    current_set = best_set[:]
    current_set_set = set(current_set)
    
    # Initial pass: try to add any missing element
    for candidate in range(1, N + 1):
        if candidate not in current_set_set:
            ok, new_s = try_add(current_set, candidate)
            if ok:
                current_set = new_s
                current_set_set.add(candidate)
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)

    # Iterative Improvement
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < deadline:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # 1. Try to add any single missing element
        # Randomize order to avoid bias
        missing = [x for x in range(1, N + 1) if x not in current_set_set]
        random.shuffle(missing)
        
        for candidate in missing:
            ok, new_s = try_add(current_set, candidate)
            if ok:
                current_set = new_s
                current_set_set.add(candidate)
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)
                break # Restart the add loop with new set state
        
        if improved:
            no_improve_count = 0
            continue
            
        # 2. No single add works. Try Perturbation.
        # Remove k random elements
        k = random.randint(2, 5)
        if len(current_set) <= k:
            k = max(1, len(current_set) // 2)
            
        # Choose elements to remove. 
        # Heuristic: Remove elements that are "large" (less constraining as 'a') 
        # or "small" (more constraining). 
        # Removing small 'a's might allow adding many others?
        # Let's just remove random ones for diversity.
        remove_indices = random.sample(range(len(current_set)), k)
        remove_indices.sort(reverse=True)
        
        perturbed_set = current_set[:]
        perturbed_set_set = set(perturbed_set)
        for idx in remove_indices:
            val = perturbed_set.pop(idx)
            perturbed_set_set.discard(val)
            
        # Greedily add back from missing elements
        # Sort missing by descending value? Or random?
        # Descending value often works well for this type of problem (larger numbers have fewer multiples).
        all_missing = [x for x in range(1, N + 1) if x not in perturbed_set_set]
        random.shuffle(all_missing) # Randomize to explore different basins
        
        for candidate in all_missing:
            if time.time() - start_time > deadline:
                break
            ok, new_s = try_add(perturbed_set, candidate)
            if ok:
                perturbed_set = new_s
                perturbed_set_set.add(candidate)
                
        if len(perturbed_set) > len(current_set):
            current_set = perturbed_set
            current_set_set = perturbed_set_set
            improved = True
            if len(current_set) > best_len:
                best_set = current_set[:]
                best_len = len(current_set)
        elif len(perturbed_set) == len(current_set) and random.random() < 0.1:
            # Accept same size with some probability to escape local minima?
            # Maybe not, as we want to maximize size.
            pass
        else:
            no_improve_count += 1
            
        # Also try a "Swap" heuristic: Remove 1, Add 1 (if it helps future adds)
        # This is harder to evaluate. Skip for now.
        
    return best_set
