import math
import time

def score_of(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Initial candidates from the seed
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
    ]
    
    best_set = list(candidates[0])
    best_score = score_of(best_set)
    
    for U in candidates:
        s = score_of(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Greedy construction: build a set by adding elements that maximize score
    # Start from a base and try adding new elements
    base_sets = [
        [0, 1, 4],
        [0, 1, 3],
        [0, 2, 5],
    ]
    
    for base in base_sets:
        if time.time() > deadline - 5:
            break
        current = list(base)
        current_score = score_of(current)
        if current_score > best_score:
            best_score = current_score
            best_set = list(current)
        
        # Greedily add elements
        for _ in range(20):  # Try to add up to 20 more elements
            if time.time() > deadline - 5:
                break
            max_u = max(current)
            # Try candidates: extend beyond current max
            best_candidate = None
            best_candidate_score = current_score
            
            # Try various extension strategies
            for mult in [2, 3, 5, 8, 13]:
                candidate_val = max_u * mult + 1
                if candidate_val < 100000:  # Keep max reasonable
                    test_set = current + [candidate_val]
                    s = score_of(test_set)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate_val
            
            # Also try some random-ish extensions
            for offset in [1, 2, 3, 5, 7, 11]:
                candidate_val = max_u + offset * (max_u // 2 + 1)
                if candidate_val < 100000:
                    test_set = current + [candidate_val]
                    s = score_of(test_set)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate_val
            
            if best_candidate is not None:
                current.append(best_candidate)
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            else:
                break
    
    # Local search: try removing or replacing elements
    if time.time() > deadline - 10:
        return best_set
    
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        if time.time() > deadline - 10:
            break
        improved = False
        iteration += 1
        
        # Try removing each element
        for i in range(len(best_set)):
            if time.time() > deadline - 10:
                break
            if best_set[i] == 0:
                continue  # Must keep 0
            test_set = [x for j, x in enumerate(best_set) if j != i]
            s = score_of(test_set)
            if s > best_score:
                best_score = s
                best_set = test_set
                improved = True
                break
        
        # Try replacing each element (except 0) with a nearby value
        for i in range(len(best_set)):
            if time.time() > deadline - 10:
                break
            if best_set[i] == 0:
                continue
            orig_val = best_set[i]
            # Try nearby values
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20]:
                new_val = orig_val + delta
                if new_val < 0:
                    continue
                if new_val in best_set:
                    continue
                test_set = list(best_set)
                test_set[i] = new_val
                s = score_of(test_set)
                if s > best_score:
                    best_score = s
                    best_set = test_set
                    improved = True
                    break
    
    # Final: try adding one more element to the best set
    if time.time() > deadline - 5:
        return best_set
    
    max_u = max(best_set)
    best_add_score = best_score
    best_add_val = None
    
    for mult in [2, 3, 5, 8, 13, 21]:
        candidate_val = max_u * mult + 1
        if candidate_val < 200000:
            test_set = best_set + [candidate_val]
            s = score_of(test_set)
            if s > best_add_score:
                best_add_score = s
                best_add_val = candidate_val
    
    for offset in [1, 2, 3, 5, 8, 13]:
        candidate_val = max_u + offset * (max_u // 3 + 1)
        if candidate_val < 200000:
            test_set = best_set + [candidate_val]
            s = score_of(test_set)
            if s > best_add_score:
                best_add_score = s
                best_add_val = candidate_val
    
    if best_add_val is not None and best_add_score > best_score:
        best_set.append(best_add_val)
        best_score = best_add_score
    
    return best_set
