import math

def score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    U_set = set(U)
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    return 1 + math.log2(ratio) / math.log2(2 * max_u + 1)

def construct():
    best_U = [0]
    best_score = 1.0
    
    # Greedy addition
    current_U = [0]
    max_val = 100
    max_size = 10
    
    while len(current_U) < max_size:
        best_candidate = None
        best_candidate_score = best_score
        
        # Try adding next integers
        # We can optimize by only checking a range
        start = max(current_U) + 1
        # Check up to max_val
        for x in range(start, max_val + 1):
            new_U = current_U + [x]
            s = score(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = x
        
        if best_candidate is None:
            break
        
        current_U.append(best_candidate)
        if best_candidate_score > best_score:
            best_score = best_candidate_score
            best_U = current_U.copy()
        else:
            # If score didn't improve, we might stop or continue?
            # Usually score drops after a certain size.
            # Let's stop if we don't improve significantly or at all.
            if best_candidate_score < best_score:
                break

    return best_U
