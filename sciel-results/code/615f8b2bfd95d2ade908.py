import time

def construct():
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i

    # Initial greedy solution
    init_set = []
    is_forbidden = [False] * (N + 1)
    for i in range(1, N + 1):
        if is_forbidden[i]:
            continue
        init_set.append(i)
        for z in init_set[:-1]:
            y1 = 2 * z - i
            if i < y1 <= N:
                is_forbidden[y1] = True
            y2 = 2 * i - z
            if z < y2 <= N:
                is_forbidden[y2] = True
            
    best_sum = sum(weights[i] for i in init_set)
    best_set = list(init_set)
    
    start_time = time.time()
    
    # B&B
    # State: index, current_set (list), current_sum, remaining_weight (sum of weights of k>=index not forbidden)
    # We need to track forbidden state to revert.
    # Let's use a list of (index, old_value) to revert is_forbidden.
    
    def solve(index, current_set, current_sum, remaining_weight):
        nonlocal best_sum, best_set
        
        if time.time() - start_time > 85.0:
            return

        if index > N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return

        # Upper Bound
        if current_sum + remaining_weight <= best_sum:
            return

        # Option 1: Exclude index
        # If we exclude index, it remains available? No, we just skip it.
        # But if it was not forbidden, we just don't pick it.
        # The remaining_weight for the next step should exclude weights[index] if it was not forbidden?
        # No, remaining_weight represents the sum of weights of candidates >= index that are NOT forbidden.
        # If we exclude index, we move to index+1.
        # The set of candidates >= index+1 that are not forbidden is the same as before, 
        # EXCEPT that index is no longer in the range.
        # So new_remaining_weight = remaining_weight - (weights[index] if not is_forbidden[index] else 0)
        
        next_remaining = remaining_weight
        if not is_forbidden[index]:
            next_remaining -= weights[index]
            
        solve(index + 1, current_set, current_sum, next_remaining)

        # Option 2: Include index
        if not is_forbidden[index]:
            # Add index to set
            current_set.append(index)
            
            # Update forbidden set
            changes = []
            for z in current_set[:-1]:
                y1 = 2 * z - index
                if index < y1 <= N and not is_forbidden[y1]:
                    is_forbidden[y1] = True
                    changes.append(y1)
                    # Subtract weight from remaining
                    # Note: y1 > index, so it was included in remaining_weight
                    next_remaining -= weights[y1]
                
                y2 = 2 * index - z
                if z < y2 <= N and not is_forbidden[y2]:
                    is_forbidden[y2] = True
                    changes.append(y2)
                    next_remaining -= weights[y2]
            
            # Recurse
            solve(index + 1, current_set, current_sum + weights[index], next_remaining)
            
            # Backtrack
            current_set.pop()
            for y in changes:
                is_forbidden[y] = False

    # Initial remaining weight
    init_remaining = sum(weights[i] for i in range(1, N + 1) if not is_forbidden[i])
    # Reset is_forbidden for B&B start?
    # The greedy run modified is_forbidden. We need a clean state.
    is_forbidden = [False] * (N + 1)
    init_remaining = sum(weights[i] for i in range(1, N + 1))
    
    solve(1, [], 0.0, init_remaining)

    return best_set
