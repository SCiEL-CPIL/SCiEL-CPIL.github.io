import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    def is_valid_addition(S_set, x):
        """Check if adding x to S_set keeps the set valid.
        Only need to check triples that include x, since triples not including x were already valid."""
        for a in S_set:
            for b in S_set:
                if a == b:
                    continue
                s1 = a + b
                s2 = a + x
                s3 = b + x
                # All three sums must be in S_set ∪ {x}
                in_s1 = (s1 in S_set) or (s1 == x)
                in_s2 = (s2 in S_set) or (s2 == x)
                in_s3 = (s3 in S_set) or (s3 == x)
                if in_s1 and in_s2 and in_s3:
                    return False
        return True

    def try_add_all(S_set):
        """Repeatedly try to add any excluded element until no more can be added."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 129):
                if x in S_set:
                    continue
                if is_valid_addition(S_set, x):
                    S_set.add(x)
                    improved = True
                    break  # Restart loop since S_set changed
        return S_set

    # Phase 1: Greedy construction from largest to smallest
    best_set = set()
    for x in range(128, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(best_set, x):
            best_set.add(x)
    
    # Phase 2: Systematic try-to-add loop
    best_set = try_add_all(set(best_set))
    best_size = len(best_set)
    
    # Phase 3: Local search with multiple random seeds
    for seed in [42, 123, 999, 7, 314]:
        if time.time() > deadline:
            break
        random.seed(seed)
        current_set = set(best_set)
        
        # Randomized local search: try to add excluded elements, or swap
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            candidates = [x for x in range(1, 129) if x not in current_set]
            if not candidates:
                break
            
            x = random.choice(candidates)
            if is_valid_addition(current_set, x):
                current_set.add(x)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(best_set)
            else:
                # Try swapping: remove one element from current_set to allow adding x
                # Only try removing elements that might be blocking x
                # Heuristic: try removing elements in current_set that are "close" to x
                removed = False
                for y in list(current_set):
                    if time.time() > deadline:
                        break
                    # Try removing y and see if x can be added
                    trial_set = current_set - {y}
                    if is_valid_addition(trial_set, x):
                        trial_set.add(x)
                        # Check if this swap actually improves or maintains size
                        # It's a swap so size stays same, but might enable future additions
                        # Do a quick try_add_all on the trial set
                        trial_set = try_add_all(set(trial_set))
                        if len(trial_set) > best_size:
                            best_set = set(trial_set)
                            best_size = len(best_set)
                        current_set = trial_set
                        removed = True
                        break
                # If we didn't remove anything, try a different random candidate
                # (the loop continues)
        
        # After each seed's local search, do a final systematic pass
        if time.time() < deadline:
            current_set = try_add_all(set(current_set))
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(best_set)
    
    # Final: ensure we return the best set found
    if time.time() < deadline - 2:
        best_set = try_add_all(set(best_set))
    
    # Convert to list of Python ints
    return [int(x) for x in sorted(best_set)]
