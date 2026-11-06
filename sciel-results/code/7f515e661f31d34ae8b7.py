"""Combined solver for Sidon set extension containing {1, 2, 5, 11} in [1, 284]."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s budget, 20s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    # Pre-compute initial sums for prescribed set
    initial_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            initial_sums.add(x + y)
    
    # Best found so far
    best_set = set(prescribed)
    best_size = len(best_set)
    best_sums = set(initial_sums)
    
    def build_sums(s):
        """Build the set of all pairwise sums a+b for a<=b in s."""
        sl = sorted(s)
        sums = set()
        for i, x in enumerate(sl):
            for y in sl[i:]:
                sums.add(x + y)
        return sums
    
    def can_add(x, current_set, current_sums):
        """Check if x can be added to current_set with given sums."""
        if x in current_set:
            return False
        new_sums = {x + y for y in current_set} | {2 * x}
        if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
            return True
        return False
    
    def get_new_sums(x, current_set):
        return {x + y for y in current_set} | {2 * x}
    
    def greedy_forward(order, time_limit=2.0):
        """Greedy forward construction with given order."""
        nonlocal best_set, best_size, best_sums
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set)
                current_set.add(x)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
        
        return len(current_set)
    
    def local_search(start_set, time_limit=30.0):
        """Fast local search with incremental sum maintenance."""
        nonlocal best_set, best_size, best_sums
        current_set = set(start_set)
        current_sums = build_sums(current_set)
        start = time.time()
        all_candidates = list(range(1, 285))
        
        while time.time() - start < time_limit and time.time() < deadline - 5:
            # Try to add a random element
            if len(current_set) < 284:
                for _attempt in range(20):
                    x = random.choice(all_candidates)
                    if x not in current_set:
                        if can_add(x, current_set, current_sums):
                            new_sums = get_new_sums(x, current_set)
                            current_set.add(x)
                            current_sums.update(new_sums)
                            if len(current_set) > best_size:
                                best_set = set(current_set)
                                best_size = len(current_set)
                                best_sums = set(current_sums)
                        break
            
            # Try to swap: remove one non-prescribed, add one new
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                x_remove = random.choice(removable)
                remaining_set = current_set - {x_remove}
                reduced_sums = build_sums(remaining_set)
                
                for _attempt in range(15):
                    x_add = random.choice(all_candidates)
                    if x_add in current_set:
                        continue
                    new_sums_test = get_new_sums(x_add, remaining_set)
                    if len(new_sums_test) == len(remaining_set) + 1 and not (new_sums_test & reduced_sums):
                        # Valid swap
                        current_set = remaining_set | {x_add}
                        current_sums = reduced_sums | new_sums_test
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)
                            best_sums = set(current_sums)
                        break
        
        return len(current_set)
    
    def greedy_lookahead(time_limit=5.0):
        """Greedy with lookahead - try multiple candidates at each step."""
        nonlocal best_set, best_size, best_sums
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        while time.time() - start < time_limit and time.time() < deadline - 5:
            # Find all valid additions
            valid_adds = []
            for x in range(1, 285):
                if x not in current_set and can_add(x, current_set, current_sums):
                    valid_adds.append(x)
            
            if not valid_adds:
                break
            
            # Pick the one that leaves the most future options (heuristic: pick smallest)
            x_add = min(valid_adds)
            new_sums = get_new_sums(x_add, current_set)
            current_set.add(x_add)
            current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
        
        return len(current_set)
    
    # Strategy 1: Multiple greedy passes with different random orders
    remaining = [x for x in range(12, 285) if x not in prescribed_set]
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        order = remaining[:]
        rng.shuffle(order)
        greedy_forward(order, time_limit=1.0)
    
    # Strategy 2: Systematic greedy (smallest first)
    if time.time() < deadline - 20:
        order = list(range(12, 285))
        greedy_forward(order, time_limit=1.0)
    
    # Strategy 3: Systematic greedy (largest first)
    if time.time() < deadline - 20:
        order = list(range(284, 11, -1))
        greedy_forward(order, time_limit=1.0)
    
    # Strategy 4: Greedy with lookahead
    if time.time() < deadline - 15:
        greedy_lookahead(time_limit=5.0)
    
    # Strategy 5: Local search from best found
    if best_size > len(prescribed) and time.time() < deadline - 10:
        local_search(best_set, time_limit=30.0)
    
    # Strategy 6: More local search restarts from perturbed best
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb best set
            start = set(best_set)
            n_remove = random.randint(1, min(3, len(start) - 4))
            if len(start) > 4:
                to_remove = set(random.sample([x for x in start if x not in prescribed_set], n_remove))
                start -= to_remove
            
            local_search(start, time_limit=15.0)
    
    # Final validation
    def _validate(a):
        """Check that a is a valid Sidon set containing {1,2,5,11} with elements in 1..284."""
        if len(a) != len(set(a)):
            return False
        for x in a:
            if x < 1 or x > 284:
                return False
        for req in (1, 2, 5, 11):
            if req not in a:
                return False
        # Check all pairwise sums a+b (a<=b) are distinct
        sums = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s = a[i] + a[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    if not _validate(list(best_set)):
        best_set = set(prescribed)
        best_size = len(best_set)
    
    return sorted(best_set)
