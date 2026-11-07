import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    
    def is_valid(s_set):
        """Check if no three distinct a,b,c in s_set have all pair sums in s_set."""
        elems = sorted(s_set)
        n = len(elems)
        in_set = set(elems)
        
        # For each pair (a,b) with a<b, check if a+b is in set.
        # If so, for each c > b, check if a+c and b+c are both in set.
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                s_ab = a + b
                if s_ab > N or s_ab not in in_set:
                    continue
                # If s_ab is in set, check for c > b
                for k in range(j + 1, n):
                    c = elems[k]
                    if (a + c) in in_set and (b + c) in in_set:
                        return False
        return True

    def is_valid_fast(s_set):
        """Faster validity check using precomputed sums."""
        in_set = set(s_set)
        elems = sorted(s_set)
        n = len(elems)
        
        # Precompute: for each element, which other elements form a sum that's in the set
        # Key insight: for a violation with a<b<c, we need a+b, a+c, b+c all in set.
        # Since b+c > a+c > a+b (for positive a<b<c), and all must be in set.
        # For fixed a,b: if a+b in set, then for any c>b, check a+c and b+c.
        
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                s_ab = a + b
                if s_ab > N:
                    break  # Since elems is sorted, larger b will only increase s_ab
                if s_ab not in in_set:
                    continue
                # Now check all c > b
                for k in range(j + 1, n):
                    c = elems[k]
                    s_ac = a + c
                    if s_ac > N:
                        break
                    if s_ac not in in_set:
                        continue
                    s_bc = b + c
                    if s_bc in in_set:
                        return False
        return True

    # Start with upper half
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    random.seed(42)
    
    def greedy_add_small(current_set):
        """Try to add small numbers greedily."""
        current = set(current_set)
        for x in range(1, N // 2 + 1):
            if x in current:
                continue
            trial = current | {x}
            if is_valid_fast(trial):
                current = trial
        return current

    # Strategy 1: Greedy add small numbers to upper half
    s1 = greedy_add_small(set(range(N // 2 + 1, N + 1)))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    # Local search with better moves
    def local_search(current_set, max_steps=2000):
        current = set(current_set)
        steps = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            improved = False
            
            # Try adding elements - prioritize smaller ones (more likely to fit)
            candidates = [x for x in range(1, N + 1) if x not in current]
            # Sort by value to try smaller first (they're more likely to be addable)
            candidates.sort()
            random.shuffle(candidates[10:])  # Keep first 10 sorted, shuffle rest
            
            for elem in candidates:
                trial = current | {elem}
                if is_valid_fast(trial):
                    current = trial
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swap: remove one, add one
            if len(current) > 0:
                to_remove_list = list(current)
                random.shuffle(to_remove_list)
                found_swap = False
                for to_remove in to_remove_list:
                    trial_removed = current - {to_remove}
                    candidates = [x for x in range(1, N + 1) if x not in trial_removed]
                    random.shuffle(candidates)
                    for elem in candidates:
                        trial = trial_removed | {elem}
                        if is_valid_fast(trial):
                            current = trial
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
            
            # If still no improvement, try remove 1 add 2
            if not improved and len(current) > 0:
                to_remove = random.choice(list(current))
                trial_removed = current - {to_remove}
                candidates = [x for x in range(1, N + 1) if x not in trial_removed]
                random.shuffle(candidates)
                
                new_current = set(trial_removed)
                added = 0
                for elem in candidates:
                    if added == 2:
                        break
                    trial_add = new_current | {elem}
                    if is_valid_fast(trial_add):
                        new_current = trial_add
                        added += 1
                
                if len(new_current) > len(current):
                    current = new_current
        
        return current

    # Multiple restarts with different strategies
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            # Upper half + greedy small
            init = greedy_add_small(set(range(N // 2 + 1, N + 1)))
        elif restart == 1:
            # Try starting from 50
            init = set(range(50, N + 1))
            while not is_valid_fast(init):
                to_remove = random.choice(list(init))
                init.remove(to_remove)
            init = greedy_add_small(init)
        elif restart == 2:
            # Try starting from 40
            init = set(range(40, N + 1))
            while not is_valid_fast(init):
                to_remove = random.choice(list(init))
                init.remove(to_remove)
            init = greedy_add_small(init)
        elif restart == 3:
            # Even numbers above 32 + some odds
            init = set(x for x in range(33, N + 1) if x % 2 == 0)
            while not is_valid_fast(init):
                to_remove = random.choice(list(init))
                init.remove(to_remove)
            init = greedy_add_small(init)
        else:
            # Random subset of upper half + some lower
            init = set()
            for x in range(N // 2 + 1, N + 1):
                if random.random() < 0.85:
                    init.add(x)
            for x in range(1, N // 2 + 1):
                if random.random() < 0.3:
                    init.add(x)
            
            while not is_valid_fast(init):
                if len(init) == 0:
                    break
                to_remove = random.choice(list(init))
                init.remove(to_remove)
            init = greedy_add_small(init)
        
        improved_set = local_search(init)
        
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final polish: try to add any remaining element
    remaining = sorted(set(range(1, N + 1)) - best_set)
    for elem in remaining:
        if time.time() > deadline - 2:
            break
        trial = best_set | {elem}
        if is_valid_fast(trial):
            best_set = trial
    
    # Ensure return type is correct
    result = list(best_set)
    result = [int(x) for x in result]
    return result
