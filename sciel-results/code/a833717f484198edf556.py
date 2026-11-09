import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 192
    max_size = 48
    
    # Helper to check if a set is valid
    def is_valid(set_list):
        s = set(set_list)
        for a in s:
            # Check if a divides sum of any 3 distinct others
            others = [x for x in s if x != a]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True

    # Greedy construction
    def greedy(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = selected + [x]
            if is_valid(candidate):
                selected.append(x)
            if len(selected) == max_size:
                break
        return selected

    # Local search / Repair
    def local_search(initial_set):
        current = set(initial_set)
        if len(current) == 0:
            return current
            
        # If already valid and max size, return
        if is_valid(list(current)) and len(current) == max_size:
            return current
            
        # Try to improve
        best_set = set(current)
        best_score = len(best_set) - 100 * (not is_valid(list(best_set)))
        
        # We need a scoring function that penalizes violations
        def get_violations(s_list):
            s = set(s_list)
            violations = 0
            for a in s:
                others = [x for x in s if x != a]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        violations += 1
            return violations

        best_violations = get_violations(list(best_set))
        
        # Simple hill climbing: swap elements
        # Candidates: all numbers 1..192 not in current set
        candidates = [x for x in range(1, n+1) if x not in current]
        
        while time.time() < deadline:
            improved = False
            # Randomly pick an element in current to remove
            if not current:
                break
            x_in = random.choice(list(current))
            # Randomly pick a candidate to add
            if not candidates:
                break
            x_out = random.choice(candidates)
            
            new_set = set(current)
            new_set.remove(x_in)
            new_set.add(x_out)
            
            new_violations = get_violations(list(new_set))
            new_size = len(new_set)
            
            # Accept if fewer violations, or same violations but larger size?
            # We want 0 violations.
            if new_violations < best_violations:
                current = new_set
                best_violations = new_violations
                if best_violations == 0 and len(current) == max_size:
                    return current
                improved = True
            elif new_violations == best_violations and new_size > len(best_set):
                # This shouldn't happen if we maintain size
                pass
                
            if not improved:
                # Stuck, try a different strategy or reset
                # Maybe just keep trying random swaps
                pass

        return current

    # Strategy 1: Greedy with large numbers first
    order1 = list(range(n, 0, -1))
    set1 = greedy(order1)
    
    if len(set1) == max_size and is_valid(set1):
        return set1

    # Strategy 2: Greedy with random order
    best_found = set1
    for _ in range(100):
        if time.time() > deadline:
            break
        order2 = list(range(1, n+1))
        random.shuffle(order2)
        set2 = greedy(order2)
        if len(set2) > len(best_found):
            best_found = set2
        if len(set2) == max_size and is_valid(set2):
            return set2

    # Strategy 3: Local search on best_found
    if len(best_found) < max_size:
        # Try to repair
        # This part is tricky. Let's just return the best found.
        # The greedy might be enough.
        pass

    return best_found
