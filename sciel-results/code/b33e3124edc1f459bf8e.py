"""Sidon set construction using CSP backtracking with forward checking and bitmask constraints."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    MAX_VAL = 284
    MAX_SUM = 2 * MAX_VAL  # 568
    
    # Precompute initial sums for prescribed elements
    initial_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            initial_sums.add(x + y)
    
    def can_add_to_sums(x, current_set, current_sums):
        """Check if x can be added to current_set maintaining Sidon property."""
        # Check 2*x
        if 2 * x in current_sums:
            return False
        # Check x + y for all y in current_set
        for y in current_set:
            if x + y in current_sums:
                return False
        return True
    
    def get_new_sums(x, current_set):
        """Get the new sums added when x is added to current_set."""
        return {x + y for y in current_set} | {2 * x}
    
    def build_sums(s):
        """Build all pairwise sums for a set."""
        sl = sorted(s)
        sums = set()
        for i, x in enumerate(sl):
            for y in sl[i:]:
                sums.add(x + y)
        return sums
    
    # Strategy 1: Systematic backtracking with forward checking
    # We'll decide elements in a specific order, trying to include each one
    # If we can't include it, we skip it (this gives us maximal sets)
    
    def backtracking_csp(order, time_limit=30.0):
        """
        CSP backtracking where 'order' determines the sequence of candidate elements.
        For each element in order, we try to include it if possible.
        This naturally finds a maximal Sidon set.
        """
        nonlocal best_size, best_solution
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break
            if x in current_set:
                continue
            if can_add_to_sums(x, current_set, current_sums):
                current_set.add(x)
                current_sums.update(get_new_sums(x, current_set))
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = list(current_set)
        
        return len(current_set)
    
    # Strategy 2: Local search with swap operations
    def local_search(start_set, time_limit=60.0):
        """Local search trying to improve by swapping elements."""
        nonlocal best_size, best_solution
        current_set = set(start_set)
        current_sums = build_sums(current_set)
        start = time.time()
        rng = random.Random(12345)
        
        all_candidates = list(range(1, MAX_VAL + 1))
        removable = [x for x in current_set if x not in prescribed_set]
        
        for iteration in range(500000):
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break
            
            # Try to add a random element
            added = False
            for _attempt in range(20):
                x = rng.choice(all_candidates)
                if x not in current_set:
                    if can_add_to_sums(x, current_set, current_sums):
                        current_set.add(x)
                        current_sums.update(get_new_sums(x, current_set))
                        added = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_solution = list(current_set)
                        break
            
            # Try swap: remove one non-prescribed, add one new
            if not added and removable:
                x_remove = rng.choice(removable)
                remaining_set = current_set - {x_remove}
                remaining_sums = build_sums(remaining_set)
                
                for _attempt in range(30):
                    x_add = rng.choice(all_candidates)
                    if x_add in current_set:
                        continue
                    if can_add_to_sums(x_add, remaining_set, remaining_sums):
                        # Valid swap
                        current_set = remaining_set | {x_add}
                        current_sums = remaining_sums | get_new_sums(x_add, remaining_set)
                        removable = [x for x in current_set if x not in prescribed_set]
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_solution = list(current_set)
                        break
        
        return len(current_set)
    
    # Strategy 3: Greedy with multiple orderings
    def greedy_with_order(order, time_limit=5.0):
        """Greedy construction with given element order."""
        nonlocal best_size, best_solution
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break
            if x in current_set:
                continue
            if can_add_to_sums(x, current_set, current_sums):
                current_set.add(x)
                current_sums.update(get_new_sums(x, current_set))
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = list(current_set)
        
        return len(current_set)
    
    best_size = len(prescribed)
    best_solution = list(prescribed)
    
    # Phase 1: Quick greedy passes with various orderings
    candidates = [x for x in range(1, MAX_VAL + 1) if x not in prescribed_set]
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        order = candidates[:]
        rng.shuffle(order)
        greedy_with_order(order, time_limit=1.0)
    
    # Phase 2: More structured orderings
    # Smallest first
    if time.time() < deadline - 20:
        greedy_with_order(sorted(candidates), time_limit=2.0)
    
    # Largest first
    if time.time() < deadline - 20:
        greedy_with_order(sorted(candidates, reverse=True), time_limit=2.0)
    
    # Spread out: prefer elements far from prescribed
    if time.time() < deadline - 20:
        def spread_score(x):
            return min(abs(x - e) for e in prescribed)
        order = sorted(candidates, key=spread_score, reverse=True)
        greedy_with_order(order, time_limit=2.0)
    
    # Phase 3: Local search from best found
    if best_size > len(prescribed) and time.time() < deadline - 10:
        local_search(best_solution, time_limit=40.0)
    
    # Phase 4: More randomized greedy with local search refinement
    if time.time() < deadline - 15:
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            order = candidates[:]
            rng.shuffle(order)
            size = greedy_with_order(order, time_limit=1.0)
            if size >= 15 and time.time() < deadline - 10:
                local_search(best_solution, time_limit=15.0)
    
    # Phase 5: Final local search if we have a decent solution
    if best_size >= 12 and time.time() < deadline - 5:
        local_search(best_solution, time_limit=30.0)
    
    return best_solution
