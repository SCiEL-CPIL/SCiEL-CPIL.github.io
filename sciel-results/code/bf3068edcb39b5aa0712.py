import random
import time

def construct():
    """
    Find a large subset of {1, ..., 1000} such that every integer has at most
    two representations a + b with a <= b from the set. Maximize the size.
    
    Strategy:
    1. Use a greedy approach with a priority queue to select elements that
       cause the least "conflict" (i.e., sums that are already close to being full).
    2. Use randomization with multiple restarts to escape local optima.
    3. Use a smarter initialization: start with a structured set (like multiples
       of a small number) and then greedily add elements.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for overhead
    
    def run_greedy(order, initial_set=None):
        current = list(initial_set) if initial_set else []
        counts = {}
        full_sums = set()
        
        # Initialize counts with the initial set
        if initial_set:
            for i in range(len(current)):
                for j in range(i, len(current)):
                    s = current[i] + current[j]
                    counts[s] = counts.get(s, 0) + 1
            for s, c in counts.items():
                if c >= 2:
                    full_sums.add(s)
        
        for x in order:
            if x in current:
                continue
                
            # Check if adding x is valid
            valid = True
            new_sums = []
            
            # Check against existing full sums
            for y in current:
                s = x + y
                if s in full_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                s_double = 2 * x
                if s_double in full_sums:
                    valid = False
                else:
                    new_sums.append(s_double)
            
            if not valid:
                continue
            
            # Check if any new sum collides with another new sum?
            # x+y1 == x+y2 => y1==y2. Distinct y gives distinct sums.
            # x+y == 2x => y=x. But y in current, x not in current. So distinct.
            # So new_sums are all distinct.
            
            # Check if adding these sums causes any count to exceed 2
            # We already checked against full_sums.
            # Now we need to check if any s in new_sums is already in counts with count 1?
            # If count is 1, it becomes 2 (full). That's allowed.
            # If count is 0, it becomes 1. That's allowed.
            # So we just need to update.
            
            # Update state
            current.append(x)
            for s in new_sums:
                c = counts.get(s, 0)
                if c == 1:
                    full_sums.add(s)
                    counts[s] = 2
                else:
                    counts[s] = 1
        
        return current

    best_set = []
    best_size = 0

    # Strategy 1: Randomized greedy with random orders
    for _ in range(20):  # Multiple random restarts
        order = list(range(1, 1001))
        random.shuffle(order)
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # Strategy 2: Start with a structured set (multiples of k) and then greedily add
    for k in [2, 3, 4, 5]:
        initial = list(range(k, 1001, k))
        order = [x for x in range(1, 1001) if x % k != 0]
        random.shuffle(order)
        res = run_greedy(order, initial)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # Strategy 3: Greedy with priority - pick elements that cause least conflicts
    # We'll use a heuristic: prefer smaller elements first (they create smaller sums)
    # and also prefer elements that are "far" from existing elements
    for _ in range(10):
        order = sorted(range(1, 1001), key=lambda x: random.random() * x)
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    return best_set
