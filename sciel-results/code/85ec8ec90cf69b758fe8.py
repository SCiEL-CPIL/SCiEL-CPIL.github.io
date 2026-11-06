"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 78.0  # safety margin before 90s kill
    n = 128
    
    # Precompute lcm table
    lcm_table = {}
    for a in range(1, n+1):
        for b in range(a, n+1):
            lcm_table[(a,b)] = a * b // gcd(a,b)
            if a != b:
                lcm_table[(b,a)] = a * b // gcd(a,b)
    
    def get_lcm(a, b):
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return lcm_table.get((a,b), a * b // gcd(a,b))
    
    # Precompute for each x, the set of pairs (a,b) that would form a bad triple with x
    # x is blocked by pair (a,b) if lcm(a,b) == lcm(a,x) == lcm(b,x)
    blocking_pairs = {}
    for x in range(1, n+1):
        blocking_pairs[x] = set()
        for a in range(1, n+1):
            if a == x:
                continue
            for b in range(a+1, n+1):
                if b == x:
                    continue
                if get_lcm(a,b) == get_lcm(a,x) == get_lcm(b,x):
                    blocking_pairs[x].add((min(a,b), max(a,b)))
    
    # Store blocking pairs as lists for faster iteration
    blocking_pairs_list = {}
    for x in range(1, n+1):
        blocking_pairs_list[x] = list(blocking_pairs[x])
    
    def can_add_to_set(x, s):
        for (a, b) in blocking_pairs_list[x]:
            if a in s and b in s:
                return False
        return True
    
    def greedy_with_order(order, s=None):
        if s is None:
            s = set()
        for x in order:
            if x in s:
                continue
            if can_add_to_set(x, s):
                s.add(x)
        return s
    
    # Try multiple orderings
    best = None
    best_size = 0
    
    # 1. Natural order
    s1 = greedy_with_order(list(range(1, n+1)))
    if len(s1) > best_size:
        best = s1
        best_size = len(s1)
    
    # 2. Reverse order
    s2 = greedy_with_order(list(range(n, 0, -1)))
    if len(s2) > best_size:
        best = s2
        best_size = len(s2)
    
    # 3. Degree-based: process elements with fewer blocking pairs first
    deg_order = sorted(range(1, n+1), key=lambda x: len(blocking_pairs_list[x]))
    s3 = greedy_with_order(deg_order)
    if len(s3) > best_size:
        best = s3
        best_size = len(s3)
    
    # 4. Reverse degree-based
    deg_order_rev = sorted(range(1, n+1), key=lambda x: -len(blocking_pairs_list[x]))
    s4 = greedy_with_order(deg_order_rev)
    if len(s4) > best_size:
        best = s4
        best_size = len(s4)
    
    # 5. Random orderings with time budget
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # 6. Kick-and-refill local improvement with optimized refill
    if best and time.time() < deadline:
        current = set(best)
        best_size = len(current)
        
        def refill(s, order=None):
            """Optimized greedy refill: add all elements that can be added."""
            if order is None:
                order = deg_order
            changed = True
            while changed:
                changed = False
                for y in order:
                    if y not in s and can_add_to_set(y, s):
                        s.add(y)
                        changed = True
            return s
        
        # Initial refill
        current = refill(set(best), deg_order)
        best_size = len(current)
        
        # Simulated Annealing style local search
        # We allow moving to worse solutions with a probability that decreases over time
        temperature = 1.0
        cooling_rate = 0.999
        min_temperature = 0.01
        
        max_trials = 100000
        trial = 0
        
        while trial < max_trials and time.time() < deadline:
            trial += 1
            
            # Choose a kick strategy
            # 1. Random single element removal
            # 2. Random pair removal
            # 3. "Worst" element removal (highest degree in conflict graph)
            
            r = rng.random()
            kick_set = set()
            current_list = list(current)
            
            if r < 0.5:
                # Random single element
                kick_set.add(rng.choice(current_list))
            elif r < 0.8:
                # Random pair
                if len(current_list) >= 2:
                    a, b = rng.sample(current_list, 2)
                    kick_set.add(a)
                    kick_set.add(b)
            else:
                # Remove element with most blocking pairs (heuristic)
                # Or random if we don't want to sort every time
                # Let's just do random for speed, but maybe bias towards "central"
                # For now, random is safer to avoid getting stuck in a specific bad local opt
                kick_set.add(rng.choice(current_list))
            
            candidate = current - kick_set
            candidate = refill(candidate, deg_order)
            
            delta = len(candidate) - len(current)
            
            if delta > 0:
                current = candidate
                if len(candidate) > best_size:
                    best_size = len(candidate)
                temperature = max(min_temperature, temperature * cooling_rate)
            elif delta == 0:
                # Accept equal with some probability to explore
                if rng.random() < 0.1:
                    current = candidate
                temperature = max(min_temperature, temperature * cooling_rate)
            else:
                # Accept worse with probability exp(delta / temperature)
                # Since delta is negative, this is exp(-abs(delta)/temp)
                prob = 2.718281828 ** (delta / temperature)
                if rng.random() < prob:
                    current = candidate
                temperature = max(min_temperature, temperature * cooling_rate)
            
            # Reset temperature occasionally if we are stuck for a long time?
            # Or just let it cool down.
    
    if best is None or len(best) == 0:
        return [1]
    return sorted(best)
# EVOLVE-BLOCK-END
