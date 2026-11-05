import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 5040
    target = 4979
    
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # BFS on (sum, min_index) to ensure distinct divisors
    # State: (current_sum, min_index_to_pick_from)
    # Level 0: {(0, 0): []}
    
    current_level = {(0, 0): []}
    
    max_depth = 12
    
    for depth in range(1, max_depth + 1):
        next_level = {}
        for (s, i), path in current_level.items():
            for j in range(i, m):
                d = divisors[j]
                new_s = s + d
                if new_s > target:
                    continue
                new_i = j + 1
                new_path = path + [d]
                if new_s == target:
                    return new_path
                state = (new_s, new_i)
                if state not in next_level:
                    next_level[state] = new_path
        if not next_level:
            break
        current_level = next_level
        if time.time() > deadline:
            break
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
    return chosen
