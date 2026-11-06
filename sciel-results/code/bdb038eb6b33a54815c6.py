import time

def construct():
    N = 256
    recip = [0.0] * (N + 1)
    prefix_sum = [0.0] * (N + 2)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i-1] + recip[i]
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_weight = 0.0
    best_set = []
    
    def is_valid_addition(x, S_set):
        for a in S_set:
            b = 2 * x - a
            if b in S_set:
                return False
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if b in S_set:
                    return False
            b = 2 * a - x
            if b in S_set:
                return False
        return True

    # Greedy
    greedy_set_set = set()
    for x in range(1, N + 1):
        if is_valid_addition(x, greedy_set_set):
            greedy_set_set.add(x)
    
    best_weight = sum(1.0 / x for x in greedy_set_set)
    best_set = list(greedy_set_set)
    
    def search(idx, S_set, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        # Simple Upper Bound
        ub_simple = current_weight + (prefix_sum[N] - prefix_sum[idx - 1])
        if ub_simple <= best_weight:
            return
        
        if idx > N:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(S_set)
            return
        
        x = idx
        
        # Option 1: Include x
        if is_valid_addition(x, S_set):
            # Calculate tighter bound for this branch
            # Blocked values y > x due to x and existing S_set
            blocked_sum = 0.0
            blocked_values = set()
            for z in S_set:
                # y = 2x - z
                y1 = 2 * x - z
                if y1 > x and y1 <= N:
                    blocked_values.add(y1)
                # y = (x + z) / 2
                if (x + z) % 2 == 0:
                    y2 = (x + z) // 2
                    if y2 > x and y2 <= N:
                        blocked_values.add(y2)
                # y = 2z - x
                y3 = 2 * z - x
                if y3 > x and y3 <= N:
                    blocked_values.add(y3)
            
            # Note: Some of these y might already be invalid due to S_set alone.
            # But subtracting them from the bound is safe (conservative) even if they were already invalid,
            # because they are definitely not added in this branch.
            # However, if they were already invalid, the simple bound didn't include them anyway?
            # No, the simple bound `prefix_sum` includes ALL y.
            # So we subtract the weights of these specific blocked y's.
            for y in blocked_values:
                blocked_sum += recip[y]
            
            ub_include = current_weight + recip[x] + (prefix_sum[N] - prefix_sum[idx]) - blocked_sum
            
            if ub_include > best_weight:
                new_set = set(S_set)
                new_set.add(x)
                search(idx + 1, new_set, current_weight + recip[x])
        
        # Option 2: Exclude x
        # The bound is the same as ub_simple, but we are at idx+1
        # We can just call search, which will compute the bound again.
        search(idx + 1, S_set, current_weight)
        
    search(1, set(), 0.0)
    
    return best_set
