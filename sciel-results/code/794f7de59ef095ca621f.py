import time

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i

    n = 24
    elements = list(range(1, 25)) # 1 to 24
    
    best_set = set()
    best_size = 0
    
    # 1. Greedy Initialization
    # Try multiple orders to find a good starting point
    orders = [
        list(range(1, 25)),       # Large weights first
        list(range(24, 0, -1)),   # Small weights first
        list(range(13, 25)),      # High half
        list(range(1, 13))        # Low half
    ]
    
    for order in orders:
        current_set = set()
        sums = {0}
        for x in order:
            wx = weights[x]
            if wx not in sums:
                current_set.add(x)
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + wx)
                sums = new_sums
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        if time.time() > deadline:
            break

    # 2. Backtracking Search
    # Use large weights first (1..24) for sparser sums and better pruning potential
    elements = list(range(1, 25))
    
    def backtrack(idx, selected, sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: Even if we take all remaining, we can't beat best_size
        if len(selected) + (n - idx) <= best_size:
            return
        
        if idx >= n:
            return
        
        x = elements[idx]
        wx = weights[x]
        
        # Option 1: Include x
        # Valid if wx is not in sums (sums of OTHERS)
        if wx not in sums:
            new_selected = selected + [x]
            new_size = len(new_selected)
            if new_size > best_size:
                best_size = new_size
                best_set = set(new_selected)
                
            # Update sums
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + wx)
            
            backtrack(idx + 1, new_selected, new_sums)
            
            if time.time() > deadline:
                return

        # Option 2: Exclude x
        backtrack(idx + 1, selected, sums)

    backtrack(0, [], {0})
    
    if not best_set:
        best_set = {1}
        
    return list(best_set)
