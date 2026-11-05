import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Run multiple random restarts
    # We can do a few deterministic ones first to establish a baseline
    # 1. Ascending (seed)
    # 2. Descending
    # 3. Random permutations
    
    def run_greedy(order):
        chosen = []
        counts = {}
        for x in order:
            # Calculate new sums
            # Optimization: if x is large, sums are large.
            # We just need to check if any sum exceeds count 2.
            
            # To speed up, we can check sums in a specific way?
            # The current O(N) check is fine for N=1000.
            
            # Check validity
            valid = True
            new_sums = []
            for y in chosen:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            # We need to check if adding these increments any count to > 2.
            # Instead of copying the dict, we can just check the current counts.
            # If count[s] is already 2, and s is in new_sums, it becomes 3 -> invalid.
            # If count[s] is 1, it becomes 2 -> OK.
            # If count[s] is 0, it becomes 1 -> OK.
            
            # However, multiple new_sums can be the same.
            # e.g. x + y1 == x + y2 is impossible since y1 != y2.
            # But x + y1 == 2*x is possible if y1 == x (impossible as y1 < x in ascending? 
            # Wait, in random order, y can be anything).
            # If y = x, then x+y = 2x. But x is not in chosen yet.
            # So x+y for y in chosen are distinct from 2x?
            # x + y = 2x => y = x. Since x is not in chosen, y != x.
            # So x+y values are distinct from 2x.
            # Are x+y values distinct from each other?
            # x + y1 = x + y2 => y1 = y2. Since y are distinct in chosen, sums are distinct.
            # So all elements in new_sums are distinct.
            
            # So we just need to check if any s in new_sums has count[s] == 2.
            
            for s in new_sums:
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        return chosen

    # 1. Ascending
    order = list(range(1, 1001))
    s = run_greedy(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # 2. Descending
    order = list(range(1000, 0, -1))
    s = run_greedy(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # 3. Random restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = run_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return best_set
