import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_set = []
    best_sums = set()
    
    # Candidate numbers from 256 down to 1
    candidates = range(256, 0, -1)
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Calculate new sums if x is added
        # new_sums = {s + x for s in best_sums} | {x}
        new_sums = set()
        for s in best_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Check 1: Distinctness
        # No element in new_sums can be in best_sums
        if new_sums & best_sums:
            continue
            
        # No element in new_sums can be duplicated within new_sums? 
        # Since s are distinct, s+x are distinct. x is distinct from s+x 
        # only if x != s+x => 0 != s, which is true for non-empty sums.
        # So new_sums are internally distinct.
        
        # Check 2: Divisibility
        valid = True
        
        # Check new_sums against best_sums
        for n in new_sums:
            for s in best_sums:
                if n % s == 0 or s % n == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check new_sums against each other
        # We only need to check pairs (n1, n2) where n1 != n2
        # Since we are adding them all at once, we must ensure no n1 | n2
        # Optimization: sort new_sums? 
        # If n1 < n2, then n2 % n1 == 0 is the only possible divisibility.
        # If n1 > n2, then n1 % n2 == 0.
        # So for each pair, check the larger % smaller.
        
        new_sums_list = list(new_sums)
        # Sort to make checking easier? 
        # Actually, just nested loop is fine if size is small.
        # Size of new_sums is 2^k.
        # If k=10, size=1024. Pairs ~ 500k.
        # If k=12, size=4096. Pairs ~ 8M. Might be slow in Python.
        
        # Let's optimize the pair check.
        # For each n in new_sums:
        #   For each m in new_sums where m > n:
        #      if m % n == 0: invalid
        
        # This is still O(N^2).
        # Can we do better?
        # If n divides m, then m is a multiple of n.
        # We can iterate multiples of n?
        # But new_sums is a set.
        
        # Let's try the nested loop but break early.
        # Also, we can check n against best_sums more efficiently?
        # If we store best_sums in a set, checking `n in best_sums` is O(1).
        # Divisibility is the bottleneck.
        
        # Let's assume the greedy set size won't exceed 10-12 before hitting a wall or timeout.
        # 2^12 = 4096. 4096^2 / 2 = 8 million checks. In Python, ~1-2 seconds.
        # If we go up to 15, 32768^2 / 2 = 500 million. Too slow.
        
        # However, the "divisibility" constraint is very strong.
        # Large numbers rarely divide each other.
        # Most checks will fail immediately or pass immediately?
        # `if m % n == 0` is fast.
        
        # Let's refine the loop structure for speed.
        
        # First, check internal divisibility of new_sums.
        # Sort new_sums to ensure n < m.
        new_sums_sorted = sorted(new_sums)
        for i in range(len(new_sums_sorted)):
            n = new_sums_sorted[i]
            # Check against larger elements in new_sums
            for j in range(i + 1, len(new_sums_sorted)):
                m = new_sums_sorted[j]
                if m % n == 0:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            continue
            
        # If we get here, new_sums are internally valid.
        # We already checked against best_sums.
        
        # Accept x
        best_set.append(x)
        best_sums = best_sums.union(new_sums)
        
    # Return as list of ints
    return list(best_set)
