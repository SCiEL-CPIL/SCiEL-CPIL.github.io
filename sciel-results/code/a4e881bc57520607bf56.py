import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target: Sidon set containing 0, diffs include 1..30, minimize max(A)
    # Lower bound for max(A) is 30 (since diff 30 must exist).
    
    # Try n=6 first. If found, max(A) must be 30.
    # Check if a perfect difference set of size 6 exists.
    
    def is_sidon(set_a):
        diffs = set()
        for i in range(len(set_a)):
            for j in range(i+1, len(set_a)):
                d = set_a[j] - set_a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def has_diffs_1_to_30(set_a):
        diffs = set()
        for i in range(len(set_a)):
            for j in range(i+1, len(set_a)):
                diffs.add(set_a[j] - set_a[i])
        return all(d in diffs for d in range(1, 31))

    # Search for n=6, max element 30
    # Elements must be in 0..30. 0 is included.
    # We need to choose 5 other elements from 1..30.
    # Backtracking
    
    def solve_n6():
        if time.time() > deadline:
            return None
        a = [0]
        # Pruning: we need to form diffs 1..30.
        # With 5 more elements, we add 5*6=30 diffs? No, 5 new elements create 5*1 + 5*4 + ... 
        # Total pairs for n=6 is 15 unordered, 30 ordered.
        
        def backtrack(current_set, start_idx):
            if time.time() > deadline:
                return None
            if len(current_set) == 6:
                if has_diffs_1_to_30(current_set):
                    return current_set
                return None
            
            # Heuristic: add elements that help cover missing diffs
            # Missing diffs in current set
            curr_diffs = set()
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    curr_diffs.add(current_set[j] - current_set[i])
            
            missing = [d for d in range(1, 31) if d not in curr_diffs]
            
            # Try adding x from start_idx to 30
            # To minimize max, we should try smaller x first?
            # But we are fixing max=30 for n=6?
            # Actually, if we find a set with max < 30, that's impossible because diff 30 is needed.
            # So max must be 30.
            # So 30 must be in the set?
            # If 30 is not in the set, max < 30, so diff 30 impossible.
            # So 30 MUST be in the set.
            
            # So we can fix 30 as the last element?
            # Or just ensure 30 is added.
            
            # Let's force 30 to be in the set.
            # If len(current_set) == 5, we must add 30.
            
            if len(current_set) == 5:
                if 30 in current_set:
                    return None # 30 already added, need one more? No, size 6.
                # Add 30
                trial = current_set + [30]
                if is_sidon(trial):
                    if has_diffs_1_to_30(trial):
                        return trial
                return None
            
            # Try adding x
            for x in range(start_idx, 31):
                if x in current_set:
                    continue
                # Pruning: if x < max(current_set) and we haven't added 30 yet, 
                # we must still be able to add 30 later.
                
                trial = current_set + [x]
                if not is_sidon(trial):
                    continue
                
                # Check if we can still reach 30
                # If x > 30, break (not possible)
                
                res = backtrack(trial, x + 1)
                if res:
                    return res
            return None

        return backtrack([0], 1)

    # Try n=6
    result = solve_n6()
    if result:
        return result

    # If n=6 fails, try n=7
    # Min max element for n=7 is 42 (since 42 distinct diffs needed, max diff = max element)
    # Actually, max element M must be at least 42.
    # We search for M = 42, 43, ...
    
    def solve_n7(M):
        if time.time() > deadline:
            return None
        a = [0]
        
        def backtrack(current_set, start_idx):
            if time.time() > deadline:
                return None
            if len(current_set) == 7:
                if has_diffs_1_to_30(current_set):
                    if max(current_set) <= M:
                        return current_set
                return None
            
            curr_diffs = set()
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    curr_diffs.add(current_set[j] - current_set[i])
            
            missing = [d for d in range(1, 31) if d not in curr_diffs]
            
            # If we can't cover missing diffs with remaining slots, prune?
            # Remaining elements: 7 - len(current_set)
            # Max new diffs: (7-k)*k + (7-k)(6-k)/2?
            
            for x in range(start_idx, M + 1):
                if x in current_set:
                    continue
                
                trial = current_set + [x]
                if not is_sidon(trial):
                    continue
                
                # Pruning: if max(trial) > M, break
                if max(trial) > M:
                    break
                
                res = backtrack(trial, x + 1)
                if res:
                    return res
            return None

        return backtrack([0], 1)

    # Try n=7 with M=42
    result = solve_n7(42)
    if result:
        return result

    # Try n=7 with M=43, etc., up to deadline
    for M in range(43, 1000):
        if time.time() > deadline:
            break
        result = solve_n7(M)
        if result:
            return result

    # Fallback: return a valid set even if not minimal, to avoid error
    # Construct a large Sidon set
    a = [0]
    for d in range(1, 31):
        if d in [y-x for i,x in enumerate(a) for y in a[i+1:]]:
            continue
        found = False
        for x in range(1, 10000):
            if x in a: continue
            trial = a + [x]
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            # Add large element
            y = 2*max(a) + 100
            while not is_sidon(a + [y]):
                y += 1
            a.append(y)
    return a

# EVOLVE-BLOCK-END
