import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)

    def solve(M, k):
        # used_mask: bit d set if diff d used
        # covered_mask: bit d set if target diff d covered
        
        def backtrack(current_set, used_mask, covered_mask):
            if time.time() > deadline:
                return None

            if len(current_set) == k:
                if (covered_mask & target_mask) == target_mask:
                    return current_set[:]
                return None

            last = current_set[-1]
            r = k - len(current_set)
            
            # Pruning: max new diffs
            max_new_diffs = r * len(current_set) + r * (r - 1) // 2
            missing_targets = 30 - bin(covered_mask & target_mask).count("1")
            
            if max_new_diffs < missing_targets:
                return None

            for x in range(last + 1, M + 1):
                valid = True
                new_mask = 0
                for y in current_set:
                    d = x - y
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if not valid:
                    continue

                new_used_mask = used_mask | new_mask
                new_covered_mask = covered_mask | (new_mask & target_mask)

                res = backtrack(current_set + [x], new_used_mask, new_covered_mask)
                if res is not None:
                    return res

            return None

        return backtrack([0], 0, 0)

    # Try k=9 first, as it's the minimum size
    # If k=9 fails to find a solution for M < 50, try k=10?
    # Let's just search M for k=9.
    
    best_sol = None
    best_M = float('inf')

    # We can search M in parallel or sequentially. Sequential is safer for memory.
    # Start from lower bound 36.
    for M in range(36, 80):
        if time.time() > deadline:
            break
        res = solve(M, 9)
        if res is not None:
            best_sol = res
            best_M = M
            break # Found minimal M for k=9

    # If no solution for k=9 in reasonable M, try k=10
    if best_sol is None:
        for M in range(36, 80):
            if time.time() > deadline:
                break
            res = solve(M, 10)
            if res is not None:
                best_sol = res
                best_M = M
                break

    if best_sol is not None:
        return best_sol

    # Fallback: Greedy (very unlikely to be needed)
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return set(y - x for i, x in enumerate(s) for y in s[i+1:])
    
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in range(1, 1000):
            if x in a: continue
            trial = a + [x]
            if len(get_diffs(trial)) == len(trial)*(len(trial)-1)//2:
                a = trial
                found = True
                break
        if not found:
            for x in range(1, 1000):
                for y in range(x+1, 1000):
                    if x in a or y in a: continue
                    trial = a + [x, y]
                    if len(get_diffs(trial)) == len(trial)*(len(trial)-1)//2:
                        a = trial
                        found = True
                        break
                if found: break
            if not found:
                max_a = max(a)
                a.append(max_a + 100)
                a.append(max_a + 200)
    return a
