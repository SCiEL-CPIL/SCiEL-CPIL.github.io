import time

def construct():
    deadline = time.time() + 115.0
    
    target_mask = (1 << 30) - 1  # bits 0..29 represent diffs 1..30
    
    def mask_for_diff(d):
        if 1 <= d <= 30:
            return 1 << (d - 1)
        return 0
    
    def search(D, max_time):
        """Find a Sidon set in [0, D] containing 0 with all diffs 1..30."""
        start_time = time.time()
        
        # We'll build the set incrementally. S[0] = 0.
        # State: current set S, set of used differences, mask of covered target diffs
        # Use iterative deepening on number of elements? Or just DFS.
        
        best_result = [None]
        
        def dfs(S, used_diffs_mask, covered_mask, depth):
            if time.time() > max_time:
                return
            
            if (covered_mask & target_mask) == target_mask:
                if best_result[0] is None:
                    best_result[0] = list(S)
                return
            
            cur_max = S[-1] if S else 0
            remaining = D - cur_max
            if remaining <= 0:
                return
            
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            # Pruning: max new differences we can add
            # If we add k more elements, max new diffs = k * len(S) + k*(k-1)//2
            # But we can add at most 'remaining' new elements (one per position)
            # Actually we can add at most min(remaining, 32 - len(S)) elements
            max_add = min(remaining, 32 - len(S))
            potential = max_add * len(S) + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return
            
            # Generate candidates: x in [cur_max+1, D]
            # For each, check Sidon constraint and compute coverage
            # Sort by coverage descending, then x ascending
            
            cands = []
            for x in range(cur_max + 1, D + 1):
                # Check Sidon: x - s not in used_diffs for any s in S
                ok = True
                new_mask = 0
                for s in S:
                    d = x - s
                    if d in used_diffs_mask:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= mask_for_diff(d)
                if not ok:
                    continue
                new_cover = new_mask & missing
                cnt = bin(new_cover).count('1')
                cands.append((cnt, x))
            
            # Sort: most coverage first, then smallest x
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates for speed
            if len(cands) > 40:
                cands = cands[:40]
            
            for _, x in cands:
                new_used = set(used_diffs_mask)
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= mask_for_diff(d)
                
                S.append(x)
                dfs(S, new_used, new_mask, depth + 1)
                S.pop()
                
                if best_result[0] is not None:
                    return
                if time.time() > max_time:
                    return
        
        dfs([0], set(), 0, 0)
        return best_result[0]
    
    # Search for minimum D
    # Lower bound: D >= 30
    # Try D from 30 upward
    for D in range(30, 100):
        if time.time() > deadline:
            break
        # Allocate remaining time proportionally
        time_left = deadline - time.time()
        time_per_D = min(10.0, time_left * 0.8)  # leave 20% for later
        res = search(D, time.time() + time_per_D)
        if res is not None:
            return res
    
    # Fallback: greedy construction
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline:
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 200):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok and cover > best_cover:
                best_cover = cover
                best_x = x
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
    
    return S
