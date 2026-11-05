import time

def construct():
    deadline = time.time() + 115.0
    target_mask = (1 << 30) - 1
    
    # Precompute mask for each diff 1..30
    diff_masks = [0] * 31
    for d in range(1, 31):
        diff_masks[d] = 1 << (d - 1)
        
    def search(D):
        S = [0]
        used_diffs = set()
        covered_mask = 0
        
        def dfs(S, used_diffs, covered_mask):
            if time.time() > deadline:
                return None
            
            if (covered_mask & target_mask) == target_mask:
                return list(S)
            
            # Pruning
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            # Max elements we can add
            max_add = min(D - S[-1], 32 - len(S))
            if max_add <= 0:
                return None
            
            # Max new diffs
            # k new elements add k*len(S) + k*(k-1)/2 diffs
            # We can add at most max_add elements
            potential = max_add * len(S) + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return None
            
            # Generate candidates
            cands = []
            for x in range(S[-1] + 1, D + 1):
                ok = True
                new_mask = 0
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= diff_masks[d]
                if not ok:
                    continue
                
                # Score: number of NEW missing diffs covered
                new_cover = new_mask & missing
                score = bin(new_cover).count('1')
                cands.append((score, x))
            
            # Sort by score desc, then x asc
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to avoid explosion
            # If D is small, this is fine.
            # If D is large, we might want to limit.
            # But we are iterating D from small.
            
            for _, x in cands:
                new_used = set(used_diffs)
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= diff_masks[d]
                
                S.append(x)
                res = dfs(S, new_used, new_mask)
                S.pop()
                
                if res is not None:
                    return res
                
            return None
        
        return dfs(S, used_diffs, covered_mask)

    # Search for minimal D
    # Lower bound: 30
    for D in range(30, 100):
        if time.time() > deadline - 5:
            break
        # Allocate time for this D
        # If we find a solution, return it.
        # We want to try small D first.
        res = search(D)
        if res is not None:
            return res
            
    # Fallback: Return a known valid set or greedy
    # Greedy construction
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
