import time

def construct():
    deadline = time.time() + 115.0
    
    # We want to find the minimum D such that there exists a Sidon set S
    # containing 0, with max(S) <= D, and differences cover 1..30.
    
    target_mask = (1 << 30) - 1  # Bits 0..29 represent diffs 1..30
    
    def search(D, max_time):
        start_time = time.time()
        best_result = [None]
        
        # used_diffs[i] is True if difference i is already used
        # We only care about diffs up to D for Sidon property, 
        # but for coverage we care about 1..30.
        # Actually, if diff > D, it's impossible.
        # So we track all diffs up to D.
        used_diffs = [False] * (D + 1)
        
        def dfs(S, covered_mask, depth):
            if time.time() > max_time:
                return False
            
            if (covered_mask & target_mask) == target_mask:
                best_result[0] = list(S)
                return True
            
            cur_max = S[-1] if S else 0
            
            # Pruning: Can we cover the remaining missing diffs?
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            # Max elements we can add
            # If we add k elements, we add at most k*|S| + k(k-1)/2 diffs.
            # But we are limited by D - cur_max.
            # Simple bound: we need at least num_missing new diffs.
            # Each new element x adds diffs x-s for s in S.
            # Max new diffs per element is len(S).
            # So if num_missing > 0, we need at least 1 more element.
            
            # If we have no more room or not enough potential, fail
            if len(S) >= 32:
                return False
                
            # Determine candidates for next element
            # We must pick x > cur_max.
            # x must be such that no diff x-s is in used_diffs.
            
            # Optimization: Try candidates that cover the most missing diffs first.
            cands = []
            
            # We can limit the search space.
            # If we are close to covering, we might need to be precise.
            # If we are far, we can be greedy.
            
            for x in range(cur_max + 1, D + 1):
                ok = True
                new_cover_mask = 0
                for s in S:
                    d = x - s
                    if d > D:
                        continue
                    if used_diffs[d]:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_cover_mask |= (1 << (d - 1))
                
                if not ok:
                    continue
                
                # Count how many new missing diffs this x covers
                new_covered = new_cover_mask & missing
                cnt = bin(new_covered).count('1')
                
                # Only consider if it covers at least one new diff, 
                # or if we are forced to add (but we are not forced to add 
                # useless elements, so we prefer useful ones).
                # However, sometimes we need to add an element that doesn't cover 
                # a target diff to enable future elements? 
                # No, Sidon sets are "rigid". If an element doesn't help cover 
                # targets, it might block others. 
                # But it might be necessary to bridge gaps?
                # Actually, if x doesn't cover any missing 1..30, it only adds 
                # large diffs. This doesn't help cover 1..30. 
                # So we can ignore x if cnt == 0.
                if cnt == 0:
                    continue
                
                cands.append((cnt, x, new_cover_mask))
            
            # Sort by coverage (desc), then by value (asc) to keep diameter small
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit branching factor to keep search fast
            # If we have too many candidates, we might miss the optimal path.
            # But for D~40, the number of valid Sidon extensions is small.
            for cnt, x, new_mask in cands:
                # Record changes
                changes = []
                for s in S:
                    d = x - s
                    if d <= D:
                        if used_diffs[d]:
                            # Should not happen due to check above, but safe
                            break
                        used_diffs[d] = True
                        changes.append(d)
                
                S.append(x)
                
                if dfs(S, covered_mask | new_mask, depth + 1):
                    return True
                
                S.pop()
                
                for d in changes:
                    used_diffs[d] = False
                
                if time.time() > max_time:
                    return False
            
            return False
        
        # Start DFS with S = [0]
        dfs([0], 0, 0)
        return best_result[0]

    # Search for minimal D
    # Lower bound for k=9 is 36.
    # Let's start from 36.
    for D in range(36, 100):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        # Allocate time proportional to difficulty? 
        # Smaller D are harder to find if they don't exist.
        # But we expect a solution around 42.
        # Give 5 seconds per D.
        time_per_D = min(5.0, time_left * 0.9)
        if time_per_D < 0.1:
            break
            
        res = search(D, time.time() + time_per_D)
        if res is not None:
            return res
            
    # Fallback: If no solution found in time, return a valid Sidon set 
    # that covers 1..30, even if not minimal.
    # We can use a greedy approach to build a valid set.
    S = [0]
    used = set()
    target = set(range(1, 31))
    
    # Greedy construction
    # Try to add elements that cover missing diffs.
    # We allow up to 10^6.
    while target:
        best_x = None
        best_cover = -1
        # Look for x in a reasonable range
        # If we have k elements, next x can be large.
        # But to keep it small, we look near current max.
        # However, if we get stuck, we need to jump.
        
        # Simple heuristic: try x from max(S)+1 to max(S)+1000
        # If no good x found, expand range.
        found = False
        search_limit = 100
        while not found and search_limit < 10000:
            for x in range(max(S) + 1, max(S) + search_limit + 1):
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
            if best_x is not None:
                found = True
            else:
                search_limit *= 2
        
        if best_x is None:
            # If we really can't find one, just pick a large random number?
            # This shouldn't happen for 1..30.
            break
            
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
        
    return S

# Test locally if needed
# if __name__ == "__main__":
#     res = construct()
#     print(res)
#     print(len(res), max(res))
#     diffs = set()
#     for i in range(len(res)):
#         for j in range(i):
#             d = res[i] - res[j]
#             if d in diffs:
#                 print("Duplicate diff:", d)
#             diffs.add(d)
#     missing = set(range(1, 31)) - diffs
#     print("Missing:", missing)
