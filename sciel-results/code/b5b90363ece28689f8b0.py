import time
import sys

N = 256

def construct():
    start_time = time.time()
    time_limit = 25.0  # Leave 5s for overhead/return
    
    best_size = N + 1
    best_set = []
    
    # Precompute pairs? No, dynamic.
    # We use a bitmask for forbidden values.
    # forbidden[i] = 1 if i cannot be added.
    # Initially, all 0.
    # When we add x, we mark x as forbidden (used).
    # Also, for any existing y in set, if x+y is a sum, does that block anything?
    # The condition for a set S to be Sidon is that all sums a+b (a<=b) are distinct.
    # If we add x to S, we need to ensure:
    # 1. x+x is not in Sums(S)
    # 2. x+y is not in Sums(S) for any y in S
    
    # What does "forbidden" mean in the context of "maximally blocked"?
    # A value z is "blocked" if adding z to the current set S would violate the Sidon property.
    # z is blocked if:
    # - 2z is in Sums(S)
    # - z+y is in Sums(S) for some y in S
    
    # So, `forbidden_mask` can represent the set of values that are *currently* blocked.
    # Initially, no values are blocked.
    # When we add x, we update the sums. Then we recompute which values are blocked.
    
    # To speed up "recompute blocked", we can maintain the set of sums.
    
    def is_blocked(z, sums, selected):
        if 2 * z in sums:
            return True
        for y in selected:
            if z + y in sums:
                return True
        return False

    def get_valid_candidates(forbidden_mask, sums, selected):
        valid = []
        for z in range(1, N + 1):
            if forbidden_mask & (1 << z):
                continue
            if not is_blocked(z, sums, selected):
                valid.append(z)
        return valid

    def compute_destruction_score(x, valid_candidates, sums, selected):
        # If we add x, what new sums are created?
        # new_sums = {x+x} U {x+y for y in selected}
        # A candidate y in valid_candidates becomes blocked if:
        # - 2y is in new_sums
        # - y+z is in new_sums for some z in selected U {x}
        # Wait, the condition for y to be blocked by the *new* set S' = S U {x} is:
        # - 2y in Sums(S')
        # - y+w in Sums(S') for some w in S'
        
        # Sums(S') = Sums(S) U NewSums
        # y is blocked if:
        # (2y in Sums(S) OR 2y in NewSums) OR (exists w in S' such that y+w in Sums(S') )
        
        # Since y was valid before, 2y is NOT in Sums(S), and y+w is NOT in Sums(S) for any w in S.
        # So y becomes blocked if:
        # 1. 2y in NewSums
        # 2. exists w in S' such that y+w in NewSums
        
        # NewSums = {2x} U {x+w for w in S}
        
        # So we check if any element of NewSums is equal to 2y or y+w (for w in S').
        
        new_sums_set = set()
        new_sums_set.add(2 * x)
        for w in selected:
            new_sums_set.add(x + w)
            
        blocked_count = 0
        for y in valid_candidates:
            if y == x:
                continue
            # Check if y is blocked by new_sums_set
            if 2 * y in new_sums_set:
                blocked_count += 1
            else:
                # Check if y+w in new_sums_set for any w in S U {x}
                blocked = False
                for w in selected:
                    if y + w in new_sums_set:
                        blocked = True
                        break
                if not blocked and (y + x) in new_sums_set:
                    # Wait, w can be x too. S' = S U {x}.
                    # So we check w in S and w=x.
                    # The loop above checks w in S.
                    # We need to check if y+x is in new_sums_set.
                    # Note: y+x is in new_sums_set if x+y is in {2x} U {x+w for w in S}.
                    # x+y = 2x => y=x (excluded)
                    # x+y = x+w => y=w (w in S). But y is a candidate, so y not in S.
                    # So y+x is NEVER in new_sums_set?
                    # Let's re-evaluate.
                    # NewSums are sums involving x and elements of S.
                    # Sums are: x+x, x+s1, x+s2, ...
                    # Does y+x appear in NewSums?
                    # y+x = x+x => y=x (no)
                    # y+x = x+s => y=s (no, y not in S)
                    # So y+x is NOT in NewSums.
                    
                    # So the condition "exists w in S' such that y+w in NewSums"
                    # reduces to "exists w in S such that y+w in NewSums".
                    # Because w=x case: y+x not in NewSums.
                    
                    # So the loop `for w in selected` is sufficient.
                    if blocked:
                        blocked_count += 1
        
        return blocked_count

    def dfs(selected, sums, forbidden_mask, depth):
        nonlocal best_size, best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        # Pruning
        if len(selected) >= best_size:
            return
            
        # Check if maximal
        valid_candidates = get_valid_candidates(forbidden_mask, sums, selected)
        
        if not valid_candidates:
            # Maximal set found
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
        
        # Heuristic: Sort by destruction score (descending)
        # To save time, we might not want to compute score for ALL if there are many.
        # But N=256, valid candidates usually drop fast.
        
        scored_candidates = []
        for x in valid_candidates:
            score = compute_destruction_score(x, valid_candidates, sums, selected)
            scored_candidates.append((score, x))
            
        # Sort by score desc, then by value asc (tie breaker)
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        
        # Limit branching factor to avoid explosion?
        # Let's try top 10 candidates first. If that doesn't find a small set, we might need more.
        # But for now, let's try all, but order them.
        # If it's too slow, we can limit to top K.
        
        for score, x in scored_candidates:
            # Optional: Prune if score is 0 and we have tried some? No, score 0 means it doesn't block anyone,
            # which is bad for minimizing size (we want to block many).
            # But we must try it to ensure maximality if others fail?
            # Actually, if we pick an element that blocks 0, the set size might grow larger.
            # Since we want to minimize size, branches with low destruction are less promising.
            
            # Add x
            new_selected = selected + [x]
            new_sums = sums.copy()
            new_sums.add(2 * x)
            for s in selected:
                new_sums.add(x + s)
                
            new_forbidden = forbidden_mask | (1 << x)
            
            dfs(new_selected, new_sums, new_forbidden, depth + 1)
            
            # Backtrack
            # (No explicit removal needed as we pass copies)

    # Initial call
    dfs([], set(), 0, 0)
    
    return best_set
