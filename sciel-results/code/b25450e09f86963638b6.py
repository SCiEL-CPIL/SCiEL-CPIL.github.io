import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    required = set(range(1, 31))
    
    # We will try to find a Sidon set with max element M
    # M must be at least 30 (to have difference 30)
    # Minimum size k: C(k,2) >= 30 -> k >= 9
    
    def is_valid_sidon_step(elements, x, used_diffs):
        """Check if adding x to elements (sorted) keeps Sidon property."""
        # x must be > max(elements)
        # Differences x - e for e in elements must not be in used_diffs
        # and must be distinct from each other (they are, since e are distinct)
        # and must not be in used_diffs
        for e in elements:
            d = x - e
            if d in used_diffs:
                return False
        return True

    def solve_for_M(M, time_limit):
        """
        Find a subset S of {0..M} containing 0, Sidon, covering required.
        Returns S or None.
        """
        elements = [0]
        used_diffs = set()
        
        # We need to select more elements.
        # Max elements is 32, but we probably need around 10-15.
        
        def backtrack(current_elements, current_used_diffs, depth):
            if time.time() > time_limit:
                return None
            
            missing = required - current_used_diffs
            if not missing:
                return list(current_elements)
            
            if len(current_elements) >= 32:
                return None
                
            # Pruning: 
            # If we can't cover the missing diffs with remaining slots, fail.
            # Max new diffs per new element = len(current_elements)
            # Remaining slots = 32 - len(current_elements)
            # This is a very weak bound. 
            # Better: If the smallest missing difference is 1, we need adjacent elements.
            
            max_elem = current_elements[-1]
            
            # Generate candidates x in (max_elem, M]
            # To optimize, we can prioritize x that cover many missing diffs.
            candidates = []
            for x in range(max_elem + 1, M + 1):
                # Check Sidon constraint
                valid = True
                new_diffs = set()
                for e in current_elements:
                    d = x - e
                    if d in current_used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Count how many missing diffs this x covers
                covered_count = len(missing & new_diffs)
                
                # Heuristic: Prefer candidates that cover more missing diffs
                # Also, to minimize M, we are iterating M, so any solution is fine.
                # But inside the search, picking "greedy" coverage might lead to dead ends.
                # However, for small M, the branching factor is small.
                candidates.append((covered_count, x, new_diffs))
            
            # Sort by coverage descending, then by x ascending (to keep elements small? 
            # Actually, larger x creates larger diffs. Smaller x creates smaller diffs.
            # We need small diffs (1..30). So smaller x (closer to max_elem) might be better for covering small missing diffs.
            # But larger x might be needed if max_elem is small.
            # Let's sort by (covered_count desc, x asc)
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for covered_count, x, new_diffs in candidates:
                # Optional: If covered_count is 0, maybe skip? No, we might need it for Sidon structure.
                # But if we have many candidates with 0, we might waste time.
                # Let's just try them.
                
                current_elements.append(x)
                current_used_diffs.update(new_diffs)
                
                result = backtrack(current_elements, current_used_diffs, depth + 1)
                
                current_elements.pop()
                current_used_diffs.difference_update(new_diffs)
                
                if result is not None:
                    return result
            
            return None
        
        return backtrack(elements, used_diffs, 0)

    # Try M from 30 upwards
    # Lower bound on M: 
    # To have diff 30, M >= 30.
    # To have Sidon set covering 1..30, M must be large enough.
    # Known Golomb rulers: 
    # Length 10, 11, etc.
    # Let's start from 30.
    
    for M in range(30, 200):
        if time.time() > deadline - 10:
            break
            
        sol = solve_for_M(M, deadline - 15)
        
        if sol is not None:
            # Verify
            diffs = set()
            for i in range(len(sol)):
                for j in range(i+1, len(sol)):
                    diffs.add(sol[j] - sol[i])
            
            if required.issubset(diffs) and len(diffs) == len(sol)*(len(sol)-1)//2:
                return sol
            # If verification fails (shouldn't happen), continue
    
    # Fallback: Return a known valid Sidon set that covers 1..30, even if M is larger
    # Example: Mian-Chowla sequence starts 0, 1, 3, 7, 15, 31... 
    # 0, 1, 3, 7, 15, 31, 63, 127, 255
    # Diffs: 1, 3, 7, 15, 31... 
    # 1 is covered. 2? 3-1=2. 3? 3-0=3. 4? 7-3=4. 5? 7-2? No 2 not in set.
    # Mian-Chowla is not guaranteed to cover 1..30 densely.
    
    # Let's construct a valid one manually or via a simple greedy that doesn't care about M minimization first.
    # Greedy: Add smallest x that is Sidon and covers a new required diff.
    a = [0]
    used = set()
    missing = set(range(1, 31))
    
    while missing:
        best_x = None
        best_cover = -1
        # Search up to 1000
        for x in range(max(a) + 1, 1000):
            new_diffs = set()
            valid = True
            for v in a:
                d = x - v
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            cover = len(missing & new_diffs)
            if cover > best_cover:
                best_cover = cover
                best_x = x
                if cover == len(missing): # Perfect
                    break
        
        if best_x is None:
            # If no x covers new diffs, just add smallest valid x to keep Sidon
            # This might loop if we can't cover more.
            # But we know solutions exist.
            for x in range(max(a) + 1, 1000):
                new_diffs = set()
                valid = True
                for v in a:
                    d = x - v
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    best_x = x
                    break
        
        if best_x is None:
            break
            
        a.append(best_x)
        a.sort()
        for v in a[:-1]:
            used.add(best_x - v)
        missing -= used
        
    # Verify final result
    diffs = set()
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            diffs.add(a[j] - a[i])
            
    if required.issubset(diffs) and len(diffs) == len(a)*(len(a)-1)//2:
        return a
    
    # Absolute fallback: Return a hardcoded valid set if all else fails
    # A known Golomb ruler of length 10 is {0, 1, 4, 10, 12, 17, 20, 24, 28, 31}? 
    # Let's check diffs for {0, 1, 4, 10, 12, 17, 20, 24, 28, 31}
    # This is a standard optimal Golomb ruler for n=10.
    # Does it cover 1..30?
    # Diffs:
    # 1: 1-0
    # 2: ? 4-2? No. 10-8? No. 
    # 12-10=2. Yes.
    # 3: 4-1=3. Yes.
    # 4: 4-0=4. Yes.
    # 5: 10-5? No. 12-7? No. 17-12=5. Yes.
    # 6: 10-4=6. Yes.
    # 7: 17-10=7. Yes.
    # 8: 20-12=8. Yes.
    # 9: 10-1=9. Yes.
    # 10: 10-0=10. Yes.
    # 11: 12-1=11. Yes.
    # 12: 12-0=12. Yes.
    # 13: 17-4=13. Yes.
    # 14: 20-6? No. 24-10=14. Yes.
    # 15: 17-2? No. 20-5? No. 24-9? No. 28-13? No. 31-16? No.
    # Let's check 15 in {0, 1, 4, 10, 12, 17, 20, 24, 28, 31}
    # 17-4=13, 20-4=16, 24-10=14, 28-12=16, 31-17=14.
    # 15 is missing?
    # 15: 15-0? No 15 not in set.
    # 17-2? No.
    # 20-5? No.
    # 24-9? No.
    # 28-13? No.
    # 31-16? No.
    # So {0, 1, 4, 10, 12, 17, 20, 24, 28, 31} does NOT cover 15.
    
    # Let's try a slightly larger ruler or a different one.
    # {0, 1, 4, 10, 12, 17, 21, 25, 29, 34}?
    # This is getting risky to hardcode.
    
    # The greedy fallback above should work if we search range wide enough.
    # Let's force the greedy to find *some* valid set.
    a = [0]
    used = set()
    missing = set(range(1, 31))
    
    # Force coverage by trying to add elements that cover the *smallest* missing diff first?
    # Or just iterate.
    
    while missing and len(a) < 32:
        found = False
        # Try to find an x that covers at least one missing diff
        for target in sorted(missing):
            # Try to form difference 'target'
            # We need x - y = target => x = y + target
            # y must be in a.
            # x must be > max(a) ? Not necessarily, but we are building sorted.
            # If we add x, it must be consistent with Sidon.
            # Let's just scan x.
            pass
        
        # Simple scan
        best_x = None
        best_gain = 0
        for x in range(max(a) + 1, max(a) + 100):
            new_diffs = set()
            valid = True
            for v in a:
                d = x - v
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            gain = len(missing & new_diffs)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            # Expand search range
            for x in range(max(a) + 100, max(a) + 500):
                new_diffs = set()
                valid = True
                for v in a:
                    d = x - v
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                gain = len(missing & new_diffs)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
        
        if best_x is None:
            # Just add smallest valid x
            for x in range(max(a) + 1, max(a) + 1000):
                new_diffs = set()
                valid = True
                for v in a:
                    d = x - v
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    best_x = x
                    break
        
        if best_x is None:
            break
            
        a.append(best_x)
        a.sort()
        for v in a[:-1]:
            used.add(best_x - v)
        missing -= used
        
    diffs = set()
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            diffs.add(a[j] - a[i])
            
    if required.issubset(diffs) and len(diffs) == len(a)*(len(a)-1)//2:
        return a
    
    # If all fails, return a set that is definitely Sidon and large enough, 
    # even if it doesn't cover 1..30 (which would score 0, but avoids crash).
    # But we must return *something*.
    return [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287]
