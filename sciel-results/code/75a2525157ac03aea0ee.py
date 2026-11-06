def construct():
    max_target = 128
    # Generate smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)
    
    smooth_set = set(smooth)
    
    # Helper to check antichain
    def is_anticain(nums):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                a, b = nums[i], nums[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True

    results = []
    for target in range(1, max_target + 1):
        # 1. Check 1 term
        if target in smooth_set:
            results.append([target])
            continue
        
        # 2. Check 2 terms
        found = False
        for i in range(len(smooth)):
            a = smooth[i]
            if a >= target:
                break
            b = target - a
            if b in smooth_set and b != a:
                if is_anticain([a, b]):
                    results.append([a, b])
                    found = True
                    break
        if found:
            continue
            
        # 3. Check 3 terms
        # Optimization: iterate pairs and check if remainder is in smooth set and forms antichain
        found = False
        for i in range(len(smooth)):
            a = smooth[i]
            if a >= target:
                break
            for j in range(i+1, len(smooth)):
                b = smooth[j]
                if a + b >= target:
                    break
                c = target - (a + b)
                if c in smooth_set and c > b: # Ensure distinct and sorted order for uniqueness
                    # Check distinctness: a, b, c must be distinct
                    # Since smooth is sorted and i<j, a<b. c>b ensures c!=a,b.
                    if is_anticain([a, b, c]):
                        results.append([a, b, c])
                        found = True
                        break
            if found:
                break
        if found:
            continue

        # 4. Fallback: 4 terms (unlikely needed for 1..128)
        # If we get here, we need a more general search
        # But for 1..128, 3 terms should suffice.
        # Let's add a generic backtracking for safety if 3 fails.
        
        # Generic search for k terms
        def search_k(k, start_idx, remaining, current_combo):
            if k == 0:
                return [] if remaining == 0 else None
            if remaining < 0:
                return None
            
            # Pruning: min possible sum with k terms from start_idx
            # max possible sum with k terms from end
            
            for i in range(start_idx, len(smooth)):
                val = smooth[i]
                if val > remaining:
                    break
                
                # Check if adding val to current_combo maintains antichain
                valid = True
                for v in current_combo:
                    if v % val == 0 or val % v == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                res = search_k(k-1, i+1, remaining - val, current_combo + [val])
                if res is not None:
                    return [val] + res
            return None

        # Try k=4, 5...
        sol = None
        for k in range(4, 8):
            sol = search_k(k, 0, target, [])
            if sol:
                break
        
        if sol:
            results.append(sol)
        else:
            raise ValueError(f"No representation found for {target}")

    return results
