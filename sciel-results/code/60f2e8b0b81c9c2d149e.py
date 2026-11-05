def construct():
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    for k in range(1, 257):
        found = None
        def dfs(start_idx, remaining, current):
            nonlocal found
            if found is not None:
                return
            if len(current) == k:
                if remaining == 0:
                    found = current[:]
                return
            if remaining < 0:
                return
            slots = k - len(current)
            if slots > num_divs - start_idx:
                return
            # Max possible sum
            if slots > 0:
                max_sum = sum(divisors[start_idx:start_idx+slots])
                if max_sum < remaining:
                    return
            # Min possible sum: smallest 'slots' divisors from start_idx to end
            # Since divisors are sorted descending, smallest are at the end
            min_sum = sum(divisors[num_divs-slots:num_divs]) if slots <= num_divs else 0
            # But we can only use divisors from start_idx onwards, so:
            available = divisors[start_idx:]
            if len(available) >= slots:
                min_sum = sum(available[-slots:])
            else:
                min_sum = sum(available)
            if min_sum > remaining:
                return
            
            for i in range(start_idx, num_divs):
                d = divisors[i]
                if d > remaining:
                    continue
                current.append(d)
                dfs(i+1, remaining-d, current)
                current.pop()
                if found is not None:
                    return
        
        dfs(0, target, [])
        if found is not None:
            return found
    
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
