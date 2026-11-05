def construct():
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    divisor_set = set(divisors)

    def find_sum(k, target_val, exclude=None):
        """
        Find k distinct divisors summing to target_val.
        exclude: set of divisors to exclude (for recursive calls or fixed first element)
        Returns list of divisors or None.
        """
        if exclude is None:
            exclude = set()
        available = [d for d in divisors if d not in exclude]
        if len(available) < k:
            return None
        
        if k == 1:
            if target_val in divisor_set and target_val not in exclude:
                return [target_val]
            return None
        
        if k == 2:
            # Use set for O(N) lookup
            seen = set()
            for d in available:
                if d > target_val:
                    continue
                comp = target_val - d
                if comp in divisor_set and comp not in exclude and comp != d and comp in available:
                    # Ensure comp is in available (distinct and not excluded)
                    # We already checked comp not in exclude and comp != d.
                    # Need to ensure comp is in available list? 
                    # available contains all divisors not in exclude.
                    # If comp is in divisor_set and not in exclude, it is in available.
                    # But we must ensure d and comp are distinct.
                    return [d, comp]
            return None

        if k == 3:
            # Nested loop with set for the third
            n_avail = len(available)
            for i in range(n_avail):
                d1 = available[i]
                if d1 > target_val:
                    continue
                rem1 = target_val - d1
                # Find 2 distinct from available[i+1:]? No, order doesn't matter, but to avoid duplicates in result set, we can just check all pairs.
                # To be efficient, we can just iterate all pairs (j, l) with j < l?
                # Or use the k=2 logic on the remaining set?
                # Simpler: iterate d2, check if rem1-d2 is in set.
                for j in range(i + 1, n_avail):
                    d2 = available[j]
                    if d1 + d2 > target_val:
                        continue
                    rem2 = target_val - d1 - d2
                    if rem2 in divisor_set and rem2 not in exclude and rem2 != d1 and rem2 != d2:
                        return [d1, d2, rem2]
            return None

        if k == 4:
            # Meet in the middle or nested loops
            # Iterate d1, then find 3-sum for rem.
            for i in range(num_divs):
                d1 = divisors[i]
                if d1 > target_val:
                    continue
                rem1 = target_val - d1
                # Find 3 distinct divisors from divisors excluding d1
                # We can call find_sum(3, rem1, {d1})
                # But find_sum(3) uses the global 'available' logic which filters by exclude.
                # This might be slow if called 60 times.
                # Optimization: Precompute pairs?
                # For now, use the recursive call.
                res = find_sum(3, rem1, {d1})
                if res:
                    return [d1] + res
            return None
        
        # General case for k > 4
        # Use DFS with pruning
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
            # Pruning: max sum
            if slots > 0:
                # Sum of largest 'slots' available from start_idx
                # Since divisors are sorted desc, divisors[start_idx : start_idx+slots] are the largest
                max_sum = sum(divisors[start_idx:start_idx+slots])
                if max_sum < remaining:
                    return
            # Pruning: min sum
            # Sum of smallest 'slots' available from start_idx
            # Smallest are at the end of the list
            if slots <= num_divs - start_idx:
                min_sum = sum(divisors[num_divs-slots:num_divs])
                # Wait, we can only use from start_idx onwards.
                # The smallest 'slots' in divisors[start_idx:] are the last 'slots' of that slice.
                available_slice = divisors[start_idx:]
                if len(available_slice) >= slots:
                    min_sum = sum(available_slice[-slots:])
                else:
                    min_sum = sum(available_slice)
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
        
        dfs(0, target_val, [])
        return found

    for k in range(1, 11):
        res = find_sum(k, target)
        if res:
            return res
            
    # Fallback
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
