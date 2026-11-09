def construct():
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums for max possible sum from index i with k items
    # suffix_sum[i][k] = sum of divisors[i : i+k] (largest k from i onwards)
    # But we need to be careful, we can pick any k from i onwards, not necessarily consecutive.
    # Max sum of k items from divisors[i:] is sum of divisors[i : i+k].
    # Min sum of k items from divisors[i:] is sum of divisors[m-k : m] if m-k >= i, else invalid.
    
    # Let's just use a simple recursive search with pruning.
    # Given m=60, even k=10 is small enough if pruned well.
    
    def search(start, k, remaining, chosen):
        if k == 0:
            return remaining == 0
        if start + k > m:
            return False
        
        # Max possible sum: take largest k from start
        max_possible = sum(divisors[start:start+k])
        if max_possible < remaining:
            return False
            
        # Min possible sum: take smallest k from the available range [start, m)
        # The smallest k elements in divisors[start:m] are the last k elements of divisors[start:m]
        # which are divisors[m-k:m]
        # But we must ensure these are within [start, m).
        # If m-k < start, then we don't have k elements left, which is handled by start+k>m.
        # So min_possible = sum(divisors[m-k:m])
        min_possible = sum(divisors[m-k:m])
        if min_possible > remaining:
            return False
            
        for i in range(start, m - k + 1):
            d = divisors[i]
            if d > remaining:
                continue
            # Prune: if even taking the largest k-1 from i+1 cannot reach remaining-d
            # Max from i+1 with k-1 items: sum(divisors[i+1 : i+1+k-1])
            max_rem = sum(divisors[i+1:i+k])
            if max_rem < remaining - d:
                continue
            # Min from i+1 with k-1 items: sum(divisors[m-(k-1):m])
            # But must be >= i+1.
            # If m-(k-1) < i+1, then min is sum(divisors[i+1:i+1+k-1])? No, min is always the smallest available.
            # The smallest k-1 available from i+1 are the last k-1 of the whole list, provided they are > i.
            # Since we are picking from i+1 to m-1, the smallest k-1 are divisors[m-(k-1):m].
            # We need m-(k-1) >= i+1 => m-k+1 >= i+1 => m-k >= i.
            # This is true because i <= m-k.
            min_rem = sum(divisors[m-(k-1):m])
            if min_rem > remaining - d:
                # Since we are iterating i in increasing order (d decreasing),
                # if the minimum possible sum of the rest is already too high,
                # then for any larger i (smaller d), the remaining sum is larger,
                # so it will also be too high.
                # Wait, if d is smaller, remaining-d is larger.
                # If min_rem > remaining-d, and we increase i, d decreases, remaining-d increases.
                # So min_rem > remaining-d will remain true.
                # Thus we can break.
                break
                
            chosen.append(d)
            if search(i + 1, k - 1, remaining - d, chosen):
                return True
            chosen.pop()
            
        return False

    for k in range(1, m + 1):
        # Quick check for k=1
        if k == 1 and target in divisors:
            return [target]
            
        chosen = []
        if search(0, k, target, chosen):
            return chosen
            
    # Fallback (should not happen)
    return []
