import time
import math

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    divisor_set = set(divisors)
    n_divs = len(divisors)
    
    # Precompute suffix sums for pruning (for max possible sum checks)
    # Also, for efficiency in the backtracking, we sort divisors in descending order
    # to try larger divisors first, which might lead to solutions with fewer terms?
    # No, we are minimizing the number of terms.
    # If we search for k=1, then k=2, etc., the order of divisors in the set doesn't change the existence.
    # However, for a fixed k, searching larger divisors first might find a solution faster?
    # Or smaller?
    # If we want to hit target 4979, and we pick k=5, we need an average of ~1000.
    # Large divisors are ~2520, 1680, 1260...
    # Let's sort descending for backtracking to prune "sum > target" faster?
    # Actually, standard subset sum DFS usually benefits from sorting descending to prune upper bounds quickly.
    
    divisors_desc = sorted(divisors, reverse=True)
    
    # k=1
    if target in divisor_set:
        return [target]
    
    # k=2
    # Two sum
    # Using set for O(N)
    for d in divisors:
        comp = target - d
        if comp > d and comp in divisor_set:
            return [d, comp]
    
    # k=3
    # Three sum
    # O(N^2)
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            c = target - a - b
            if c > b and c in divisor_set:
                return [a, b, c]
    
    # k=4
    # Meet in the middle: Pairs
    # Create all pair sums (i, j) with i < j
    # Map sum -> list of (i, j)
    # Then for each pair (s1, i1, j1), check if (target - s1) is in map, and if disjoint.
    
    # Optimization: We can just generate pairs and store in a dict.
    # Since N=60, pairs ~ 1770. This is small.
    
    pair_map = {}
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            continue
        for j in range(i + 1, n_divs):
            b = divisors[j]
            s = a + b
            if s >= target:
                break
            if s not in pair_map:
                pair_map[s] = []
            pair_map[s].append((i, j))
    
    for s1, pairs1 in pair_map.items():
        s2 = target - s1
        if s2 not in pair_map:
            continue
        # Avoid double counting by ensuring s1 <= s2 or specific index check?
        # We just need to find ONE valid set.
        # Iterate through pairs in s1
        for (i1, j1) in pairs1:
            for (i2, j2) in pair_map[s2]:
                # Check disjointness
                if len({i1, j1, i2, j2}) == 4:
                    return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
    
    # k >= 5
    # Backtracking with pruning
    # We sort divisors descending to maximize the sum quickly?
    # Or ascending?
    # If we want to find a solution, the order doesn't matter for correctness, only speed.
    # Pruning:
    # 1. remaining_sum < 0
    # 2. remaining_sum > max_possible_sum of remaining elements
    # 3. Not enough elements left to reach remaining_k
    
    # To maximize pruning effectiveness, sorting in descending order is often better for subset sum
    # because you exceed the target sum faster, pruning the "too big" branch early.
    
    # However, we need to be careful with the "min possible sum" check if we want to ensure we don't skip valid solutions.
    # Standard DFS:
    # def dfs(start, k, rem, chosen):
    #   if k == 0: return rem == 0
    #   if not enough elements: return False
    #   for i in range(start, N):
    #      d = divisors[i]
    #      if d > rem: break (if sorted asc)
    #      ...
    
    # Let's use divisors_desc (sorted descending).
    # If we pick d, new_rem = rem - d.
    # If new_rem < 0: prune (since all subsequent d are smaller, they would also be positive, but we need to sum to 0? No, we need sum to target. So d must be <= rem).
    # Wait, if sorted descending, we can't simply "break" when d > rem, because smaller d later might be <= rem.
    # So we must skip d > rem.
    
    # Let's stick to divisors (ascending) for the inner loop logic simplicity, or just iterate all and check.
    # N=60 is small.
    
    # Precompute suffix sums for max_possible?
    # If we are at index `start` in `divisors_desc`, the max sum of `k` items is sum(divisors_desc[start:start+k]).
    
    # Let's use divisors_desc.
    # Max sum of k items starting from index `start` is sum(divisors_desc[start : start+k]).
    # If rem > max_sum: prune.
    
    # Also, min sum of k items starting from index `start`?
    # Since sorted descending, the smallest k items from `start` onwards are the last k items in the available range?
    # No, the smallest are at the end of the list.
    # This makes min-sum pruning harder with descending sort.
    
    # Let's use `divisors` (ascending) for DFS.
    # Max sum of k items from index `start`: sum(divisors[N-k : N])? No, we can only pick from `start` onwards.
    # So max sum is sum(divisors[N-k : N]) if start <= N-k, else sum(divisors[start : N]) if N-start < k?
    # Actually, the largest k elements in divisors[start:] are the last k elements of divisors.
    # So max_possible = sum(divisors[N-k:N]).
    # This is constant for a given k!
    # Wait, no. We are picking from `start` to `N-1`.
    # The largest k elements in that range are indeed the last k elements of the entire array, provided `start` is small enough to include them.
    # If `start` is large, we might not be able to pick the very largest ones.
    # Specifically, the available elements are divisors[start:N].
    # The largest k elements in this subarray are divisors[N-k:N] IF start <= N-k.
    # If start > N-k, then we must pick from divisors[start:N], and the max sum is sum(divisors[start:N]) (which has length < k, so impossible anyway).
    
    # Min sum of k elements from index `start`:
    # The smallest k elements in divisors[start:N] are divisors[start:start+k].
    
    # So:
    # min_possible = sum(divisors[start:start+k])
    # max_possible = sum(divisors[N-k:N]) if start <= N-k else sum(divisors[start:N]) (but if N-start < k, it's invalid)
    
    # Let's implement this.
    
    # Precompute prefix sums for divisors to get sum(start:start+k) quickly?
    # Or just use suffix sums for max?
    
    prefix_sums = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix_sums[i+1] = prefix_sums[i] + divisors[i]
        
    def get_sum(l, r):
        return prefix_sums[r] - prefix_sums[l]
        
    # max_possible for k items from start:
    # The k largest items in divisors[start:] are the k largest in the whole list, UNLESS start > N-k.
    # If start <= N-k, max_sum = get_sum(N-k, N)
    # If start > N-k, we can't pick k items, so return -infinity or handle invalid case.
    
    # Let's refine the DFS.
    
    def dfs(start, k, rem, chosen):
        if k == 0:
            return rem == 0
        
        if n_divs - start < k:
            return False
            
        # Min possible sum
        min_sum = get_sum(start, start + k)
        if rem < min_sum:
            return False
            
        # Max possible sum
        # The largest k elements available from index `start` onwards.
        # If start <= n_divs - k, the largest k elements are the last k of the array.
        if start <= n_divs - k:
            max_sum = get_sum(n_divs - k, n_divs)
        else:
            # We must pick from start to n_divs-1, but there are fewer than k elements?
            # No, we checked n_divs - start < k earlier.
            # So this case implies start > n_divs - k, which means n_divs - start < k.
            # So this branch is unreachable if the check above is correct.
            # Wait. If start = 58, N=60, k=3. N-start = 2 < 3. Checked.
            # If start = 55, N=60, k=3. N-start = 3. start (55) > N-k (57)? No. 55 <= 57.
            # So we use max_sum = get_sum(57, 60).
            # But we can only pick from 55, 56, 57, 58, 59.
            # The largest 3 are 57, 58, 59. Which is get_sum(57, 60). Correct.
            # What if start = 58, k=2? N-k=58. start=58 <= 58.
            # Max sum = get_sum(58, 60). Elements 58, 59. Correct.
            
            pass # Should not happen
            
        if rem > max_sum:
            return False
            
        for i in range(start, n_divs):
            d = divisors[i]
            if d > rem:
                # Since sorted ascending, if d > rem, all subsequent j > i will also have divisors[j] > rem.
                # So we can break.
                break
                
            chosen.append(d)
            if dfs(i + 1, k - 1, rem - d, chosen):
                return True
            chosen.pop()
            
        return False
    
    for k in range(5, n_divs + 1):
        chosen = []
        if dfs(0, k, target, chosen):
            return chosen
    
    # Fallback (should not reach)
    return []
