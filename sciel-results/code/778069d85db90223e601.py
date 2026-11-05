import time

def construct():
    deadline = time.time() + 75
    target = 200
    universe = list(range(1, 101))
    
    # Pre-compute weights w_i: number of subsets of [1,100]\{i} summing to 200-i
    weights = [0] * 101
    for i in range(1, 101):
        if target - i < 0:
            weights[i] = 0
            continue
        
        max_sum_needed = target - i
        dp = [0] * (max_sum_needed + 1)
        dp[0] = 1
        for j in range(1, 101):
            if j == i:
                continue
            if j > max_sum_needed:
                continue
            for k in range(max_sum_needed, j - 1, -1):
                dp[k] += dp[k - j]
        weights[i] = dp[target - i]
    
    def find_bad_subset(S_set):
        """
        Returns a list of elements in S_set that sum to target, or None if no such subset exists.
        """
        mask = (1 << (target + 1)) - 1
        reachable = 1
        choice = {}
        
        for x in S_set:
            if x > target:
                continue
            new_sums = (reachable << x) & mask
            to_add = new_sums & ~reachable
            if to_add:
                for k in range(target + 1):
                    if (to_add >> k) & 1:
                        choice[k] = (k - x, x)
            
            reachable |= new_sums
            if (reachable >> target) & 1:
                subset = []
                current_sum = target
                while current_sum > 0:
                    if current_sum not in choice:
                        break
                    prev_sum, elem = choice[current_sum]
                    subset.append(elem)
                    current_sum = prev_sum
                return subset
        return None

    # Strategy 1: Greedy deletion
    S1 = set(universe)
    while time.time() < deadline:
        bad_set = find_bad_subset(S1)
        if bad_set is None:
            break
        candidates = sorted(bad_set, key=lambda x: (-weights[x], -x))
        S1.remove(candidates[0])
    
    # Strategy 2: Greedy construction - add elements in order of increasing weight
    S2 = set()
    sorted_by_weight = sorted(universe, key=lambda x: (weights[x], x))
    for x in sorted_by_weight:
        if time.time() > deadline:
            break
        test_set = S2 | {x}
        if find_bad_subset(test_set) is None:
            S2.add(x)
    
    # Pick the larger result
    if len(S1) >= len(S2):
        return list(S1)
    else:
        return list(S2)
