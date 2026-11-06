"""Improved program with conflict graph and greedy coloring for subset mean avoidance."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    n = 512
    target_size = 48
    
    def _build_dp(arr):
        """Build DP table: dp[k] is bitset of all subset sums of size k."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp
    
    def _is_valid_incremental(existing, x):
        """Check if adding x to existing set keeps it valid. Uses DP for speed."""
        if not existing:
            return True
        m = len(existing)
        total_sum = sum(existing)
        
        # Precompute DP for existing
        dp = _build_dp(existing)
        
        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: no element a in existing is the mean of a subset of existing∪{x} of size >= 2 that includes x
        # Such a subset has x plus t elements from existing, total size t+1 >= 2, so t >= 1
        # Mean = a means: (sum of t elements from existing + x) / (t+1) = a
        # => sum of t elements = (t+1)*a - x
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp[t] >> target) & 1:
                    return False
        return True
    
    def _is_valid_full(selected):
        """Full validation of a complete set."""
        m = len(selected)
        if m < 2:
            return True
        for i, a in enumerate(selected):
            others = [selected[j] for j in range(m) if j != i]
            if not others:
                continue
            dp_others = _build_dp(others)
            for k in range(2, len(others) + 1):
                target = k * a
                if target > sum(others):
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        return True
    
    def _build_conflict_graph():
        """Build conflict graph where edges connect pairs that form 3-element AP with any number in [1,n].
        Edge (a,b) exists if there exists c in [1,n] such that (a+b)/2 = c or (a+c)/2 = b or (b+c)/2 = a.
        This means a+b is even and (a+b)/2 is in [1,n], or a+c is even and (a+c)/2 is in [1,n], etc.
        Actually, simpler: edge (a,b) exists if a+b is even (so (a+b)/2 is integer) AND 1 <= (a+b)/2 <= n.
        But we also need to check if that midpoint is NOT in {a,b} (which it won't be since a!=b).
        So edge exists iff a+b is even and 2 <= a+b <= 2*n."""
        conflicts = [set() for _ in range(n + 1)]
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                s = a + b
                # Check if s is even and s/2 is in [1, n] and s/2 != a and s/2 != b
                if s % 2 == 0:
                    mid = s // 2
                    if 1 <= mid <= n and mid != a and mid != b:
                        conflicts[a].add(b)
                        conflicts[b].add(a)
        return conflicts
    
    def _greedy_conflict_order(conflicts, selected_set):
        """Order remaining numbers by fewest conflicts with selected set, then fewest total conflicts."""
        remaining = [x for x in range(1, n + 1) if x not in selected_set]
        # Score: (conflicts with selected, total conflicts)
        scored = []
        for x in remaining:
            sel_conflicts = len(conflicts[x] & selected_set)
            total_conflicts = len(conflicts[x])
            scored.append((sel_conflicts, total_conflicts, x))
        scored.sort()
        return [x for _, _, x in scored]
    
    def _greedy_from_order(order, selected_start=None):
        """Try to build a valid set by greedily adding elements in the given order."""
        if selected_start is None:
            selected = []
        else:
            selected = list(selected_start)
        for x in order:
            if x in selected:
                continue
            if _is_valid_incremental(selected, x):
                selected.append(x)
                if len(selected) >= target_size:
                    break
        return selected
    
    def _local_search(selected, max_iterations=30):
        """Try to improve the set by swapping elements."""
        selected = list(selected)
        for _ in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 3.0:
                    break
                candidate_remove = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                for x in range(1, n + 1):
                    if x in remaining or x == candidate_remove:
                        continue
                    if _is_valid_incremental(remaining, x):
                        new_selected = remaining + [x]
                        if len(new_selected) >= len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return selected
    
    # Build conflict graph
    conflicts = _build_conflict_graph()
    
    # Strategy 1: Greedy with conflict-aware ordering
    # Start with numbers that have fewest conflicts overall
    initial_order = _greedy_conflict_order(conflicts, set())
    best = _greedy_from_order(initial_order)
    
    if len(best) >= target_size:
        return best
    
    # Strategy 2: Powers of 2 (few subset sum representations)
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    result = _greedy_from_order(powers + list(range(3, n + 1)))
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 3: Greedy from natural order
    order = list(range(1, n + 1))
    result = _greedy_from_order(order)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 4: Greedy from descending order
    order_desc = list(range(n, 0, -1))
    result = _greedy_from_order(order_desc)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 5: Multiple random restarts with conflict-aware reordering
    for trial in range(80):
        if time.time() > deadline - 20.0:
            break
        random.seed(trial)
        # Start with a random subset, then use conflict-aware ordering for the rest
        random_start_size = random.randint(0, 10)
        random_start = random.sample(range(1, n + 1), random_start_size)
        # Build initial set greedily from random start
        selected = []
        for x in random_start:
            if _is_valid_incremental(selected, x):
                selected.append(x)
        
        # Now use conflict-aware ordering for remaining
        remaining_order = _greedy_conflict_order(conflicts, set(selected))
        full_order = selected + [x for x in remaining_order if x not in selected]
        result = _greedy_from_order(full_order, selected_start=selected)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 6: Local search on best found
    if len(best) < target_size:
        improved = _local_search(best, max_iterations=25)
        if len(improved) > len(best):
            best = improved
            if len(best) >= target_size:
                return best
    
    # Strategy 7: Try starting with large numbers
    for trial in range(30):
        if time.time() > deadline - 8.0:
            break
        random.seed(3000 + trial)
        large_nums = list(range(n, n // 2, -1))
        small_nums = list(range(1, n // 2 + 1))
        random.shuffle(large_nums)
        random.shuffle(small_nums)
        order = large_nums[:len(large_nums)//2] + small_nums[:len(small_nums)//2]
        order += large_nums[len(large_nums)//2:] + small_nums[len(small_nums)//2:]
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 8: Modulo-based ordering
    for mod in [3, 4, 5]:
        if time.time() > deadline - 3.0:
            break
        for rem in range(mod):
            order = [x for x in range(1, n + 1) if x % mod == rem]
            result = _greedy_from_order(order)
            if len(result) > len(best):
                best = result
                if len(best) >= target_size:
                    return best
    
    # Strategy 9: Final local search
    if len(best) < target_size and time.time() < deadline - 2.0:
        improved = _local_search(best, max_iterations=40)
        if len(improved) > len(best):
            best = improved
    
    return best
# EVOLVE-BLOCK-END
