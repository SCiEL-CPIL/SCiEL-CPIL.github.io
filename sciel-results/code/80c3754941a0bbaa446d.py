import time
import random

def construct():
    deadline = time.time() + 85.0
    n = 512
    target_size = 48

    # Bitset DP: dp[k] is a Python int where bit s is set iff s is a subset-sum of size k
    def build_dp(arr):
        m = len(arr)
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def is_valid(arr):
        m = len(arr)
        if m < 3:
            return True
        for i in range(m):
            a = arr[i]
            others = arr[:i] + arr[i + 1:]
            dp = build_dp(others)
            total = sum(others)
            for k in range(2, m):
                target = k * a
                if target > total:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def can_add(arr, x, dp=None, total=None):
        m = len(arr)
        if dp is None:
            dp = build_dp(arr)
        if total is None:
            total = sum(arr)
        # Check 1: x is not mean of any subset of arr of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > total:
                break
            if (dp[k] >> target) & 1:
                return False
        # Check 2: no a in arr is mean of subset of (arr \ {a}) + {x}
        # sum of t elements from (arr \ {a}) plus x = (t+1)*a
        # sum of t elements from (arr \ {a}) = (t+1)*a - x
        # We need DP of (arr \ {a}), but that's expensive.
        # Alternative: for each a, check if (t+1)*a - x is a subset-sum of size t from arr \ {a}
        # (t+1)*a - x = (t+1)*a - x
        # Note: (t+1)*a - x must be sum of t elements from arr \ {a}
        # Equivalent: x + (sum of t elements from arr\{a}) = (t+1)*a
        # i.e., (t+1)*a is a subset-sum of size (t+1) from arr + {x} that includes x
        # = (t+1)*a - x is a subset-sum of size t from arr \ {a}
        # This is hard to check without rebuilding DP.
        # Fast necessary check: for each a, (t+1)*a - x must be achievable.
        # Use the existing dp of arr: (t+1)*a - x is sum of t elements from arr\{a}
        # = (t+1)*a - x is sum of t elements from arr, none of which is a (since a not in subset)
        # But dp[t] includes subsets that may contain a.
        # If (t+1)*a - x is a subset-sum of size t from arr, and that subset doesn't contain a,
        # then it's a valid witness. But we can't easily exclude a.
        # Conservative: if (t+1)*a - x is a subset-sum of size t from arr, it MIGHT be valid.
        # For safety, we'll do a full check only when this passes.
        for a in arr:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total:
                    break
                if (dp[t] >> target) & 1:
                    # Potential conflict; need to verify it's not using a
                    # Rebuild DP without a for this specific check
                    others = [v for v in arr if v != a]
                    dp_others = build_dp(others)
                    total_others = total - a
                    if target <= total_others and (dp_others[t] >> target) & 1:
                        return False
        return True

    # Greedy build with conflict-aware ordering
    def greedy_build(order, max_size=target_size):
        selected = []
        sel_dp = [1]
        sel_sum = 0
        for x in order:
            if len(selected) >= max_size:
                break
            if x in selected:
                continue
            if can_add(selected, x, sel_dp, sel_sum):
                selected.append(x)
                sel_sum += x
                sel_dp = build_dp(selected)
        return selected

    # Compute conflict counts for ordering
    conflict_count = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= n and mid != a and mid != b:
                    conflict_count[a] += 1
                    conflict_count[b] += 1

    # Multiple greedy orders
    best = []
    best_size = 0

    orders = [
        sorted(range(1, n + 1), key=lambda x: (conflict_count[x], x)),
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
    ]

    for order in orders:
        if time.time() > deadline - 15.0:
            break
        result = greedy_build(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= target_size:
                return best

    # Simulated annealing on bitmask
    def anneal(start_set, max_iters=20000):
        selected = list(start_set)
        sel_set = set(selected)
        cur_dp = build_dp(selected)
        cur_sum = sum(selected)
        best_set = list(selected)
        best_size_local = len(selected)
        T = 50.0
        cool = 0.9995

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Random move: add or remove a random element
            if random.random() < 0.5 and len(selected) < n:
                # Add
                candidates = [x for x in range(1, n + 1) if x not in sel_set]
                if not candidates:
                    break
                x = random.choice(candidates)
                if can_add(selected, x, cur_dp, cur_sum):
                    selected.append(x)
                    sel_set.add(x)
                    cur_sum += x
                    cur_dp = build_dp(selected)
                    if len(selected) > best_size_local:
                        best_set = list(selected)
                        best_size_local = len(selected)
                else:
                    # Accept with probability based on T (but invalid states are bad)
                    # We don't want to go invalid, so skip
                    pass
            else:
                # Remove
                if len(selected) <= 2:
                    continue
                i = random.randrange(len(selected))
                x = selected[i]
                selected.pop(i)
                sel_set.discard(x)
                cur_sum -= x
                cur_dp = build_dp(selected)
                # Always accept removal (it preserves validity)

        return best_set, best_size_local

    # Run annealing from best
    if best_size < target_size:
        improved, improved_size = anneal(best, max_iters=30000)
        if improved_size > best_size:
            best = improved
            best_size = improved_size
            if best_size >= target_size:
                return best

    # Final: try to add any remaining valid elements
    if best_size < target_size and time.time() < deadline - 3.0:
        best_set = set(best)
        sel_dp = build_dp(best)
        sel_sum = sum(best)
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if can_add(best, x, sel_dp, sel_sum):
                best.append(x)
                best_set.add(x)
                sel_sum += x
                sel_dp = build_dp(best)
                if len(best) >= target_size:
                    break

    return best
