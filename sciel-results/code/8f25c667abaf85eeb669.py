"""Optimized solver for minimum distinct divisors of 5040 summing to 4979."""
import time

# Precompute divisors of 5040, sorted descending
_DIVS = sorted([d for d in range(1, 5041) if 5040 % d == 0], reverse=True)
_N = len(_DIVS)
# Suffix sums: _SUF[i] = sum of _DIVS[i:] (sum of all divisors from index i to end)
_SUF = [0] * (_N + 1)
for _i in range(_N - 1, -1, -1):
    _SUF[_i] = _SUF[_i + 1] + _DIVS[_i]

_TARGET = 4979
_N_MAX = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill


def construct():
    start_time = time.time()
    best = None  # (k, chosen_list)

    # Quick greedy upper bound for initial bound
    rem = _TARGET
    greedy = []
    for d in _DIVS:
        if d <= rem:
            greedy.append(d)
            rem -= d
            if rem == 0:
                break
    if rem == 0:
        best = (len(greedy), greedy)

    # Iterative deepening: try k = 1, 2, 3, ...
    # Upper bound on k: len(greedy) if found, else _N_MAX
    upper_k = best[0] if best else _N_MAX

    def dfs(k_remaining, start_idx, current_sum, path):
        nonlocal best
        # Check deadline
        if time.time() - start_time > _DEADLINE:
            return False
        if k_remaining == 0:
            if current_sum == _TARGET:
                nonlocal_best = list(path)
                best = (len(nonlocal_best), nonlocal_best)
                return True
            return False
        if current_sum > _TARGET:
            return False
        # Pruning: even taking the k_remaining largest available, can we reach target?
        # We need current_sum + (sum of k_remaining largest from start_idx onward) >= _TARGET
        # But we also need enough elements: _N - start_idx >= k_remaining
        if _N - start_idx < k_remaining:
            return False
        # Upper bound check: max possible sum with k_remaining from start_idx
        # Take the k_remaining largest from start_idx
        max_possible = 0
        for i in range(start_idx, min(start_idx + k_remaining, _N)):
            max_possible += _DIVS[i]
        if current_sum + max_possible < _TARGET:
            return False
        # Lower bound: min possible sum (take k_remaining smallest from start_idx)
        # Actually we need current_sum + min_possible <= _TARGET for feasibility
        # min_possible = sum of k_remaining smallest from start_idx to end
        # But for distinct divisors, the smallest available are at the end
        # Let's compute: we need to pick k_remaining from indices start_idx.._N-1
        # The minimum sum is the k_remaining smallest, which are the last k_remaining
        min_possible = 0
        for i in range(_N - 1, _N - 1 - k_remaining, -1):
            if i < start_idx:
                break
            min_possible += _DIVS[i]
        if current_sum + min_possible > _TARGET:
            return False

        # Branch: try each divisor from start_idx
        # We can also prune: if _DIVS[i] > _TARGET - current_sum, skip (too big)
        for i in range(start_idx, _N):
            d = _DIVS[i]
            if d > _TARGET - current_sum:
                continue
            # If we take d, remaining sum needed is _TARGET - current_sum - d
            # with k_remaining - 1 more divisors from i+1 onward
            path.append(d)
            if dfs(k_remaining - 1, i + 1, current_sum + d, path):
                path.pop()
                return True
            path.pop()
            # Check deadline inside loop too
            if time.time() - start_time > _DEADLINE:
                return False
        return False

    for k in range(1, upper_k + 1):
        if time.time() - start_time > _DEADLINE:
            break
        # Quick feasibility: sum of k largest divisors must be >= _TARGET
        sum_k_largest = sum(_DIVS[:k])
        if sum_k_largest < _TARGET:
            continue
        # Also, sum of k smallest divisors must be <= _TARGET
        sum_k_smallest = sum(_DIVS[_N - k:])
        if sum_k_smallest > _TARGET:
            continue
        # Try DFS
        if dfs(k, 0, 0, []):
            if best is not None:
                return best[1]
            break

    # Return best found
    if best is not None:
        return best[1]
    # Fallback: greedy (should not happen for practical numbers)
    return greedy


# Precompute at module load for speed
_DIVS = sorted([d for d in range(1, 5041) if 5040 % d == 0], reverse=True)
_N = len(_DIVS)
_SUF = [0] * (_N + 1)
for _i in range(_N - 1, -1, -1):
    _SUF[_i] = _SUF[_i + 1] + _DIVS[_i]
_TARGET = 4979
_N_MAX = 256
_DEADLINE = 85.0
