import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 55  # Safety margin for 60s limit

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)

    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    smooth_set = set(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}

    # Precompute incompatibility: for each pair, check if one divides the other
    # incompat[i] is a set of indices j where smooth_list[i] and smooth_list[j] are incompatible
    incompat = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a % b == 0 or b % a == 0:
                incompat[i].add(j)
                incompat[j].add(i)

    def is_antichain(indices):
        """Check if a list of smooth indices forms an antichain."""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if indices[j] in incompat[indices[i]]:
                    return False
        return True

    # Precompute all antichains of size 1, 2, 3 and map (sum, size) -> list of antichains
    # For size 1: trivial
    antichains_by_sum_size = {}  # key: (sum, size), value: list of index tuples

    # Size 1
    for i, v in enumerate(smooth_list):
        key = (v, 1)
        if key not in antichains_by_sum_size:
            antichains_by_sum_size[key] = []
        antichains_by_sum_size[key].append((i,))

    # Size 2
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j not in incompat[i]:
                s = smooth_list[i] + smooth_list[j]
                if s <= 128:
                    key = (s, 2)
                    if key not in antichains_by_sum_size:
                        antichains_by_sum_size[key] = []
                    antichains_by_sum_size[key].append((i, j))

    # Size 3
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j in incompat[i]:
                continue
            for k in range(j + 1, n_smooth):
                if k in incompat[i] or k in incompat[j]:
                    continue
                s = smooth_list[i] + smooth_list[j] + smooth_list[k]
                if s <= 128:
                    key = (s, 3)
                    if key not in antichains_by_sum_size:
                        antichains_by_sum_size[key] = []
                    antichains_by_sum_size[key].append((i, j, k))

    def find_solution(n, max_k=30):
        """Find minimum-k antichain representation for target n."""
        # Try k = 1, 2, 3, ...
        for k in range(1, max_k + 1):
            if time.time() - start_time > DEADLINE:
                break

            key = (n, k)
            if key in antichains_by_sum_size:
                # Return the first one (any valid one is fine for minimization)
                idx_tuple = antichains_by_sum_size[key][0]
                return [smooth_list[i] for i in idx_tuple]

            # For k >= 4, use DFS search
            if k >= 4:
                # Quick feasibility: need at least k distinct smooth numbers <= n
                candidates = [i for i in range(n_smooth) if smooth_list[i] <= n]
                if len(candidates) < k:
                    continue

                # DFS with pruning
                sys.setrecursionlimit(10000)
                failed_states = set()

                def dfs(start_idx, rem_sum, rem_k, used_indices):
                    if rem_k == 0:
                        return [] if rem_sum == 0 else None

                    if start_idx >= n_smooth:
                        return None

                    state_key = (start_idx, rem_sum, rem_k, tuple(sorted(used_indices)))
                    if state_key in failed_states:
                        return None

                    # Pruning: check if enough candidates remain
                    remaining_candidates = n_smooth - start_idx
                    if rem_k > remaining_candidates:
                        failed_states.add(state_key)
                        return None

                    # Try each candidate from start_idx onwards
                    for i in range(start_idx, n_smooth):
                        val = smooth_list[i]
                        if val > rem_sum:
                            continue

                        # Check compatibility with all used
                        compatible = True
                        for u in used_indices:
                            if i in incompat[u]:
                                compatible = False
                                break

                        if not compatible:
                            continue

                        new_used = used_indices | {i}
                        result = dfs(i + 1, rem_sum - val, rem_k - 1, new_used)
                        if result is not None:
                            return [i] + result

                    failed_states.add(state_key)
                    return None

                result = dfs(0, n, k, frozenset())
                if result is not None:
                    return [smooth_list[i] for i in result]

        return None

    answers = []
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Emergency fallback: use any valid representation
            if n in smooth_set:
                answers.append([n])
            else:
                # Try to find any solution
                for k in range(2, 31):
                    key = (n, k)
                    if key in antichains_by_sum_size:
                        idx_tuple = antichains_by_sum_size[key][0]
                        answers.append([smooth_list[i] for i in idx_tuple])
                        break
                else:
                    answers.append([n])  # Last resort (may be invalid)
            continue

        ans = find_solution(n)
        if ans is None:
            # Fallback
            if n in smooth_set:
                ans = [n]
            else:
                ans = [n]
        answers.append(ans)

    return answers
