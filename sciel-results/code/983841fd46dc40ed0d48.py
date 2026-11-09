import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 82  # 90s limit, leave 8s margin

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
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}

    # Precompute incompatibility mask for each smooth number
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a % b == 0 or b % a == 0:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)

    # Precompute for each target: candidate indices, sorted values, and prefix sums
    target_data = {}
    for n in range(1, 129):
        cand_indices = [i for i in range(n_smooth) if smooth_list[i] <= n]
        cand_indices.sort(key=lambda i: smooth_list[i])
        cand_vals = [smooth_list[i] for i in cand_indices]
        # Precompute prefix sums for O(1) range sum queries
        prefix = [0] * (len(cand_vals) + 1)
        for i in range(len(cand_vals)):
            prefix[i + 1] = prefix[i] + cand_vals[i]
        target_data[n] = (cand_indices, cand_vals, prefix)

    def find_min_antichain(n):
        """Find minimum k such that there's an antichain of k smooth numbers summing to n."""
        cand_indices, cand_vals, prefix = target_data[n]
        if not cand_indices:
            return None

        # Try k = 1, 2, 3, ...
        for k in range(1, 31):
            if time.time() - start_time > DEADLINE:
                break

            if len(cand_vals) < k:
                continue

            # Quick feasibility check using prefix sums (O(1))
            total_cand = len(cand_vals)
            min_possible = prefix[k] - prefix[0]
            max_possible = prefix[total_cand] - prefix[total_cand - k]
            if n > max_possible or n < min_possible:
                continue

            # Use DFS with memoization on failed states
            failed_states = set()

            sys.setrecursionlimit(10000)

            def dfs(start, rem_sum, rem_k, used_mask):
                """Returns list of smooth-number indices if valid antichain found, else None."""
                if rem_k == 0:
                    if rem_sum == 0:
                        return []
                    return None

                if start >= len(cand_indices):
                    return None

                state_key = (start, rem_sum, rem_k, used_mask)
                if state_key in failed_states:
                    return None

                # Pruning: check if enough candidates remain
                avail_count = len(cand_indices) - start
                if rem_k > avail_count:
                    failed_states.add(state_key)
                    return None

                # Min and max possible sums using prefix sums (O(1))
                min_possible = prefix[start + rem_k] - prefix[start]
                max_possible = prefix[start + avail_count] - prefix[start + avail_count - rem_k]
                if rem_sum > max_possible or rem_sum < min_possible:
                    failed_states.add(state_key)
                    return None

                # Try each candidate from start onwards
                for i in range(start, len(cand_indices)):
                    ci = cand_indices[i]
                    val = smooth_list[ci]

                    if val > rem_sum:
                        continue

                    # Check compatibility with all used
                    compatible = True
                    mask = used_mask
                    while mask:
                        low_bit = mask & (-mask)
                        idx_in_cand = low_bit.bit_length() - 1
                        pi = cand_indices[idx_in_cand]
                        if incompat[pi] & (1 << ci):
                            compatible = False
                            break
                        mask ^= low_bit

                    if not compatible:
                        continue

                    new_mask = used_mask | (1 << i)
                    result = dfs(i + 1, rem_sum - val, rem_k - 1, new_mask)
                    if result is not None:
                        return [ci] + result

                failed_states.add(state_key)
                return None

            path_indices = dfs(0, n, k, 0)
            if path_indices is not None:
                return [smooth_list[i] for i in path_indices]

        return None

    answers = []
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Emergency fallback
            if n in val_to_idx:
                answers.append([n])
            else:
                ans = find_min_antichain(n)
                answers.append(ans if ans else [n])
            continue

        ans = find_min_antichain(n)
        if ans is None:
            # Fallback
            if n in val_to_idx:
                ans = [n]
            else:
                ans = [n]
        answers.append(ans)

    return answers
