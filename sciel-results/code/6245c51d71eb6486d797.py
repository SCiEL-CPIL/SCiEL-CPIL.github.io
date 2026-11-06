import time
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin under 90s wall-clock

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    n = len(smooth)
    is_smooth = set(smooth)

    # 2. Precompute pairwise conflict: conflict[i][j] = True if smooth[i] divides smooth[j] or vice versa
    # We only care about distinct elements in an antichain
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = smooth[i], smooth[j]
            if a < b:
                if b % a == 0:
                    conflict[i][j] = conflict[j][i] = True
            else:
                if a % b == 0:
                    conflict[i][j] = conflict[j][i] = True

    # 3. best[n] = minimum number of terms for target n, and the actual tuple
    best_count = [1000] * 129
    best_tuple = [None] * 129

    # 4. DFS to enumerate all valid antichains (subsets of smooth numbers, no element divides another)
    #    For each antichain, record its sum and length.
    #    We use bitmask for the antichain to check conflicts quickly.
    #    State: (current_mask, start_idx, current_sum, current_len)
    #    Pruning: if current_len >= best_count[current_sum], we still need to explore
    #    because extending might give a better (shorter) solution for a larger sum? No -
    #    extending always increases length. So if current_len >= best_count[current_sum],
    #    any extension will have length > current_len >= best_count[current_sum],
    #    but it might cover a different sum. So we can't prune based on current_sum alone.
    #    However, if current_sum > 128, stop.
    #    Also, we can prune if current_len >= 30 (max allowed).

    # Better approach: enumerate all antichains, and for each, update best for its sum.
    # The number of antichains of 5-smooth numbers up to 128 should be manageable.

    # Let's use a different strategy: for each target n from 1 to 128, find the minimum
    # number of terms. We can do this by checking k=1,2,3,... and for each k, checking
    # all combinations of k smooth numbers that sum to n and form an antichain.

    # But that's expensive. Instead, let's enumerate all antichains once.

    # Enumerate all valid antichains via DFS
    # An antichain is a subset where no element divides another.
    # We build it by choosing elements in increasing order of index.

    def dfs_anticheins(start_idx, current_mask, current_sum, current_len):
        if time.time() > DEADLINE:
            return
        if current_sum > 128:
            return
        if current_len > 30:
            return

        # Update best for current_sum
        if current_sum > 0 and current_len < best_count[current_sum]:
            best_count[current_sum] = current_len
            best_tuple[current_sum] = current_mask

        # Try adding more elements
        for i in range(start_idx, n):
            val = smooth[i]
            if current_sum + val > 128:
                break
            # Check if val conflicts with any element in current_mask
            ok = True
            for j in range(n):
                if (current_mask >> j) & 1:
                    if conflict[j][i]:
                        ok = False
                        break
            if not ok:
                continue
            new_mask = current_mask | (1 << i)
            dfs_anticheins(i + 1, new_mask, current_sum + val, current_len + 1)
            if time.time() > DEADLINE:
                return

    dfs_anticheins(0, 0, 0, 0)

    # 5. For any target not covered, do a targeted search
    for target in range(1, 129):
        if best_count[target] < 1000:
            continue
        # Try k=1,2,3,...
        for k in range(1, 31):
            if time.time() > DEADLINE:
                break
            found = False
            # For k=1
            if k == 1:
                if target in is_smooth:
                    idx = smooth.index(target)
                    best_count[target] = 1
                    best_tuple[target] = 1 << idx
                    found = True
            elif k == 2:
                for i in range(n):
                    if smooth[i] >= target:
                        break
                    rem = target - smooth[i]
                    if rem <= smooth[i]:
                        continue
                    if rem in is_smooth:
                        j = smooth.index(rem)
                        if not conflict[i][j]:
                            best_count[target] = 2
                            best_tuple[target] = (1 << i) | (1 << j)
                            found = True
                            break
            else:
                # For k >= 3, do a limited search
                # Try combinations of k smooth numbers
                # This is expensive, so limit to small k
                if k <= 5:
                    # Use recursive search for k elements
                    def search_k(rem, k_rem, start_idx, chosen_mask, chosen_list):
                        if time.time() > DEADLINE:
                            return None
                        if k_rem == 0:
                            if rem == 0:
                                return chosen_mask
                            return None
                        if rem < 0:
                            return None
                        # Lower bound: minimum possible sum of k_rem elements >= start_idx
                        # is sum of the k_rem smallest smooth numbers >= start_idx
                        # But they must be > last chosen element
                        # Simplified: just iterate
                        for i in range(start_idx, n):
                            val = smooth[i]
                            if val > rem:
                                break
                            # Check conflict with all chosen
                            ok = True
                            for j in range(n):
                                if (chosen_mask >> j) & 1:
                                    if conflict[j][i]:
                                        ok = False
                                        break
                            if not ok:
                                continue
                            new_mask = chosen_mask | (1 << i)
                            res = search_k(rem - val, k_rem - 1, i + 1, new_mask, chosen_list + [val])
                            if res is not None:
                                return res
                        return None

                    res = search_k(target, k, 0, 0, [])
                    if res is not None:
                        best_count[target] = k
                        best_tuple[target] = res
                        found = True
                        break
            if found:
                break

    # 6. Construct result
    result = []
    for target in range(1, 129):
        if best_tuple[target] is not None:
            mask = best_tuple[target]
            terms = []
            for i in range(n):
                if (mask >> i) & 1:
                    terms.append(smooth[i])
            terms.sort()
            result.append(terms)
        else:
            # Should not happen, but fallback: use [target] if smooth, else try 2 terms
            if target in is_smooth:
                result.append([target])
            else:
                found = False
                for a in smooth:
                    if a >= target:
                        break
                    b = target - a
                    if b <= a:
                        continue
                    if b in is_smooth:
                        if b % a != 0 and a % b != 0:
                            result.append([a, b])
                            found = True
                            break
                if not found:
                    # 3 terms
                    found = False
                    for a in smooth:
                        if 3 * a >= target:
                            break
                        for b in smooth:
                            if b <= a:
                                continue
                            if a + 2 * b >= target:
                                break
                            c = target - a - b
                            if c <= b:
                                continue
                            if c in is_smooth:
                                if (c % a != 0 and a % c != 0 and
                                    c % b != 0 and b % c != 0 and
                                    b % a != 0 and a % b != 0):
                                    result.append([a, b, c])
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        result.append([target])
    return result
