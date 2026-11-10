import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin vs 150s kill

    # ---- small n: standard (optimal) construction ----
    if n <= 3:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for k in range(min(len(new), len(block)), 0, -1):
                    if new[len(new) - k:] == block[:k]:
                        overlap = k
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # ---- n >= 4: reorder the (n-1)-perm blocks to maximize overlap ----
    base_symbols = list(range(1, n))
    # standard recursive superpermutation S_{n-1}
    s_prev = [1]
    for m in range(2, n):
        if time.time() > deadline - 5:
            break
        perms, seen = [], set()
        for i in range(len(s_prev) - m + 2):
            window = tuple(s_prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for k in range(min(len(new), len(block)), 0, -1):
                if new[len(new) - k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        s_prev = new

    # order of (n-1)-permutations as they appear in S_{n-1}
    perm_list, seen = [], set()
    for i in range(len(s_prev) - (n - 1) + 1):
        w = tuple(s_prev[i:i + n - 1])
        if len(set(w)) == n - 1 and w not in seen:
            seen.add(w)
            perm_list.append(w)
    M = len(perm_list)
    if M != (n - 1) ** (n - 1) and M < 1:
        M = len(perm_list)

    # map (n-1)-perm tuple -> index
    perm_to_idx = {p: i for i, p in enumerate(perm_list)}

    # overlap matrix: O[i][j] = max k s.t. perm_list[i][-k:] == perm_list[j][:k]
    O = np.zeros((M, M), dtype=np.int8)
    for k in range(n - 1, 0, -1):
        suffix_map = {}
        prefix_map = {}
        for idx, p in enumerate(perm_list):
            suff = p[-k:]
            pref = p[:k]
            suffix_map.setdefault(suff, []).append(idx)
            prefix_map.setdefault(pref, []).append(idx)
        for suff, i_list in suffix_map.items():
            if suff in prefix_map:
                j_list = prefix_map[suff]
                for i in i_list:
                    for j in j_list:
                        if O[i, j] == 0:
                            O[i, j] = k

    # ---- initial order: the standard recursive order (high overlap) ----
    def standard_order():
        # S_{n-1} was built from (n-2)-perms; the block order at the last step
        # equals the order of (n-1)-perms in S_{n-1}. We already have that as perm_list order.
        return list(range(M))

    def total_overlap(order):
        total = 0
        for i in range(M - 1):
            total += O[order[i], order[i + 1]]
        return total

    def length_from(order):
        # total length = sum over blocks (2(n-1)+1) minus overlaps
        return M * (2 * (n - 1) + 1) - total_overlap(order)

    best_order = standard_order()
    best_ov = total_overlap(best_order)

    # ---- local search: swap moves ----
    def delta_swap(order, i, j):
        if i == j:
            return 0
        if abs(i - j) == 1:
            if i > j:
                i, j = j, i
            left = order[i - 1] if i > 0 else None
            mid1 = order[i]
            mid2 = order[i + 1]
            right = order[i + 2] if i + 2 < M else None
            old = 0
            if left is not None:
                old += O[left, mid1]
            old += O[mid1, mid2]
            if right is not None:
                old += O[mid2, right]
            new = 0
            if left is not None:
                new += O[left, mid2]
            new += O[mid2, mid1]
            if right is not None:
                new += O[mid1, right]
            return new - old
        else:
            if i > j:
                i, j = j, i
            L1 = order[i - 1] if i > 0 else None
            B = order[i]
            R1 = order[i + 1]
            L2 = order[j - 1]
            E = order[j]
            R2 = order[j + 1] if j + 1 < M else None
            old = 0
            if L1 is not None:
                old += O[L1, B]
            old += O[B, R1]
            old += O[L2, E]
            if R2 is not None:
                old += O[E, R2]
            new = 0
            if L1 is not None:
                new += O[L1, E]
            new += O[E, R1]
            new += O[L2, B]
            if R2 is not None:
                new += O[B, R2]
            return new - old

    def delta_2opt(order, a, b):
        # reverse segment [a..b]; only boundaries a-1|a and b|b+1 change
        if a == b:
            return 0
        L = order[a - 1] if a > 0 else None
        R = order[b + 1] if b + 1 < M else None
        old = 0
        if L is not None:
            old += O[L, order[a]]
        if R is not None:
            old += O[order[b], R]
        new = 0
        if L is not None:
            new += O[L, order[b]]
        if R is not None:
            new += O[order[a], R]
        return new - old

    def local_search(order, ov):
        improved = True
        while time.time() < deadline - 2 and improved:
            improved = False
            # pass 1: random swaps
            for _ in range(20000):
                if time.time() > deadline - 2:
                    break
                i, j = random.sample(range(M), 2)
                d = delta_swap(order, i, j)
                if d > 0:
                    order[i], order[j] = order[j], order[i]
                    ov += d
                    improved = True
            # pass 2: random 2-opt
            for _ in range(4000):
                if time.time() > deadline - 2:
                    break
                a, b = random.sample(range(M), 2)
                if a > b:
                    a, b = b, a
                d = delta_2opt(order, a, b)
                if d > 0:
                    order[a:b + 1] = reversed(order[a:b + 1])
                    ov += d
                    improved = True
        return order, ov

    # ---- random restarts (fresh independent orders) to escape local minima ----
    for _ in range(400):
        if time.time() > deadline - 3:
            break
        cand = list(range(M))
        random.shuffle(cand)
        cand, ov = local_search(cand, total_overlap(cand))
        if ov > best_ov:
            best_ov = ov
            best_order = cand

    # ---- polish the best order with more swaps + 2-opt ----
    polish = best_order[:]
    polish, ov = local_search(polish, total_overlap(polish))
    if ov > best_ov:
        best_ov = ov
        best_order = polish

    # ---- reconstruct string ----
    res = []
    first = list(perm_list[best_order[0]]) + [n] + list(perm_list[best_order[0]])
    res.extend(first)
    for idx in range(1, M):
        prev = best_order[idx - 1]
        curr = best_order[idx]
        block = list(perm_list[curr]) + [n] + list(perm_list[curr])
        ov = O[prev, curr]
        res.extend(block[ov:])
    result_str = "".join(map(str, res))

    # ---- verify; fallback to standard construction if invalid ----
    target = 1
    for x in range(1, n + 1):
        target *= x
    seen = set()
    for i in range(len(result_str) - n + 1):
        window = tuple(map(int, result_str[i:i + n]))
        if len(set(window)) == n:
            seen.add(window)
    if len(seen) == target:
        return result_str

    # fallback: standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen_p = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen_p:
                seen_p.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for k in range(min(len(new_word), len(block)), 0, -1):
                if new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            new_word += block[overlap:]
        word = new_word
    return "".join(map(str, word))
