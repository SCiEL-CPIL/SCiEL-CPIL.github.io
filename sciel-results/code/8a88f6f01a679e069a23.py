import numpy as np
import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s

    # 1. Greedy nearest-neighbor reordering of (n-1)-permutations for better overlap
    #    Standard construction: for each (n-1)-perm p, emit block p + [n] + p
    #    The overlap between consecutive blocks depends on ordering.
    #    Greedy: start with first perm, then always pick the next perm that
    #    maximizes suffix-prefix overlap with the current end of the word.
    
    # First, generate all (n-1)-permutations in the standard order
    # (they appear in the (n-1)-superpermutation in first-appearance order)
    # We'll use the standard recursive construction up to n-1 to get the order,
    # then reorder using greedy nearest-neighbor for the final step.
    
    # Build the (n-1) superpermutation to extract the permutation order
    def standard_superperm(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                best_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        best_overlap = o
                        break
                new += block[best_overlap:]
            word = new
        return word

    if n <= 1:
        return "1"
    
    # Get the (n-1) superpermutation and extract the (n-1)-permutation order
    sp_n1 = standard_superperm(n - 1)
    perms_n1, seen = [], set()
    for i in range(len(sp_n1) - (n - 1) + 1):
        window = tuple(sp_n1[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_n1.append(window)
    
    N = len(perms_n1)
    
    # Precompute overlap matrix
    # overlaps[i][j] = longest suffix of perms_n1[i] that is a prefix of perms_n1[j]
    overlaps = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        a = perms_n1[i]
        for j in range(N):
            b = perms_n1[j]
            max_ov = n - 1
            ov = 0
            for k in range(1, max_ov + 1):
                if a[-k] == b[k-1]:
                    ov = k
                else:
                    break
            overlaps[i][j] = ov

    # Multiple random greedy starts
    best_order = None
    best_len = float('inf')
    
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        remaining = list(range(N))
        start_idx = random.choice(remaining)
        remaining.remove(start_idx)
        order = [start_idx]
        current = start_idx
        
        while remaining:
            best_idx = -1
            best_ov = -1
            for idx in remaining:
                ov = overlaps[current][idx]
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            order.append(best_idx)
            remaining.remove(best_idx)
            current = best_idx
            
        # Build word
        word = []
        for idx in order:
            p = perms_n1[idx]
            block = list(p) + [n] + list(p)
            best_overlap = 0
            for o in range(min(len(word), len(block)), -1, -1):
                if word[len(word) - o:] == block[:o]:
                    best_overlap = o
                    break
            word += block[best_overlap:]
            
        if len(word) < best_len:
            best_len = len(word)
            best_order = order[:]
            
    if best_order is None:
        # Fallback to original greedy
        remaining = list(range(N))
        order = [remaining.pop(0)]
        current = order[0]
        while remaining:
            best_idx = -1
            best_ov = -1
            for idx in remaining:
                ov = overlaps[current][idx]
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            order.append(best_idx)
            remaining.remove(best_idx)
            current = best_idx
        best_order = order

    # Build final word from best_order
    word = []
    for idx in best_order:
        p = perms_n1[idx]
        block = list(p) + [n] + list(p)
        best_overlap = 0
        for o in range(min(len(word), len(block)), -1, -1):
            if word[len(word) - o:] == block[:o]:
                best_overlap = o
                break
        word += block[best_overlap:]
        
    S = list(word)
    L = len(S)

    # 2. Build counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    best_S = S[:]
    best_len = L

    def rebuild_counts(S):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i + n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def is_valid(S):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        expected = 1
        for k in range(1, n + 1):
            expected *= k
        return len(seen) == expected

    # 3. Removal pass: scan positions, try to remove if no window drops to 0
    def removal_pass(S, counts, deadline):
        removed_any = False
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if not valid:
                i -= 1
                continue

            # Perform removal
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                c = counts[p] - 1
                if c == 0:
                    del counts[p]
                else:
                    counts[p] = c

            # Add straddling windows
            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            for k in range(start_new, end_new + 1):
                wc = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    wc.append(S[s_idx])
                p_new = tuple(wc)
                counts[p_new] = counts.get(p_new, 0) + 1

            S.pop(i)
            removed_any = True

        return removed_any

    # 3b. Bidirectional removal
    def removal_pass_bidirectional(S, counts, deadline):
        removed_any = False
        left = 0
        right = len(S) - 1
        while left <= right and time.time() <= deadline:
            # Try removing from right
            i = right
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if valid:
                for j in range(start_old, end_old + 1):
                    p = tuple(S[j:j + n])
                    c = counts[p] - 1
                    if c == 0:
                        del counts[p]
                    else:
                        counts[p] = c
                new_len = len(S) - 1
                start_new = max(0, i - n + 1)
                end_new = min(i - 1, new_len - n)
                for k in range(start_new, end_new + 1):
                    wc = []
                    for idx in range(k, k + n):
                        s_idx = idx if idx < i else idx + 1
                        wc.append(S[s_idx])
                    p_new = tuple(wc)
                    counts[p_new] = counts.get(p_new, 0) + 1
                S.pop(i)
                removed_any = True
                right -= 1
            else:
                right -= 1

            # Try removing from left
            if left > right:
                break
            i = left
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if valid:
                for j in range(start_old, end_old + 1):
                    p = tuple(S[j:j + n])
                    c = counts[p] - 1
                    if c == 0:
                        del counts[p]
                    else:
                        counts[p] = c
                new_len = len(S) - 1
                start_new = max(0, i - n + 1)
                end_new = min(i - 1, new_len - n)
                for k in range(start_new, end_new + 1):
                    wc = []
                    for idx in range(k, k + n):
                        s_idx = idx if idx < i else idx + 1
                        wc.append(S[s_idx])
                    p_new = tuple(wc)
                    counts[p_new] = counts.get(p_new, 0) + 1
                S.pop(i)
                removed_any = True
                left += 1
            else:
                left += 1

        return removed_any

    # 4. Perturbation strategies
    def perturb_swap(S, counts):
        i, j = random.sample(range(len(S)), 2)
        S[i], S[j] = S[j], S[i]
        counts = rebuild_counts(S)
        return S, counts

    def perturb_insert(S, counts):
        pos = random.randint(0, len(S))
        sym = random.randint(1, n)
        S.insert(pos, sym)
        counts = rebuild_counts(S)
        return S, counts

    # 5. Main loop with bidirectional removal
    stuck_rounds = 0
    max_stuck = 50
    while time.time() < deadline:
        if stuck_rounds % 2 == 0:
            removed = removal_pass(S, counts, deadline)
        else:
            removed = removal_pass_bidirectional(S, counts, deadline)
        if removed:
            best_S = S[:]
            best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            r = random.random()
            if r < 0.4 and len(S) > n:
                S, counts = perturb_swap(S, counts)
            elif r < 0.7 and len(S) < best_len * 1.15:
                S, counts = perturb_insert(S, counts)
            else:
                S, counts = perturb_swap(S, counts)

    # 6. Final polish: multiple removal passes on the best found
    if time.time() < deadline - 3:
        S2 = best_S[:]
        counts2 = rebuild_counts(S2)
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            removal_pass(S2, counts2, deadline)
            removal_pass_bidirectional(S2, counts2, deadline)
        if len(S2) < len(best_S) and is_valid(S2):
            best_S = S2

    # 7. Validate
    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
