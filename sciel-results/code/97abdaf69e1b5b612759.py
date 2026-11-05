import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 112  # 8s safety margin before 120s

    # 1. Greedy earliest-insertion superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build blocks for each permutation
        blocks = []
        for p in perms:
            block = list(p) + [m] + list(p)
            blocks.append(block)
        
        # Greedy: pick next block with best overlap to current word
        new = []
        remaining = list(range(len(blocks)))
        while remaining:
            best_idx = -1
            best_overlap = -1
            for idx in remaining:
                block = blocks[idx]
                o = 0
                max_o = min(len(new), len(block))
                for o in range(max_o, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        break
                if o > best_overlap:
                    best_overlap = o
                    best_idx = idx
            if best_idx == -1:
                best_idx = remaining[0]
                best_overlap = 0
                block = blocks[best_idx]
            else:
                block = blocks[best_idx]
            new += block[best_overlap:]
            remaining.remove(best_idx)
        word = new

    S = list(word)
    L = len(S)

    # 2. Build counts
    def rebuild_counts(S):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i + n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    counts = rebuild_counts(S)
    at_risk = set(p for p, c in counts.items() if c == 1)

    best_S = S[:]
    best_len = L

    def is_valid(S):
        if len(S) < n:
            return False
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        return len(seen) == len(range(1, n+1)) * (1 if n == 1 else __import__('math').factorial(n))

    # 3. Removal pass: scan positions, try to remove if no at-risk window is destroyed
    def removal_pass(S, counts, at_risk, deadline):
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

            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                c = counts[p] - 1
                if c == 0:
                    del counts[p]
                    at_risk.discard(p)
                else:
                    counts[p] = c
                    if c == 1:
                        at_risk.add(p)

            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            for k in range(start_new, end_new + 1):
                wc = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    wc.append(S[s_idx])
                p_new = tuple(wc)
                c = counts.get(p_new, 0)
                if c == 0:
                    counts[p_new] = 1
                    at_risk.add(p_new)
                elif c == 1:
                    counts[p_new] = 2
                    at_risk.discard(p_new)
                else:
                    counts[p_new] = c + 1

            S.pop(i)
            removed_any = True

        return removed_any

    # 4. Bidirectional removal
    def removal_pass_bidirectional(S, counts, at_risk, deadline):
        removed_any = False
        left = 0
        right = len(S) - 1
        while left <= right and time.time() <= deadline:
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
                        at_risk.discard(p)
                    else:
                        counts[p] = c
                        if c == 1:
                            at_risk.add(p)
                new_len = len(S) - 1
                start_new = max(0, i - n + 1)
                end_new = min(i - 1, new_len - n)
                for k in range(start_new, end_new + 1):
                    wc = []
                    for idx in range(k, k + n):
                        s_idx = idx if idx < i else idx + 1
                        wc.append(S[s_idx])
                    p_new = tuple(wc)
                    c = counts.get(p_new, 0)
                    if c == 0:
                        counts[p_new] = 1
                        at_risk.add(p_new)
                    elif c == 1:
                        counts[p_new] = 2
                        at_risk.discard(p_new)
                    else:
                        counts[p_new] = c + 1
                S.pop(i)
                removed_any = True
                right -= 1
            else:
                right -= 1

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
                        at_risk.discard(p)
                    else:
                        counts[p] = c
                        if c == 1:
                            at_risk.add(p)
                new_len = len(S) - 1
                start_new = max(0, i - n + 1)
                end_new = min(i - 1, new_len - n)
                for k in range(start_new, end_new + 1):
                    wc = []
                    for idx in range(k, k + n):
                        s_idx = idx if idx < i else idx + 1
                        wc.append(S[s_idx])
                    p_new = tuple(wc)
                    c = counts.get(p_new, 0)
                    if c == 0:
                        counts[p_new] = 1
                        at_risk.add(p_new)
                    elif c == 1:
                        counts[p_new] = 2
                        at_risk.discard(p_new)
                    else:
                        counts[p_new] = c + 1
                S.pop(i)
                removed_any = True
                left += 1
            else:
                left += 1

        return removed_any

    # 5. Targeted perturbation: modify symbols at positions where removal is blocked
    def targeted_perturb(S, counts, at_risk):
        """Find positions where removal is blocked, and try changing the symbol there."""
        blocked_positions = []
        for i in range(len(S)):
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    blocked_positions.append(i)
                    break
        
        if not blocked_positions:
            return S, counts, at_risk
        
        # Pick a random blocked position and try changing its symbol
        i = random.choice(blocked_positions)
        old_sym = S[i]
        # Try each other symbol with lightweight count update
        for new_sym in range(1, n + 1):
            if new_sym == old_sym:
                continue
            S[i] = new_sym
            counts = rebuild_counts(S)
            at_risk = set(p for p, c in counts.items() if c == 1)
            # Quick removal pass to see if this helps
            test_S = S[:]
            test_counts = dict(counts)
            test_at_risk = set(at_risk)
            removed = removal_pass(test_S, test_counts, test_at_risk, deadline)
            if removed:
                return S, counts, at_risk
            S[i] = old_sym
        
        # If no single change helped, try a swap with a nearby position
        if len(S) > 2:
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < len(S) - 1:
                neighbors.append(i + 1)
            if neighbors:
                j = random.choice(neighbors)
                S[i], S[j] = S[j], S[i]
                counts = rebuild_counts(S)
                at_risk = set(p for p, c in counts.items() if c == 1)
        
        return S, counts, at_risk

    # 6. Random swap perturbation
    def perturb_swap(S, counts, at_risk):
        i, j = random.sample(range(len(S)), 2)
        S[i], S[j] = S[j], S[i]
        counts = rebuild_counts(S)
        at_risk = set(p for p, c in counts.items() if c == 1)
        return S, counts, at_risk

    # 6b. Insertion perturbation: insert a random symbol at a random position
    def perturb_insert(S, counts, at_risk):
        pos = random.randint(0, len(S))
        sym = random.randint(1, n)
        S.insert(pos, sym)
        counts = rebuild_counts(S)
        at_risk = set(p for p, c in counts.items() if c == 1)
        return S, counts, at_risk

    # 7. Main loop
    stuck_rounds = 0
    max_stuck = 50
    while time.time() < deadline:
        if stuck_rounds % 2 == 0:
            removed = removal_pass(S, counts, at_risk, deadline)
        else:
            removed = removal_pass_bidirectional(S, counts, at_risk, deadline)
        
        if removed:
            if len(S) < best_len:
                best_S = S[:]
                best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            
            # Use targeted perturbation first, then random
            if stuck_rounds % 3 == 0:
                S, counts, at_risk = targeted_perturb(S, counts, at_risk)
            elif stuck_rounds % 5 == 0:
                S, counts, at_risk = perturb_insert(S, counts, at_risk)
            else:
                S, counts, at_risk = perturb_swap(S, counts, at_risk)

    # 8. Final polish
    if time.time() < deadline - 2:
        S2 = best_S[:]
        counts2 = rebuild_counts(S2)
        at_risk2 = set(p for p, c in counts2.items() if c == 1)
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            removal_pass(S2, counts2, at_risk2, deadline)
            removal_pass_bidirectional(S2, counts2, at_risk2, deadline)
        if len(S2) < len(best_S) and is_valid(S2):
            best_S = S2

    # 9. Validate
    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
