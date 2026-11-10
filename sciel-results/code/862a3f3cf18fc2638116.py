import time
import random

def construct():
    deadline = time.time() + 85
    n = 256

    # Precompute conflicts using bitmasks for speed
    # conflict_mask[i] is a bitmask where bit j is set if (i*j)%(i+j)==0
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    def is_valid(s):
        s_mask = 0
        for a in s:
            if conflict_mask[a] & s_mask:
                return False
            s_mask |= (1 << a)
        return True

    def greedy_fill(s, candidates, order):
        """Add elements from candidates (filtered by order) to s if valid."""
        s = set(s)
        s_mask = 0
        for a in s:
            s_mask |= (1 << a)
        for e in order:
            if e in s:
                continue
            if not (conflict_mask[e] & s_mask):
                s.add(e)
                s_mask |= (1 << e)
        return s

    def local_search(s, max_iters=1000):
        """Local search with add, 1-swap, and 2-swap moves."""
        s = set(s)
        best_size = len(s)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Build mask of current set
            s_mask = 0
            for a in s:
                s_mask |= (1 << a)

            # 1. Try to add any non-member
            added = False
            for b in range(1, n + 1):
                if (1 << b) & s_mask:
                    continue
                if not (conflict_mask[b] & s_mask):
                    s.add(b)
                    s_mask |= (1 << b)
                    best_size = len(s)
                    improved = True
                    added = True
                    break
            if added:
                continue

            # 2. Try 1-swap: remove a, add b
            # b must conflict with exactly one element in S
            for b in range(1, n + 1):
                if (1 << b) & s_mask:
                    continue
                conflicts_with_b = conflict_mask[b] & s_mask
                if not conflicts_with_b:
                    continue
                if bin(conflicts_with_b).count('1') == 1:
                    a = conflicts_with_b.bit_length() - 1
                    s.discard(a)
                    s.add(b)
                    s_mask = s_mask & ~(1 << a) | (1 << b)
                    best_size = len(s)
                    improved = True
                    break
            if improved:
                continue

            # 3. Try 2-swap: remove {a1, a2}, add {b1, b2}
            # This is more expensive, so do it less frequently
            if _ % 5 == 4:
                for b1 in range(1, n + 1):
                    if (1 << b1) & s_mask:
                        continue
                    conf_b1 = conflict_mask[b1] & s_mask
                    if not conf_b1 or bin(conf_b1).count('1') > 2:
                        continue
                    for b2 in range(b1 + 1, n + 1):
                        if (1 << b2) & s_mask:
                            continue
                        # Check if b1 and b2 conflict with each other
                        if (1 << b2) & conflict_mask[b1]:
                            continue
                        conf_b2 = conflict_mask[b2] & s_mask
                        if not conf_b2 or bin(conf_b2).count('1') > 2:
                            continue
                        # The combined conflicts of b1 and b2 with S
                        combined_conf = conf_b1 | conf_b2
                        if bin(combined_conf).count('1') > 2:
                            continue
                        # Try removing the conflicting elements
                        to_remove = []
                        temp = combined_conf
                        while temp:
                            to_remove.append(temp.bit_length() - 1)
                            temp &= temp - 1
                        new_s = s - set(to_remove) | {b1, b2}
                        if len(new_s) > len(s):
                            # Verify validity
                            if is_valid(new_s):
                                s = new_s
                                s_mask = 0
                                for a in s:
                                    s_mask |= (1 << a)
                                best_size = len(s)
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break

        return s, best_size

    best = []
    best_size = 0

    # Phase 1: Initial Construction
    odds = set(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))

    # Sort evens by number of conflicts with odds
    odds_mask = 0
    for o in odds:
        odds_mask |= (1 << o)

    def conflict_count(e):
        return bin(conflict_mask[e] & odds_mask).count('1')

    evens_sorted = sorted(evens, key=conflict_count)

    s = set(odds)
    s_mask = 0
    for o in odds:
        s_mask |= (1 << o)
    for e in evens_sorted:
        if not (conflict_mask[e] & s_mask):
            s.add(e)
            s_mask |= (1 << e)

    s, sz = local_search(s, max_iters=2000)
    if sz > best_size:
        best = list(s)
        best_size = sz

    # Phase 2: Restart with Randomized Kick
    all_elements = list(range(1, n + 1))

    for trial in range(50):
        if time.time() > deadline - 5:
            break

        # Start from best and kick
        s = set(best)

        # Kick: remove K random elements
        K = random.randint(5, 15)
        to_remove = random.sample(list(s), min(K, len(s)))
        for r in to_remove:
            s.discard(r)

        # Build mask for greedy re-add
        s_mask = 0
        for a in s:
            s_mask |= (1 << a)

        # Greedily re-add from all elements not in s, in random order
        remaining = [e for e in all_elements if e not in s]
        random.shuffle(remaining)

        for e in remaining:
            if not (conflict_mask[e] & s_mask):
                s.add(e)
                s_mask |= (1 << e)

        # Local search
        s, sz = local_search(s, max_iters=1000)

        if sz > best_size:
            best = list(s)
            best_size = sz

    return best
