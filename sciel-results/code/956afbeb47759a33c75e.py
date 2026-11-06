import time
import random

def construct():
    start = time.time()
    deadline = start + 80.0
    n_max = 4096

    def build_sums(S):
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S[i] + S[j] + S[l])
        return sums

    def can_add(x, S, sums):
        if 3 * x in sums:
            return False
        for s in S:
            if 2 * x + s in sums:
                return False
        k = len(S)
        for i in range(k):
            s = S[i]
            for j in range(i, k):
                t = S[j]
                if x + s + t in sums:
                    return False
        return True

    def add_x(x, S, sums):
        S.append(x)
        sums.add(3 * x)
        for s in S[:-1]:
            sums.add(2 * x + s)
        k = len(S)
        for i in range(k - 1):
            s = S[i]
            for j in range(i, k - 1):
                t = S[j]
                sums.add(x + s + t)

    def greedy_forward(start_list, deadline):
        S = list(start_list)
        sums = build_sums(S)
        existing = set(S)
        for x in range(1, n_max + 1):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add(x, S, sums):
                add_x(x, S, sums)
                existing.add(x)
        return S

    def greedy_backward(start_list, deadline):
        S = list(start_list)
        sums = build_sums(S)
        existing = set(S)
        for x in range(n_max, 0, -1):
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add(x, S, sums):
                add_x(x, S, sums)
                existing.add(x)
        return S

    def density_swap(S, deadline, max_rounds=40):
        """Try to increase density: remove a large element, add a smaller one."""
        S = sorted(S)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            k = len(S)
            # Try removing each element (larger elements first - they waste space)
            for idx in range(k - 1, -1, -1):
                if time.time() > deadline:
                    break
                orig = S[idx]
                S_no = S[:idx] + S[idx + 1:]
                sums_no = build_sums(S_no)
                existing_no = set(S_no)
                # Find the smallest element we can add
                best = None
                for new_x in range(1, n_max + 1):
                    if new_x in existing_no:
                        continue
                    if can_add(new_x, S_no, sums_no):
                        best = new_x
                        break
                if best is not None and best < orig:
                    # Accept: replace orig with best (smaller -> denser)
                    S[idx] = best
                    S.sort()
                    return True
        return False

    def local_search(S, deadline, max_rounds=60):
        """Iterative density swaps to pack the set denser."""
        S = sorted(S)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            if not density_swap(S, deadline, max_rounds=1):
                break
        return S

    # --- Strong seed: powers of 3 (perfect B_3[1] set) ---
    pow3 = []
    v = 1
    while v <= n_max:
        pow3.append(v)
        v *= 3

    # --- Multiple strategies, keep the best (largest) set ---
    best = []
    best_len = 0

    def consider(S):
        nonlocal best, best_len
        if len(S) > best_len:
            best = sorted(S)
            best_len = len(S)

    # 1. Forward greedy from powers of 3
    consider(greedy_forward(pow3, deadline))
    # 2. Backward greedy from powers of 3
    consider(greedy_backward(pow3, deadline))
    # 3. Forward greedy from empty
    consider(greedy_forward([], deadline))
    # 4. Backward greedy from empty
    consider(greedy_backward([], deadline))

    # 5. Local search (density swaps) on the best so far
    if time.time() < deadline - 3:
        consider(local_search(best, deadline))

    # 6. Randomized restarts with varied skeletons
    random.seed(12345)
    for attempt in range(40):
        if time.time() > deadline - 2:
            break
        # Mix: powers of 3 plus a few random small elements
        skel = list(pow3)
        extra = random.randint(0, 4)
        for _ in range(extra):
            cand = random.randint(1, 100)
            if cand not in skel:
                skel.append(cand)
        skel.sort()
        # Verify skeleton validity
        valid = True
        sk_sums = set()
        k = len(skel)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s_val = skel[i] + skel[j] + skel[l]
                    if s_val in sk_sums:
                        valid = False
                        break
                    sk_sums.add(s_val)
            if not valid:
                break
        if not valid:
            continue
        direction = random.choice([greedy_forward, greedy_backward])
        res = direction(skel, deadline)
        consider(res)
        # Polish with local search if promising
        if len(res) >= best_len - 1 and time.time() < deadline - 2:
            consider(local_search(res, deadline))

    return best[:80]
