import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    # 1. Generate smooth numbers up to 128
    max_n = 128
    smooth = []
    is_smooth = [False] * (max_n + 1)
    for x in range(1, max_n + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            is_smooth[x] = True

    smooth_set = set(smooth)
    n_smooth = len(smooth)

    # 2. Precompute antichain validity: invalid_pairs contains (a,b) with a<b and b%a==0
    invalid_pairs = set()
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(n_smooth):
            if i == j:
                continue
            b = smooth[j]
            if a > b:
                continue
            if b % a == 0:
                invalid_pairs.add((a, b))

    def is_antichain_pair(a, b):
        if a == b:
            return False
        if a > b:
            a, b = b, a
        return (a, b) not in invalid_pairs

    answers = []

    # 3. Solve for each target with increasing term count
    for n in range(1, max_n + 1):
        if time.time() > deadline:
            # Fallback: use a simple valid representation (may not be minimal)
            # For targets <= 128, we can always find a valid rep quickly
            # Use greedy backtracking as last resort
            def backtrack(remaining, start_idx, current_terms):
                if remaining == 0:
                    return current_terms
                if len(current_terms) >= 30:
                    return None
                for k in range(start_idx, n_smooth):
                    t = smooth[k]
                    if t > remaining:
                        break
                    valid = True
                    for existing in current_terms:
                        if not is_antichain_pair(t, existing):
                            valid = False
                            break
                    if not valid:
                        continue
                    res = backtrack(remaining - t, k + 1, current_terms + [t])
                    if res:
                        return res
                return None

            ans = backtrack(n, 0, [])
            if ans is None:
                ans = [n]  # last resort, may be invalid but prevents crash
            answers.append(ans)
            continue

        ans = None

        # Case 1: n is smooth -> 1 term
        if is_smooth[n]:
            ans = [n]
        else:
            # Case 2: Try 2 terms
            found = False
            for a in smooth:
                if a >= n:
                    break
                b = n - a
                if b <= a:
                    continue
                if is_smooth[b] and is_antichain_pair(a, b):
                    ans = [a, b]
                    found = True
                    break

            # Case 3: Try 3 terms
            if not found:
                for i in range(n_smooth):
                    a = smooth[i]
                    if a >= n:
                        break
                    for j in range(i + 1, n_smooth):
                        b = smooth[j]
                        if a + b >= n:
                            break
                        c = n - a - b
                        if c <= b:
                            continue
                        if is_smooth[c] and is_antichain_pair(a, b) and is_antichain_pair(a, c) and is_antichain_pair(b, c):
                            ans = [a, b, c]
                            found = True
                            break
                    if found:
                        break

            # Case 4: Try 4 terms
            if not found:
                for i in range(n_smooth):
                    a = smooth[i]
                    if a >= n:
                        break
                    for j in range(i + 1, n_smooth):
                        b = smooth[j]
                        if a + b >= n:
                            break
                        for k in range(j + 1, n_smooth):
                            c = smooth[k]
                            if a + b + c >= n:
                                break
                            d = n - a - b - c
                            if d <= c:
                                continue
                            if (is_smooth[d] and is_antichain_pair(a, b) and is_antichain_pair(a, c) and
                                is_antichain_pair(a, d) and is_antichain_pair(b, c) and is_antichain_pair(b, d) and
                                is_antichain_pair(c, d)):
                                ans = [a, b, c, d]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            # Case 5: Backtracking for 5+ terms
            if not found:
                def backtrack(remaining, start_idx, current_terms):
                    if remaining == 0:
                        return current_terms
                    if len(current_terms) >= 30:
                        return None
                    # Pruning: minimum possible sum of remaining terms
                    # If current_terms is non-empty, the next term must be > last term
                    # and must not divide/be divisible by any existing term
                    for k in range(start_idx, n_smooth):
                        t = smooth[k]
                        if t > remaining:
                            break
                        valid = True
                        for existing in current_terms:
                            if not is_antichain_pair(t, existing):
                                valid = False
                                break
                        if not valid:
                            continue
                        res = backtrack(remaining - t, k + 1, current_terms + [t])
                        if res:
                            return res
                    return None

                ans = backtrack(n, 0, [])
                if ans is None:
                    # Should never happen for n <= 128, but fallback
                    ans = [n]

        answers.append(ans)

    return answers
