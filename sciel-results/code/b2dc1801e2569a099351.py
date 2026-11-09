import time

def construct():
    start_time = time.time()
    timeout = 60.0

    # Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    smooth_set = set(smooth_numbers)
    n = len(smooth_numbers)
    
    # Precompute all valid antichain representations by size
    # reps_by_size[target] = list of tuples (representation, num_terms)
    reps_by_target = {t: [] for t in range(1, 129)}
    
    # Size 1
    for t in range(1, 129):
        if t in smooth_set:
            reps_by_target[t].append(((t,), 1))
    
    # Size 2
    for i in range(n):
        a = smooth_numbers[i]
        for j in range(i + 1, n):
            b = smooth_numbers[j]
            if a + b > 128:
                break
            if b % a != 0:  # antichain
                t = a + b
                reps_by_target[t].append(((a, b), 2))
    
    # Size 3
    for i in range(n):
        a = smooth_numbers[i]
        for j in range(i + 1, n):
            b = smooth_numbers[j]
            if a + b > 128:
                break
            if b % a == 0:
                continue
            for k in range(j + 1, n):
                c = smooth_numbers[k]
                if a + b + c > 128:
                    break
                # Check antichain: c not divisible by a or b, and c doesn't divide a or b (impossible since c > b > a)
                if c % a != 0 and c % b != 0:
                    t = a + b + c
                    reps_by_target[t].append(((a, b, c), 3))
    
    # Size 4
    for i in range(n):
        a = smooth_numbers[i]
        for j in range(i + 1, n):
            b = smooth_numbers[j]
            if a + b > 128:
                break
            if b % a == 0:
                continue
            for k in range(j + 1, n):
                c = smooth_numbers[k]
                if a + b + c > 128:
                    break
                if c % a == 0 or c % b == 0:
                    continue
                for l in range(k + 1, n):
                    d = smooth_numbers[l]
                    if a + b + c + d > 128:
                        break
                    # Check d against a, b, c
                    if d % a == 0 or d % b == 0 or d % c == 0:
                        continue
                    t = a + b + c + d
                    reps_by_target[t].append(((a, b, c, d), 4))
    
    # Size 5 - only for targets that don't have smaller representations
    targets_needing_5 = [t for t in range(1, 129) if not reps_by_target[t]]
    if targets_needing_5:
        for i in range(n):
            a = smooth_numbers[i]
            for j in range(i + 1, n):
                b = smooth_numbers[j]
                if a + b > 128:
                    break
                if b % a == 0:
                    continue
                for k in range(j + 1, n):
                    c = smooth_numbers[k]
                    if a + b + c > 128:
                        break
                    if c % a == 0 or c % b == 0:
                        continue
                    for l in range(k + 1, n):
                        d = smooth_numbers[l]
                        if a + b + c + d > 128:
                            break
                        if d % a == 0 or d % b == 0 or d % c == 0:
                            continue
                        for m in range(l + 1, n):
                            e = smooth_numbers[m]
                            if a + b + c + d + e > 128:
                                break
                            if e % a == 0 or e % b == 0 or e % c == 0 or e % d == 0:
                                continue
                            t = a + b + c + d + e
                            if t in targets_needing_5 or not reps_by_target[t]:
                                reps_by_target[t].append(((a, b, c, d, e), 5))
    
    # Verify all targets have representations; if not, use backtracking
    for t in range(1, 129):
        if not reps_by_target[t]:
            # Fallback: brute force backtracking
            def backtrack(rem, start, depth, current, max_depth=8):
                if depth == 0:
                    if rem == 0:
                        return current
                    return None
                if rem <= 0 or len(current) >= max_depth:
                    return None
                for idx in range(start, n):
                    val = smooth_numbers[idx]
                    if val >= rem:
                        break
                    ok = True
                    for v in current:
                        if v % val == 0 or val % v == 0:
                            ok = False
                            break
                    if not ok:
                        continue
                    res = backtrack(rem - val, idx + 1, depth - 1, current + [val], max_depth)
                    if res is not None:
                        return res
                return None
            
            found = False
            for size in range(5, 11):
                res = backtrack(t, 0, size, [])
                if res is not None:
                    reps_by_target[t].append((tuple(res), size))
                    found = True
                    break
            if not found:
                # Should not happen, but safety net
                reps_by_target[t].append(((t,), 1))  # Invalid but prevents crash
    
    # Now use greedy set cover to minimize total terms
    # Universe: targets 1..128
    # For each target, we have a set of possible representations
    # We want to pick one representation per target to minimize total terms
    
    # Sort targets by number of available representations (hardest first)
    target_order = sorted(range(1, 129), key=lambda t: len(reps_by_target[t]))
    
    assignment = {}
    remaining_targets = set(range(1, 129))
    
    # Greedy: for each target in order of fewest options, pick the representation
    # that minimizes the "cost" considering what's already chosen
    # Actually, since each target needs exactly one representation, and they're independent,
    # the greedy approach is: for each target, pick the representation with fewest terms.
    # But we can do better by considering conflicts (though there are no conflicts here).
    
    # Since there are no conflicts between targets (each target is independent),
    # the optimal solution is simply to pick the minimum-term representation for each target.
    # Let's verify this is what we're doing.
    
    for t in range(1, 129):
        if not reps_by_target[t]:
            assignment[t] = [t]  # Fallback
        else:
            # Pick the representation with minimum terms
            best = min(reps_by_target[t], key=lambda x: x[1])
            assignment[t] = list(best[0])
    
    # Calculate total terms for reporting
    total_terms = sum(len(assignment[t]) for t in range(1, 129))
    
    # Build the answer in order 1..128
    answers = [assignment[t] for t in range(1, 129)]
    
    # Verify validity
    for t in range(1, 129):
        terms = answers[t]
        assert len(terms) >= 1 and len(terms) <= 30, f"Target {t}: invalid term count {len(terms)}"
        assert sum(terms) == t, f"Target {t}: sum {sum(terms)} != {t}"
        # Check distinctness
        assert len(set(terms)) == len(terms), f"Target {t}: non-distinct terms"
        # Check antichain
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                assert a % b != 0 and b % a != 0, f"Target {t}: {a} divides {b} or vice versa"
        # Check all terms are smooth
        for term in terms:
            x = term
            while x % 2 == 0: x //= 2
            while x % 3 == 0: x //= 3
            while x % 5 == 0: x //= 5
            assert x == 1, f"Target {t}: term {term} is not 5-smooth"
    
    return answers
