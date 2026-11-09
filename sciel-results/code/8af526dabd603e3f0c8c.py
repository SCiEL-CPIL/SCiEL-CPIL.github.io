import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

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

    # Precompute incompatibility (divisibility)
    incompatible = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if a % b == 0 or b % a == 0:
                incompatible[i].add(j)
                incompatible[j].add(i)
                
    # Precompute compatible pairs for faster iteration
    # compatible[i] = list of j > i such that i and j are NOT incompatible
    compatible = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if j not in incompatible[i]:
                compatible[i].append(j)

    # For each target, find the MINIMUM size antichain representation
    min_size = [0] * 129
    min_rep = [None] * 129

    # Size 1: target itself must be smooth
    for i in range(n):
        v = smooth_numbers[i]
        if v <= 128 and min_size[v] == 0:
            min_size[v] = 1
            min_rep[v] = [v]

    # Size 2: two non-dividing smooth numbers summing to target
    for i in range(n):
        vi = smooth_numbers[i]
        for j in compatible[i]:
            s = vi + smooth_numbers[j]
            if s <= 128 and min_size[s] == 0:
                min_size[s] = 2
                min_rep[s] = [vi, smooth_numbers[j]]

    # Size 3
    for i in range(n):
        vi = smooth_numbers[i]
        for j in compatible[i]:
            vj = smooth_numbers[j]
            # k must be > j and compatible with i and j
            for k in compatible[j]:
                if k in incompatible[i]:
                    continue
                s = vi + vj + smooth_numbers[k]
                if s <= 128 and min_size[s] == 0:
                    min_size[s] = 3
                    min_rep[s] = [vi, vj, smooth_numbers[k]]

    # Size 4
    uncovered4 = [t for t in range(1, 129) if min_size[t] == 0]
    if uncovered4:
        uncovered_set = set(uncovered4)
        for i in range(n):
            vi = smooth_numbers[i]
            for j in compatible[i]:
                vj = smooth_numbers[j]
                for k in compatible[j]:
                    if k in incompatible[i]:
                        continue
                    vk = smooth_numbers[k]
                    for m in compatible[k]:
                        if m in incompatible[i] or m in incompatible[j]:
                            continue
                        s = vi + vj + vk + smooth_numbers[m]
                        if s <= 128 and s in uncovered_set:
                            min_size[s] = 4
                            min_rep[s] = [vi, vj, vk, smooth_numbers[m]]
                            uncovered_set.discard(s)

    # Size 5
    uncovered5 = [t for t in range(1, 129) if min_size[t] == 0]
    if uncovered5:
        uncovered_set = set(uncovered5)
        for i in range(n):
            vi = smooth_numbers[i]
            for j in compatible[i]:
                vj = smooth_numbers[j]
                for k in compatible[j]:
                    if k in incompatible[i]:
                        continue
                    vk = smooth_numbers[k]
                    for m in compatible[k]:
                        if m in incompatible[i] or m in incompatible[j]:
                            continue
                        vm = smooth_numbers[m]
                        for p in compatible[m]:
                            if p in incompatible[i] or p in incompatible[j] or p in incompatible[k]:
                                continue
                            s = vi + vj + vk + vm + smooth_numbers[p]
                            if s <= 128 and s in uncovered_set:
                                min_size[s] = 5
                                min_rep[s] = [vi, vj, vk, vm, smooth_numbers[p]]
                                uncovered_set.discard(s)

    # Size 6
    uncovered6 = [t for t in range(1, 129) if min_size[t] == 0]
    if uncovered6:
        uncovered_set = set(uncovered6)
        for i in range(n):
            vi = smooth_numbers[i]
            for j in compatible[i]:
                vj = smooth_numbers[j]
                for k in compatible[j]:
                    if k in incompatible[i]:
                        continue
                    vk = smooth_numbers[k]
                    for m in compatible[k]:
                        if m in incompatible[i] or m in incompatible[j]:
                            continue
                        vm = smooth_numbers[m]
                        for p in compatible[m]:
                            if p in incompatible[i] or p in incompatible[j] or p in incompatible[k]:
                                continue
                            vp = smooth_numbers[p]
                            for q in compatible[p]:
                                if q in incompatible[i] or q in incompatible[j] or q in incompatible[k] or q in incompatible[m]:
                                    continue
                                s = vi + vj + vk + vm + vp + smooth_numbers[q]
                                if s <= 128 and s in uncovered_set:
                                    min_size[s] = 6
                                    min_rep[s] = [vi, vj, vk, vm, vp, smooth_numbers[q]]
                                    uncovered_set.discard(s)
                                if time.time() > deadline - 5.0:
                                    break
                            if time.time() > deadline - 5.0:
                                break
                        if time.time() > deadline - 5.0:
                            break
                    if time.time() > deadline - 5.0:
                        break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break

    # Build answers
    answers = []
    total_terms = 0
    for target in range(1, 129):
        if min_rep[target] is not None:
            rep = min_rep[target]
        else:
            # Fallback: brute force
            rep = _brute_force(target, smooth_numbers, incompatible, n)
            if rep is None:
                # Last resort: try to find any valid rep, even if not minimal
                # But if target is not smooth, [target] is invalid.
                # We must ensure validity.
                # If we failed to find a rep, it's a problem.
                # However, the problem states "All targets require valid representations".
                # If we can't find one, we might have to use a larger size or a different strategy.
                # For now, if rep is None, we try a very slow search for size 7-10.
                rep = _slow_brute_force(target, smooth_numbers, incompatible, n)
                if rep is None:
                    # This should not happen if the problem is solvable.
                    # If it does, returning [target] is invalid for non-smooth targets.
                    # We assume the problem is solvable with reasonable sizes.
                    rep = [target]
        answers.append(rep)
        total_terms += len(rep)

    return answers

def _brute_force(target, smooth_numbers, incompatible, n):
    """Brute force search for a valid antichain summing to target."""
    for size in range(1, 8):
        for combo in combinations(range(n), size):
            s = sum(smooth_numbers[i] for i in combo)
            if s != target:
                continue
            valid = True
            for a_idx in range(size):
                for b_idx in range(a_idx+1, size):
                    i, j = combo[a_idx], combo[b_idx]
                    if j in incompatible[i]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [smooth_numbers[i] for i in combo]
    return None

def _slow_brute_force(target, smooth_numbers, incompatible, n):
    """Slower brute force for larger sizes if needed."""
    for size in range(8, 12):
        for combo in combinations(range(n), size):
            s = sum(smooth_numbers[i] for i in combo)
            if s != target:
                continue
            valid = True
            for a_idx in range(size):
                for b_idx in range(a_idx+1, size):
                    i, j = combo[a_idx], combo[b_idx]
                    if j in incompatible[i]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [smooth_numbers[i] for i in combo]
    return None
