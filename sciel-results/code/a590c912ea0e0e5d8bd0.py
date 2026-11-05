"""Improved program: bitmask-based maximum clique search with multi-start hubs."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s
    N = 512

    # Precompute compatibility: compat[a] is bitmask of b in 1..N with lcm(a,b) <= N
    # Index: number x maps to bit (x-1)
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a * b // gcd(a, b) <= N:
                mask |= (1 << (b - 1))
        compat[a] = mask

    def get_divisors(x):
        divs = []
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                divs.append(d)
                if d != x // d:
                    divs.append(x // d)
        return divs

    def count_divisors(x):
        return len(get_divisors(x))

    # Score each number by its divisor count (proxy for hub quality)
    div_counts = [(count_divisors(k), k) for k in range(1, N + 1)]
    div_counts.sort(key=lambda x: (-x[0], x[1]))

    # Greedy expansion: given a starting set (as bitmask), add compatible numbers
    def expand(start_mask):
        s = start_mask
        # Numbers already in set
        in_set = [False] * (N + 1)
        for x in range(1, N + 1):
            if (s >> (x - 1)) & 1:
                in_set[x] = True

        # Find all numbers compatible with entire set
        candidates_mask = (1 << N) - 1
        for x in range(1, N + 1):
            if in_set[x]:
                candidates_mask &= compat[x]

        # Greedy: repeatedly add the number with most connections among remaining candidates
        changed = True
        while changed:
            changed = False
            # Among candidates not in set, find best
            best_x = 0
            best_score = -1
            for x in range(1, N + 1):
                if in_set[x]:
                    continue
                if not ((candidates_mask >> (x - 1)) & 1):
                    continue
                # Score: number of other candidates it's compatible with
                score = bin(candidates_mask & compat[x]).count('1')
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x == 0:
                break
            s |= (1 << (best_x - 1))
            in_set[best_x] = True
            candidates_mask &= compat[best_x]
            changed = True
        return s

    # Local search: 1-swap and 2-swap improvements
    def local_search(s):
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            in_set = [False] * (N + 1)
            for x in range(1, N + 1):
                if (s >> (x - 1)) & 1:
                    in_set[x] = True

            # Compute candidates compatible with all in set
            candidates_mask = (1 << N) - 1
            for x in range(1, N + 1):
                if in_set[x]:
                    candidates_mask &= compat[x]

            # Try 1-swap: remove one element, add a better one
            for rem in range(1, N + 1):
                if not in_set[rem]:
                    continue
                s_without = s & ~(1 << (rem - 1))
                # Candidates compatible with s_without
                cm = (1 << N) - 1
                for x in range(1, N + 1):
                    if (s_without >> (x - 1)) & 1:
                        cm &= compat[x]
                # Find best addable
                best_add = 0
                best_score = -1
                for add in range(1, N + 1):
                    if (s >> (add - 1)) & 1:
                        continue
                    if not ((cm >> (add - 1)) & 1):
                        continue
                    score = bin(cm & compat[add]).count('1')
                    if score > best_score:
                        best_score = score
                        best_add = add
                if best_add != 0:
                    new_s = s_without | (1 << (best_add - 1))
                    if bin(new_s).count('1') > bin(s).count('1'):
                        s = new_s
                        improved = True
                        break
                if time.time() > deadline - 3:
                    break

            # Try 2-add: add two compatible elements at once
            if not improved:
                cm = (1 << N) - 1
                for x in range(1, N + 1):
                    if in_set[x]:
                        cm &= compat[x]
                addable = [x for x in range(1, N + 1)
                           if not in_set[x] and ((cm >> (x - 1)) & 1)]
                if len(addable) >= 2:
                    for i in range(len(addable)):
                        if time.time() > deadline - 3:
                            break
                        for j in range(i + 1, len(addable)):
                            a, b = addable[i], addable[j]
                            if (compat[a] >> (b - 1)) & 1:
                                new_s = s | (1 << (a - 1)) | (1 << (b - 1))
                                if bin(new_s).count('1') > bin(s).count('1'):
                                    s = new_s
                                    improved = True
                                    break
                        if improved:
                            break
        return s

    best_mask = 0
    best_size = 0

    # Multi-start: try each top hub's divisor set
    for cnt, hub in div_counts[:30]:
        if time.time() > deadline - 5:
            break
        divs = get_divisors(hub)
        start_mask = 0
        for d in divs:
            if d <= N:
                start_mask |= (1 << (d - 1))
        expanded = expand(start_mask)
        expanded = local_search(expanded)
        sz = bin(expanded).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = expanded

    # Also try single-element starts for high-degree nodes
    degree = [(bin(compat[x]).count('1'), x) for x in range(1, N + 1)]
    degree.sort(key=lambda x: (-x[0], x[1]))
    for deg, x in degree[:20]:
        if time.time() > deadline - 5:
            break
        start_mask = 1 << (x - 1)
        expanded = expand(start_mask)
        expanded = local_search(expanded)
        sz = bin(expanded).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = expanded

    # Convert to list
    result = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            result.append(x)

    if not result:
        result = [1]

    return result
# EVOLVE-BLOCK-END
