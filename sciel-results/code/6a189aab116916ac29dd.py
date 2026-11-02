import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    total_space = q ** n

    def tuple_to_int(t):
        res = 0
        for i, d in enumerate(t):
            res += d * powers[i]
        return res

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Precompute neighbors as lists (for iteration) and sets (for membership)
    neighbors_list = [None] * total_space
    neighbors_set = [None] * total_space
    for i in range(total_space):
        nb = [i]
        for pos in range(n):
            dv = (i // powers[pos]) % q
            for nv in range(q):
                if nv != dv:
                    nb.append(i + (nv - dv) * powers[pos])
        neighbors_list[i] = nb
        neighbors_set[i] = frozenset(nb)

    def product_init():
        if q == 3 and n == 7:
            tetracode = [[a, b, (a+b)%3, (a+2*b)%3] for a in range(3) for b in range(3)]
            rest = n - 4
            words = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=rest):
                    words.append(tuple_to_int(h + list(tail)))
            return words
        else:
            # Fallback: all-zero + greedy
            return [0]

    def greedy_init():
        uncovered = set(range(total_space))
        codewords = []
        while uncovered and time.time() < deadline - 40:
            best_w = None
            best_cov = 0
            for w in range(total_space):
                cov = len(neighbors_set[w] & uncovered)
                if cov > best_cov:
                    best_cov = cov
                    best_w = w
            codewords.append(best_w)
            uncovered -= neighbors_set[best_w]
        return codewords

    def local_search(codewords, time_limit):
        counts = [0] * total_space
        for w in codewords:
            for nb in neighbors_list[w]:
                counts[nb] += 1

        # Phase 1: Remove redundant
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            idxs = list(range(len(codewords)))
            random.shuffle(idxs)
            for idx in idxs:
                if idx >= len(codewords):
                    continue
                w = codewords[idx]
                ok = True
                for nb in neighbors_list[w]:
                    if counts[nb] <= 1:
                        ok = False
                        break
                if ok:
                    for nb in neighbors_list[w]:
                        counts[nb] -= 1
                    codewords.pop(idx)
                    changed = True

        # Phase 2: Swaps
        max_iters = 300
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
            if not codewords:
                break
            idx = random.randint(0, len(codewords) - 1)
            w = codewords[idx]
            uncovered_if_removed = set()
            for nb in neighbors_list[w]:
                if counts[nb] == 1:
                    uncovered_if_removed.add(nb)

            if not uncovered_if_removed:
                for nb in neighbors_list[w]:
                    counts[nb] -= 1
                codewords.pop(idx)
                # Cascade removals
                for i in range(len(codewords)):
                    wi = codewords[i]
                    rem = all(counts[nb] > 1 for nb in neighbors_list[wi])
                    if rem:
                        for nb in neighbors_list[wi]:
                            counts[nb] -= 1
                        codewords.pop(i)
                        break
                continue

            # Find best replacement
            best_cand = None
            best_score = -1
            best_exist = False

            # Check existing codewords (size reduction)
            for w2 in codewords:
                if w2 == w:
                    continue
                if uncovered_if_removed.issubset(neighbors_set[w2]):
                    best_cand = w2
                    best_exist = True
                    best_score = 10000
                    break

            if best_cand is None:
                # Search all space
                for w2 in range(total_space):
                    if w2 == w:
                        continue
                    if uncovered_if_removed.issubset(neighbors_set[w2]):
                        # Score: how many new words does w2 cover?
                        extra = sum(1 for nb in neighbors_list[w2] if counts[nb] == 0)
                        if extra > best_score:
                            best_score = extra
                            best_cand = w2
                            best_exist = False

            if best_cand is not None:
                for nb in neighbors_list[w]:
                    counts[nb] -= 1
                codewords.pop(idx)
                if not best_exist:
                    codewords.append(best_cand)
                    for nb in neighbors_list[best_cand]:
                        counts[nb] += 1

        # Final cleanup
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(codewords)):
                wi = codewords[i]
                if all(counts[nb] > 1 for nb in neighbors_list[wi]):
                    for nb in neighbors_list[wi]:
                        counts[nb] -= 1
                    codewords.pop(i)
                    changed = True
                    break

        return codewords

    best_size = total_space + 1
    best_ints = []

    # Try product code
    init = product_init()
    res = local_search(init[:], deadline - 10)
    if len(res) < best_size:
        best_size = len(res)
        best_ints = res[:]

    # Try greedy
    if time.time() < deadline - 40:
        init = greedy_init()
        res = local_search(init[:], deadline - 10)
        if len(res) < best_size:
            best_size = len(res)
            best_ints = res[:]

    # Perturbed restarts
    while time.time() < deadline - 10:
        base = product_init()
        random.shuffle(base)
        keep = int(len(base) * random.uniform(0.6, 0.95))
        perturbed = base[:keep]
        for _ in range(random.randint(0, 8)):
            perturbed.append(random.randint(0, total_space - 1))
        perturbed = list(set(perturbed))
        res = local_search(perturbed, deadline - 10)
        if len(res) < best_size:
            best_size = len(res)
            best_ints = res[:]

    return [int_to_tuple(v) for v in best_ints]
