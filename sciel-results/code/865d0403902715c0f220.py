import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 80.0


def _factorize(n, primes):
    """Return exponent vector of n with respect to primes list."""
    vec = []
    for p in primes:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        vec.append(e)
    return tuple(vec)


def construct():
    if time.time() > _DEADLINE:
        return [2]

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    nprimes = len(primes)

    all_vecs = {}
    for n in range(2, 41):
        all_vecs[n] = _factorize(n, primes)

    prime_vecs = [all_vecs[p] for p in primes]
    prods = {tuple([0] * nprimes)}
    for v in prime_vecs:
        new_prods = set(prods)
        for pv in prods:
            new_pv = tuple(pv[i] + v[i] for i in range(nprimes))
            new_prods.add(new_pv)
        prods = new_prods

    # Precompute difference set D = {p1 - p2 : p1, p2 in prods}
    # This enables O(1) conflict checks against the base prime set
    prod_list = list(prods)
    diff_set = set()
    for p1 in prod_list:
        for p2 in prod_list:
            diff_set.add(tuple(p1[i] - p2[i] for i in range(nprimes)))

    candidates = [n for n in range(2, 41) if n not in primes]
    candidate_vecs = [all_vecs[n] for n in candidates]

    # Filter: candidate vector must not be in diff_set
    valid_idx = [i for i in range(len(candidates)) if candidate_vecs[i] not in diff_set]
    valid_idx.sort(key=lambda i: (sum(candidate_vecs[i]), max(candidate_vecs[i]), candidates[i]))

    best_count = nprimes
    best_selection = list(primes)

    def _check_valid_addition(selected_vecs, new_vec):
        """Check if adding new_vec to selected_vecs keeps all subset products distinct."""
        all_v = selected_vecs + [new_vec]
        sums = [tuple([0] * nprimes)]
        for v in all_v:
            new_sums = [tuple(s[i] + v[i] for i in range(nprimes)) for s in sums]
            sums.extend(new_sums)
        if len(set(sums)) != len(sums):
            return False
        for i in range(len(sums)):
            for j in range(i + 1, len(sums)):
                diff = tuple(sums[i][d] - sums[j][d] for d in range(nprimes))
                if diff in diff_set:
                    return False
        return True

    def _search(start_idx, selected_vecs, selected_nums, remaining):
        nonlocal best_count, best_selection
        if time.time() > _DEADLINE - 5:
            return
        cur_count = nprimes + len(selected_nums)
        if cur_count > best_count:
            best_count = cur_count
            best_selection = list(primes) + list(selected_nums)

        if remaining == 0:
            return
        if cur_count + remaining <= best_count:
            return

        for i in range(start_idx, len(valid_idx)):
            if time.time() > _DEADLINE - 5:
                return
            v = candidate_vecs[valid_idx[i]]
            n = candidates[valid_idx[i]]
            if _check_valid_addition(selected_vecs, v):
                selected_vecs.append(v)
                selected_nums.append(n)
                _search(i + 1, selected_vecs, selected_nums, remaining - 1)
                selected_vecs.pop()
                selected_nums.pop()

    for target in range(min(6, len(valid_idx)), 0, -1):
        if time.time() > _DEADLINE - 10:
            break
        _search(0, [], [], target)
        if best_count >= nprimes + target:
            break

    result = best_selection[:best_count]
    return [int(x) for x in result]
