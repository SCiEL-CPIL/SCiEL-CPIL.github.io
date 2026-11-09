"""Combined: primes-guaranteed base + multi-strategy greedy with deadline guard."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0

    def _block_products(values):
        prods = set()
        n = len(values)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= values[j]
                prods.add(p)
        return prods

    def _valid(values):
        n = len(values)
        expected = n * (n + 1) // 2
        return len(_block_products(values)) == expected

    def _greedy(cands):
        vals, prods = [], set()
        used = set()
        for x in cands:
            if time.time() > deadline:
                break
            if x in used:
                continue
            tails, p = [x], x
            for a in reversed(vals):
                p *= a
                tails.append(p)
            if not prods.intersection(tails):
                vals.append(x)
                prods.update(tails)
                used.add(x)
        return vals

    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    best = list(primes)

    for cands in [list(range(2,129)), list(range(3,128,2)), primes+[x for x in range(4,129) if x not in set(primes)]]:
        if time.time() > deadline - 3:
            break
        alt = _greedy(cands)
        if len(alt) > len(best):
            best = alt

    if not _valid(best):
        best = list(primes)
    return best
# EVOLVE-BLOCK-END
