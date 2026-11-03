"""Covering code K_q(n, radius): greedy init + randomized local search with deadline guard."""
import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _all_neighbors(w, q, n, r):
    """Return the set of words within Hamming distance r of w."""
    out = {w}
    for rad in range(1, r + 1):
        for idx in itertools.combinations(range(n), rad):
            for delta in itertools.product(range(1, q), repeat=rad):
                v = list(w)
                for i, d in zip(idx, delta):
                    v[i] = (v[i] + d) % q
                out.add(tuple(v))
    return out


def _initial_code(q, n, r):
    """Direct sum of a radius-1 perfect ternary code (tetracode) on 4 coords with all tails."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    if q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    else:
        # fallback: all words on 4 coords (still valid, just larger)
        tetracode = [list(w) for w in itertools.product(range(q), repeat=4)]
    heads = [tuple(h) for h in tetracode]
    code = [h + t for h in heads for t in itertools.product(range(q), repeat=n - 4)]
    return code


def construct(q, n, radius):
    random.seed(12345)
    r = radius

    code = _initial_code(q, n, r)
    best = list(code)
    best_len = len(best)

    # Precompute neighborhoods for the current code (recomputed when code changes)
    def build_neighs(code):
        return [_all_neighbors(tuple(c), q, n, r) for c in code]

    neighs = build_neighs(code)
    covered = set()
    for s in neighs:
        covered |= s

    full_space = q ** n

    def is_full():
        return len(covered) == full_space

    # Greedy removal pass: drop any codeword whose removal keeps full coverage.
    def greedy_remove(code, neighs):
        covered = set()
        for s in neighs:
            covered |= s
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                if not is_full():
                    break
                # try removing i
                # check if covered remains full without neighs[i]
                # recompute covered without i
                new_cov = set()
                for j, s in enumerate(neighs):
                    if j != i:
                        new_cov |= s
                if len(new_cov) == full_space:
                    # remove
                    code.pop(i)
                    neighs.pop(i)
                    covered = new_cov
                    changed = True
                    break
        return code, neighs

    code, neighs = greedy_remove(code, neighs)
    best = list(code)
    best_len = len(best)

    # Randomized local search: remove random codeword if still full, or swap.
    def try_remove(i, code, neighs, covered):
        new_cov = set()
        for j, s in enumerate(neighs):
            if j != i:
                new_cov |= s
        if len(new_cov) == full_space:
            code.pop(i)
            neighs.pop(i)
            return new_cov, True
        return covered, False

    def add_random(code, neighs, covered):
        while True:
            w = [random.randrange(q) for _ in range(n)]
            if w not in code:
                nw = tuple(w)
                if nw not in [tuple(c) for c in code]:
                    code.append(w)
                    ns = _all_neighbors(nw, q, n, r)
                    neighs.append(ns)
                    covered |= ns
                    return len(code) - 1

    # Use a set for membership of code words
    code_set = set(tuple(c) for c in code)

    def add_random2(code, neighs, covered, code_set):
        while True:
            w = [random.randrange(q) for _ in range(n)]
            tw = tuple(w)
            if tw not in code_set:
                code.append(w)
                code_set.add(tw)
                ns = _all_neighbors(tw, q, n, r)
                neighs.append(ns)
                covered |= ns
                return len(code) - 1

    def remove_i(i, code, neighs, covered, code_set):
        code_set.discard(tuple(code[i]))
        code.pop(i)
        neighs.pop(i)
        # recompute covered
        new_cov = set()
        for s in neighs:
            new_cov |= s
        return new_cov

    # Local search loop with deadline
    while time.time() < _DEADLINE:
        # Phase 1: try removing random codewords
        for _ in range(200):
            if time.time() >= _DEADLINE:
                break
            if not code:
                break
            i = random.randrange(len(code))
            new_cov, ok = try_remove(i, code, neighs, covered)
            if ok:
                covered = new_cov
                code_set.discard(tuple(code[i]) if i < len(code) else None)
                # rebuild code_set to be safe
                code_set = set(tuple(c) for c in code)
                if len(code) < best_len:
                    best = list(code)
                    best_len = len(best)

        # Phase 2: if not full, add; then try swaps (add then remove)
        if len(covered) < full_space:
            # add a random word that helps
            for _ in range(50):
                if time.time() >= _DEADLINE:
                    break
                # pick a random uncovered word, add its neighborhood center = itself
                # sample until we find uncovered
                tries = 0
                while tries < 20:
                    w = [random.randrange(q) for _ in range(n)]
                    tw = tuple(w)
                    if tw not in covered:
                        add_random2(code, neighs, covered, code_set)
                        break
                    tries += 1
            # after adding, try removals again
            for _ in range(200):
                if time.time() >= _DEADLINE:
                    break
                if not code:
                    break
                i = random.randrange(len(code))
                new_cov, ok = try_remove(i, code, neighs, covered)
                if ok:
                    covered = new_cov
                    code_set = set(tuple(c) for c in code)
                    if len(code) < best_len:
                        best = list(code)
                        best_len = len(best)
        else:
            # full coverage: do swap moves - remove one, if broken add one, then try to remove more
            for _ in range(50):
                if time.time() >= _DEADLINE:
                    break
                if len(code) <= 1:
                    break
                i = random.randrange(len(code))
                saved_code = list(code)
                saved_neighs = list(neighs)
                saved_covered = set(covered)
                saved_set = set(code_set)
                new_cov, ok = try_remove(i, code, neighs, covered)
                if not ok:
                    # need to add a word to restore coverage
                    missing = set()
                    for s in saved_neighs:
                        missing |= s
                    missing = set(itertools.product(range(q), repeat=n)) - new_cov
                    if not missing:
                        continue
                    # pick a random missing word to add
                    mw = random.choice(list(missing))
                    code.append(list(mw))
                    code_set.add(mw)
                    ns = _all_neighbors(mw, q, n, r)
                    neighs.append(ns)
                    covered = new_cov | ns
                    # now try to remove some words to get back to smaller size
                    for _ in range(30):
                        if time.time() >= _DEADLINE:
                            break
                        j = random.randrange(len(code))
                        nc2, ok2 = try_remove(j, code, neighs, covered)
                        if ok2:
                            covered = nc2
                            code_set = set(tuple(c) for c in code)
                    if len(covered) == full_space and len(code) < best_len:
                        best = list(code)
                        best_len = len(best)
                    # keep the state (it's a valid full cover)
                else:
                    covered = new_cov
                    code_set = set(tuple(c) for c in code)
                    if len(code) < best_len:
                        best = list(code)
                        best_len = len(best)

    # Final validity check on best
    def verify(codewords):
        cov = set()
        for c in codewords:
            cov |= _all_neighbors(tuple(c), q, n, r)
        return len(cov) == full_space

    if not verify(best):
        # fall back to initial
        best = _initial_code(q, n, r)

    # Ensure distinctness
    seen = set()
    out = []
    for c in best:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            out.append(list(c))
    return out
